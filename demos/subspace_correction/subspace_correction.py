from firedrake import *
from firedrake.petsc import PETSc
from pyop2 import base as pyop2
from pyop2 import sequential as seq
import numpy
import collections
import itertools
import functools

import ufl
from ufl.algorithms import map_integrands, MultiFunction


class ReplaceArguments(MultiFunction):
    def __init__(self, test, trial):
        self.args = {0: test, 1: trial}
        super(ReplaceArguments, self).__init__()

    expr = MultiFunction.reuse_if_untouched

    def argument(self, o):
        return self.args[o.number()]


class SubspaceCorrectionPrec(object):
    """Given a bilinear form, constructs a subspace correction preconditioner
    for it.  Currently, this is intended to approximate the solution
    of high-order Lagrange (eventually Bernstein as well)
    discretization by the solution of local problems on each vertex
    patch together with a global low-order discretization.

    :arg a:  A bilinear form defined in UFL
    :arg bcs: Optional strongly enforced boundary conditions
    """

    def __init__(self, a, bcs=None):
        self.a = a
        if bcs is None:
            bcs = ()
        try:
            bcs = tuple(bcs)
        except TypeError:
            bcs = (bcs, )
        nodes = set()
        for bc in bcs:
            nodes.update(bc.nodes)
        self.bc_nodes = nodes

        # one phase of the preconditioner involves restricting the problem
        # to the patch around each vertex and solving it.  So, we'll need
        # to grab that information from the mesh's plex object
        mesh = a.ufl_domain()
        self.mesh = mesh

        test, trial = a.arguments()

        V = test.function_space()
        assert V == trial.function_space()

        dof_section = V._dm.getDefaultSection()
        dm = mesh._plex

        # This includes halo vertices, we might need to filter some out
        vstart, vend = dm.getDepthStratum(0)

        # range for cells
        cstart, cend = dm.getHeightStratum(0)

        patches = []

        patch_faces = []
        # section from plex cells to firedrake cell numbers
        cell_numbering = mesh._cell_numbering
        # # section for plex vertices to firedrake vertices
        # vtx_numbering = mesh._vertex_numbering
        for v in range(vstart, vend):
            closure, orientation = dm.getTransitiveClosure(v, useCone=False)
            cells = closure[numpy.logical_and(cstart <= closure, closure < cend)]
            # find faces that are on boundary of cell patch
            boundary_faces = []
            for c in cells:
                faces = dm.getCone(c)
                for f in faces:
                    # Only select faces if they are not on the domain boundary
                    if dm.getLabelValue("exterior_facets", f) == 1:
                        continue
                    closure, _ = dm.getTransitiveClosure(f, useCone=True)
                    if v not in closure:
                        boundary_faces.append(f)
            patch_faces.append(boundary_faces)
            # Both of the vertices and cells are in plex numbering,
            patches.append(numpy.array([cell_numbering.getOffset(c)
                                        for c in cells], dtype=numpy.int32))

        # Have a functionspace V
        cell_node_map = V.cell_node_map().values
        # shape (ncell, ndof_per_cell)

        dof_patches = []
        glob_patches = []
        bc_masks = []
        from functools import partial
        for patch, faces in zip(patches, patch_faces):
            local = collections.defaultdict(partial(next, itertools.count()))
            dof_patch = numpy.empty((len(patch), cell_node_map.shape[-1]),
                                    dtype=numpy.int32)
            bc_mask = numpy.zeros(dof_patch.shape, dtype=bool)
            for i, c in enumerate(patch):
                for j, dof in enumerate(cell_node_map[c, :]):
                    dof_patch[i, j] = local[dof]
                    # Mask out global dirichlet bcs
                    if dof in self.bc_nodes:
                        bc_mask[i, j] = True
            glob_patch = numpy.empty(dof_patch.max() + 1, dtype=numpy.int32)
            for i, j in numpy.ndindex(dof_patch.shape):
                glob_patch[dof_patch[i, j]] = cell_node_map[patch[i], j]

            # Mask out dofs on boundary of patch
            # These are the faces on the boundary that are *not* on
            # the global domain boundary.
            for f in faces:
                closure, _ = dm.getTransitiveClosure(f, useCone=True)
                for p in closure:
                    off = dof_section.getOffset(p)
                    for j in range(dof_section.getDof(p)):
                        bc_mask[numpy.where(dof_patch == local[off + j])] = True

            glob_patches.append(glob_patch)
            dof_patches.append(dof_patch)
            bc_masks.append(numpy.unique(dof_patch[bc_mask]))

        self.patches = patches
        self.dof_patches = dof_patches
        self.glob_patches = glob_patches
        self.bc_patches = bc_masks

    def P1_operator(self, P1space):
        mapper = ReplaceArguments(TestFunction(P1space),
                                  TrialFunction(P1space))
        return map_integrands.map_integrand_dags(mapper, self.a)

    def compile_kernels(self):
        from firedrake.tsfc_interface import compile_form
        from pyop2.compilation import load
        import ctypes
        kernels = compile_form(self.a, "subspace_form")
        compiled_kernels = []
        for k in kernels:
            # Don't want to think about mixed yet
            assert k.indices == (0, 0)
            kinfo = k.kinfo
            assert kinfo.integral_type == "cell"
            assert not kinfo.oriented
            assert len(kinfo.coefficient_map) == 0

            kernel = kinfo.kernel
            compiled_kernels.append(kernel)
        self.kernels = compiled_kernels


# Fake up some PyOP2 objects so we can abuse the PyOP2 code
# compilation pipeline to get a callable function pointer for
# assembling into a dense matrix.
# FIXME: Not correct for VectorElement yet.
class DenseSparsity(object):
    def __init__(self, nrows, ncols):
        self.shape = (1, 1)
        self._nrows = nrows
        self._ncols = ncols
        self._dims = (((1, 1), ), )
        self.dims = self._dims

    def __getitem__(self, *args):
        return self


class MatArg(seq.Arg):
    def __init__(self, data, map, idx, access, flatten):
        self.data = data
        self._map = map
        self._idx = idx
        self._access = access
        self._flatten = flatten
        self._block_shape = tuple(tuple((mr.arity, mc.arity)
                                        for mc in map[1])
                                  for mr in map[0])
        self.cache_key = None

    def c_addto(self, i, j, buf_name, tmp_name, tmp_decl,
                extruded=None, is_facet=False, applied_blas=False):
        # Override global c_addto to index the map locally rather than globally.
        from pyop2.utils import as_tuple
        maps = as_tuple(self.map, op2.Map)
        nrows = maps[0].split[i].arity
        ncols = maps[1].split[j].arity
        rows_str = "%s + n * %s" % (self.c_map_name(0, i), nrows)
        cols_str = "%s + n * %s" % (self.c_map_name(1, j), ncols)

        if extruded is not None:
            rows_str = extruded + self.c_map_name(0, i)
            cols_str = extruded + self.c_map_name(1, j)

        if is_facet:
            nrows *= 2
            ncols *= 2

        ret = []
        rbs, cbs = self.data.sparsity[i, j].dims[0][0]
        rdim = rbs * nrows
        addto_name = buf_name
        addto = 'MatSetValues'
        if self.data._is_vector_field:
            addto = 'MatSetValuesBlocked'
            if self._flatten:
                if applied_blas:
                    idx = "[(%%(ridx)s)*%d + (%%(cidx)s)]" % rdim
                else:
                    idx = "[%(ridx)s][%(cidx)s]"
                ret = []
                idx_l = idx % {'ridx': "%d*j + k" % rbs,
                               'cidx': "%d*l + m" % cbs}
                idx_r = idx % {'ridx': "j + %d*k" % nrows,
                               'cidx': "l + %d*m" % ncols}
                # Shuffle xxx yyy zzz into xyz xyz xyz
                ret = ["""
                %(tmp_decl)s;
                for ( int j = 0; j < %(nrows)d; j++ ) {
                   for ( int k = 0; k < %(rbs)d; k++ ) {
                      for ( int l = 0; l < %(ncols)d; l++ ) {
                         for ( int m = 0; m < %(cbs)d; m++ ) {
                            %(tmp_name)s%(idx_l)s = %(buf_name)s%(idx_r)s;
                         }
                      }
                   }
                }""" % {'nrows': nrows,
                        'ncols': ncols,
                        'rbs': rbs,
                        'cbs': cbs,
                        'idx_l': idx_l,
                        'idx_r': idx_r,
                        'buf_name': buf_name,
                        'tmp_decl': tmp_decl,
                        'tmp_name': tmp_name}]
                addto_name = tmp_name

            rmap, cmap = maps
            rdim, cdim = self.data.dims[i][j]
            if rmap.vector_index is not None or cmap.vector_index is not None:
                raise NotImplementedError
        ret.append("""%(addto)s(%(mat)s, %(nrows)s, %(rows)s,
                                         %(ncols)s, %(cols)s,
                                         (const PetscScalar *)%(vals)s,
                                         %(insert)s);""" %
                   {'mat': self.c_arg_name(i, j),
                    'vals': addto_name,
                    'addto': addto,
                    'nrows': nrows,
                    'ncols': ncols,
                    'rows': rows_str,
                    'cols': cols_str,
                    'insert': "INSERT_VALUES" if self.access == op2.WRITE else "ADD_VALUES"})
        return "\n".join(ret)


class DenseMat(pyop2.Mat):
    def __init__(self, nrows, ncols):
        mat = PETSc.Mat().createDense(((nrows, nrows), (ncols, ncols)),
                                      bsize=1,
                                      comm=PETSc.COMM_SELF)
        mat.setUp()
        self._sparsity = DenseSparsity(nrows, ncols)
        self.handle = mat
        self.dtype = numpy.dtype(PETSc.ScalarType)

    def __call__(self, access, path, flatten=False):
        path_maps = [arg.map for arg in path]
        path_idxs = [arg.idx for arg in path]
        return MatArg(self, path_maps, path_idxs, access, flatten)


class JITModule(seq.JITModule):
    @classmethod
    def _cache_key(cls, *args, **kwargs):
        # Don't want to cache these anywhere I think.
        return None

# Works in 3D too!
# M = UnitCubeMesh(5, 5, 4)
M = RectangleMesh(10, 10, 2.0, 2.0)
V = FunctionSpace(M, "CG", 2)
bcs = DirichletBC(V, 0, (1, 2, 3, 4, 5, 6))
u = TrialFunction(V)
v = TestFunction(V)
a = inner(grad(u), grad(v))*dx

L = v*dx
u = Function(V)

SCP = SubspaceCorrectionPrec(a, bcs=bcs)

SCP.compile_kernels()

# Build the patch matrices
matrices = []
for glob_patch in SCP.glob_patches:
    size = glob_patch.shape[0]
    matrices.append(DenseMat(size, size))

x = matrices[0]
matarg = x(op2.INC, (u.cell_node_map()[op2.i[0]], v.cell_node_map()[op2.i[1]]))
matarg.position = 0

# No coefficients yet
coordarg = M.coordinates.dat(op2.READ, M.coordinates.cell_node_map(), flatten=True)
coordarg.position = 1
itspace = pyop2.build_itspace([matarg, coordarg], op2.Subset(M.cell_set, [0]))
kernel = SCP.kernels[0]
mod = JITModule(kernel, itspace, matarg, coordarg)
callable = mod._fun

coordarg = M.coordinates.dat._data.ctypes.data
coordmap = M.coordinates.cell_node_map()._values.ctypes.data
for i in range(len(SCP.patches)):
    cells = SCP.patches[i]
    end = cells.shape[0]
    cells = cells.ctypes.data
    matarg = matrices[i].handle.handle
    matmap = SCP.dof_patches[i].ctypes.data
    callable(0, end, cells, matarg, matmap, matmap, coordarg, coordmap)
    matrices[i].handle.assemble()
    # matrices[i].handle.view()
    matrices[i].handle.zeroRowsColumns(SCP.bc_patches[i])
    # print "\n\n"

ksps = [PETSc.KSP().create() for _ in matrices]
vecs = []
for ksp, mat in zip(ksps, matrices):
    ksp.setOperators(mat.handle, mat.handle)
    ksp.setOptionsPrefix("sub_")
    ksp.setFromOptions()
    vecs.append(mat.handle.createVecs())


class PatchPC(object):

    def apply(self, pc, x, y):
        y.set(0)
        # Apply y <- PC(x)
        for ksp, (lx, b), patch_dofs, bc in zip(ksps, vecs, SCP.glob_patches, SCP.bc_patches):
            b.array[:] = x.array_r[patch_dofs]
            # Homogeneous dirichlet bcs on patch boundary
            # FIXME: Condense bcs nodes out of system entirely
            b.array[bc] = 0
            ksp.solve(b, lx)
        for (ly, _), patch_dofs in zip(vecs, SCP.glob_patches):
            y.array[patch_dofs] += ly.array_r[:]



numpy.set_printoptions(linewidth=200, precision=4)

A = assemble(a, bcs=bcs)

b = assemble(L)

solver = LinearSolver(A, options_prefix="")
solver.solve(u, b)

# diag = A.M.values.diagonal()
# print residual.dat.data_ro / diag

exact = Function(V)
solve(a == L, exact, bcs=bcs)

print norm(assemble(exact - u))
# print u.dat.data_ro
# Pk -> P1 restriction on reference element
# np.dot(np.dot(P2.dual.to_riesz(P1.get_nodal_basis()), P1.get_coeffs().T).T, P2_residual)
# Generally:
# np.linalg.solve(Pkmass, PkP1mass)

from tsfc.coffee import generate
from tsfc.kernel_interface import KernelBuilderBase
from gem import gem, impero_utils

output = gem.Variable("A", (3, ))

i = gem.Index("i")
j = gem.Index("j")

P1 = FunctionSpace(M, "CG", 1)

weights = gem.Literal(numpy.dot(V.fiat_element.dual.to_riesz(P1.fiat_element.get_nodal_basis()),
                                P1.fiat_element.get_coeffs().T).T)

input = gem.Variable("residual", (6, 1))

rvalue = gem.IndexSum(gem.Product(gem.Indexed(weights, (i, j)), gem.Indexed(input, (j, 0))), j)

lvalue = gem.Indexed(output, (i, ))

ir = impero_utils.compile_gem([lvalue], [rvalue], [i])

body = generate(ir, {i: i.name, j: j.name})
