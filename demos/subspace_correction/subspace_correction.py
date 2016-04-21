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
from impl.patches import get_cell_facet_patches, get_dof_patches


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
        # if bcs is None:
        #     bcs = ()
        # try:
        #     bcs = tuple(bcs)
        # except TypeError:
        #     bcs = (bcs, )
        # nodes = set()
        # for bc in bcs:
        #     nodes.update(bc.nodes)
        # self.bc_nodes = nodes

        # # one phase of the preconditioner involves restricting the problem
        # # to the patch around each vertex and solving it.  So, we'll need
        # # to grab that information from the mesh's plex object
        # mesh = a.ufl_domain()
        # self.mesh = mesh

        # test, trial = a.arguments()

        # V = test.function_space()
        # assert V == trial.function_space()

        # dof_section = V._dm.getDefaultSection()
        # dm = mesh._plex

        # # This includes halo vertices, we might need to filter some out
        # vstart, vend = dm.getDepthStratum(0)

        # # range for cells
        # cstart, cend = dm.getHeightStratum(0)

        # patches = []

        # patch_faces = []
        # # section from plex cells to firedrake cell numbers
        # cell_numbering = mesh._cell_numbering
        # # # section for plex vertices to firedrake vertices
        # # vtx_numbering = mesh._vertex_numbering
        # for v in range(vstart, vend):
        #     closure, orientation = dm.getTransitiveClosure(v, useCone=False)
        #     cells = closure[numpy.logical_and(cstart <= closure, closure < cend)]
        #     # find faces that are on boundary of cell patch
        #     scells = set(cells)
        #     boundary_faces = []
        #     for c in cells:
        #         faces = dm.getCone(c)
        #         for f in faces:
        #             # Only select faces if they are not on the domain boundary
        #             if dm.getLabelValue("exterior_facets", f) == 1:
        #                 continue
        #             f_cells = set(dm.getSupport(f))
        #             if len(f_cells.difference(scells)) > 0:
        #                 # One of the cells is not in our patch.
        #                 boundary_faces.append(f)
        #     patch_faces.append(boundary_faces)
        #     # Both of the vertices and cells are in plex numbering,
        #     patches.append(numpy.array([cell_numbering.getOffset(c)
        #                                 for c in cells], dtype=numpy.int32))

        # # Have a functionspace V
        # cell_node_map = V.cell_node_map().values
        # # shape (ncell, ndof_per_cell)

        # dof_patches = []
        # glob_patches = []
        # bc_masks = []
        # from functools import partial
        # for patch, faces in zip(patches, patch_faces):
        #     local = collections.defaultdict(partial(next, itertools.count()))
        #     dof_patch = numpy.empty((len(patch), cell_node_map.shape[-1]),
        #                             dtype=numpy.int32)
        #     bc_mask = numpy.zeros(dof_patch.shape, dtype=bool)
        #     for i, c in enumerate(patch):
        #         for j, dof in enumerate(cell_node_map[c, :]):
        #             dof_patch[i, j] = local[dof]
        #             # Mask out global dirichlet bcs
        #             if dof in self.bc_nodes:
        #                 bc_mask[i, j] = True
        #     glob_patch = numpy.empty(dof_patch.max() + 1, dtype=numpy.int32)
        #     for i, j in numpy.ndindex(dof_patch.shape):
        #         glob_patch[dof_patch[i, j]] = cell_node_map[patch[i], j]

        #     # Mask out dofs on boundary of patch
        #     # These are the faces on the boundary that are *not* on
        #     # the global domain boundary.
        #     for f in faces:
        #         closure, _ = dm.getTransitiveClosure(f, useCone=True)
        #         for p in closure:
        #             off = dof_section.getOffset(p)
        #             for j in range(dof_section.getDof(p)):
        #                 bc_mask[numpy.where(dof_patch == local[off + j])] = True

        #     glob_patches.append(glob_patch)
        #     dof_patches.append(dof_patch)
        #     bc_masks.append(numpy.unique(dof_patch[bc_mask]))

        # self.patches = patches
        # self.dof_patches = dof_patches
        # self.glob_patches = glob_patches
        # self.bc_patches = bc_masks
        # self.patch_faces = patch_faces

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
        mat = PETSc.Mat().create(comm=PETSc.COMM_SELF)
        mat.setSizes(((nrows, nrows), (ncols, ncols)),
                     bsize=1)
        mat.setType(mat.Type.DENSE)
        mat.setOptionsPrefix("scp_")
        mat.setFromOptions()
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

import sys

L = int(sys.argv[1])
k = int(sys.argv[2])
M = RectangleMesh(L, L, 2.0, 2.0)
M.coordinates.dat.data[:] -= 1
V = FunctionSpace(M, "CG", k)
bcs = DirichletBC(V, 0, (1, 2, 3, 4)) # , 5, 6))
u = TrialFunction(V)
v = TestFunction(V)
a = inner(grad(u), grad(v))*dx
coords = SpatialCoordinate(M)
x = variable(coords[0])
y = variable(coords[1])
sx = sin(pi*x)
sy = sin(pi*y)
cx = cos(pi*x)
cy = cos(pi*y)
xx = x*x
yy = y*y
exact_expr = sin(pi*x)*sin(pi*y)*exp(-10*(xx + yy))

forcing = -(diff(diff(exact_expr, x), x) + diff(diff(exact_expr, y), y))

L = forcing*v*dx
u = Function(V, name="solution")

import time

start = time.time()
SCP = SubspaceCorrectionPrec(a, bcs=bcs)

SCP.compile_kernels()

cells, facets = get_cell_facet_patches(M._plex, M._cell_numbering)
dof_patches, glob_patches, bc_patches = get_dof_patches(M._plex, V._dm.getDefaultSection(),
                                                        V.cell_node_map().values,
                                                        bcs.nodes,
                                                        cells,
                                                        facets)

print 'making patches took', time.time() - start
# from IPython import embed; embed()
# import sys
# sys.exit(0)
# build the patch matrices
matrices = []
for i in range(len(glob_patches.offset) - 1):
    size = glob_patches.offset[i+1] - glob_patches.offset[i]
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
for i in range(len(dof_patches.offset) -1):
    cell = cells.value[cells.offset[i]:].ctypes.data
    end = cells.offset[i+1] - cells.offset[i]
    matarg = matrices[i].handle.handle
    matmap = dof_patches.value[dof_patches.offset[i]:].ctypes.data
    callable(0, end, cell, matarg, matmap, matmap, coordarg, coordmap)
    matrices[i].handle.assemble()
    # matrices[i].handle.view()
    matrices[i].handle.zeroRowsColumns(bc_patches.value[bc_patches.offset[i]:bc_patches.offset[i+1]])
    # print "\n\n"


class PatchPC(object):

    def setUp(self, pc):
        self.ksps = [PETSc.KSP().create() for _ in matrices]
        self.vecs = []
        pfx = pc.getOptionsPrefix()
        for ksp, mat in zip(self.ksps, matrices):
            ksp.setOperators(mat.handle, mat.handle)
            ksp.setOptionsPrefix(pfx + "sub_")
            ksp.setFromOptions()
            self.vecs.append(mat.handle.createVecs())

    def view(self, pc, viewer=None):
        if viewer is not None:
            comm = viewer.comm
        else:
            comm = pc.comm

        PETSc.Sys.Print("Vertex-patch preconditioner, all subsolves identical", comm=comm)
        self.ksps[0].view(viewer)

    def apply(self, pc, x, y):
        y.set(0)
        # Apply y <- PC(x)
        for i in range(len(glob_patches.offset) - 1):
            ksp = self.ksps[i]
            lx, b = self.vecs[i]
            patch_dofs = glob_patches.value[glob_patches.offset[i]:glob_patches.offset[i+1]]
            bc = bc_patches.value[bc_patches.offset[i]:bc_patches.offset[i+1]]
            b.array[:] = x.array_r[patch_dofs]
            # Homogeneous dirichlet bcs on patch boundary
            # FIXME: Condense bcs nodes out of system entirely
            b.array[bc] = 0
            ksp.solve(b, lx)
        for i in range(len(glob_patches.offset) - 1):
            ly = self.vecs[i][0]
            patch_dofs = glob_patches.value[glob_patches.offset[i]:glob_patches.offset[i+1]]
            y.array[patch_dofs] += ly.array_r[:]


numpy.set_printoptions(linewidth=200, precision=4)

# # diag = A.M.values.diagonal()
# # print residual.dat.data_ro / diag

# exact = Function(V)
# solve(a == L, exact, bcs=bcs)

# print norm(assemble(exact - u))
# print u.dat.data_ro
# Pk -> P1 restriction on reference element
# np.dot(np.dot(P2.dual.to_riesz(P1.get_nodal_basis()), P1.get_coeffs().T).T, P2_residual)
# Generally:
# np.linalg.solve(Pkmass, PkP1mass)


def get_transfer_kernel(Pk, P1, restriction=True,
                        matrix_kernel=True):
    """Compile a kernel that will map between Pk and P1.

    :arg Pk: The high order Pk space (a FunctionSpace).
    :arg P1: The P1 space on the same mesh (a FunctionSpace).
    :kwarg restriction: If True compute a restriction operator, if
         False, a prolongation operator.
    :kwarg matrix_kernel: If True, return an operator that computes a
         element matrix, otherwise an operator to do the transfer
         matrix free.
    :returns: a COFFEE FunDecl object.

    The prolongation maps a solution in P1 into Pk using the natural
    embedding.  The restriction maps a residual in the dual of Pk into
    the dual of P1 (it is the dual of the prolongation), computed
    using linearity of the test function.
    """
    # Mapping of a residual in Pk into a residual in P1
    from coffee import base as coffee
    from tsfc.coffee import generate as generate_coffee
    from tsfc.kernel_interface import prepare_coefficient, prepare_arguments
    from gem import gem, impero_utils as imp
    import ufl
    import numpy

    # Pk should be at least the same size as P1
    assert Pk.fiat_element.space_dimension() >= P1.fiat_element.space_dimension()
    # In the general case we should compute this by doing:
    # numpy.linalg.solve(Pkmass, PkP1mass)
    matrix = numpy.dot(Pk.fiat_element.dual.to_riesz(P1.fiat_element.get_nodal_basis()),
                       P1.fiat_element.get_coeffs().T).T

    if restriction:
        Vout, Vin = P1, Pk
        weights = gem.Literal(matrix)
        name = "Pk_P1_mapper"
    else:
        # Prolongation
        Vout, Vin = Pk, P1
        weights = gem.Literal(matrix.T)
        name = "P1_Pk_mapper"

    i = gem.Index("i")
    j = gem.Index("j")

    funargs = []
    if matrix_kernel:
        outarg, prepare, outgem, finalise = prepare_arguments((ufl.TestFunction(Vout),
                                                               ufl.TrialFunction(Vin)),
                                                              (i, j), False)
    else:
        outarg, prepare, outgem, finalise = prepare_arguments((ufl.TestFunction(Vout), )
                                                              (i, ), False)
    assert len(prepare) == 0
    assert len(finalise) == 0
    funargs.append(outarg)

    if matrix_kernel:
        exprs = [gem.Indexed(weights, (i, j))]
    else:
        inarg, prepare, ingem = prepare_coefficient(ufl.Coefficient(Vin), "input")
        assert len(prepare) == 0
        funargs.append(inarg)
        exprs = [gem.IndexSum(gem.Product(gem.Indexed(weights, (i, j)),
                                          gem.Indexed(ingem, (j, ))), j)]

    ir = imp.compile_gem(outgem, exprs, (i, j))

    body = generate_coffee(ir, {i: i.name, j: j.name})
    function = coffee.FunDecl("void", name, funargs, body,
                              pred=["static", "inline"])

    return op2.Kernel(function, name=function.name)

P1 = FunctionSpace(M, "CG", 1)


u = TestFunction(P1)
v = TrialFunction(P1)
low = inner(grad(u), grad(v))*dx
lowbc = DirichletBC(P1, 0, (1, 2, 3, 4))

lo_op = assemble(low, bcs=lowbc).M.handle

sp = op2.Sparsity((P1.dof_dset,
                   V.dof_dset),
                  (P1.cell_node_map(),
                   V.cell_node_map()),
                  "mapper")
mat = op2.Mat(sp, PETSc.ScalarType)

op2.par_loop(get_transfer_kernel(V, P1), M.cell_set,
             mat(op2.WRITE, (P1.cell_node_map([lowbc])[op2.i[0]],
                             V.cell_node_map([bcs])[op2.i[1]])))

mat.assemble()
mat._force_evaluation()
transfer = mat.handle


# Needs to be composed with something else to do the high order
# smoothing.  Otherwise we get into a situation where the residual in
# the P1 space is zero but there are still some high frequency
# components.  The idea is that we use the patch-based smoother above.
# For example:
# python subspace_correction.py  \
#     -ksp_type cg \
#     -ksp_monitor_true_residual \
#     -pc_type composite \
#     -pc_composite_pcs jacobi,python \
#     -pc_composite_type additive \
#     -sub_0_pc_type jacobi \
#     -sub_1_pc_type python \
#     -sub_1_pc_python_type __main__.P1PC  \
#     -sub_1_lo_pc_type hypre
#
# Multiplicative doesn't work nearly as well as additive, need to understand why.
# Example composed with the Patch PC.
# python subspace_correction.py \
#     -ksp_type cg \
#     -ksp_monitor_true_residual \
#     -pc_type composite \
#     -pc_composite_type additive \
#     -pc_composite_pcs python,python \
#     -sub_0_pc_python_type __main__.PatchPC \
#     -sub_0_sub_ksp_type preonly \
#     -sub_0_sub_pc_type lu \
#     -sub_1_pc_python_type __main__.P1PC \
#     -sub_1_lo_pc_type lu
class P1PC(object):

    def setUp(self, pc):
        self.pc = PETSc.PC().create()
        self.pc.setOptionsPrefix(pc.getOptionsPrefix() + "lo_")
        self.pc.setOperators(lo_op, lo_op)
        self.pc.setUp()
        self.pc.setFromOptions()

    def view(self, pc, viewer=None):
        if viewer is not None:
            comm = viewer.comm
        else:
            comm = pc.comm

        PETSc.Sys.Print("Low-order P1, inner pc follows", comm=comm)
        self.pc.view(viewer)

    def apply(self, pc, x, y):
        l = transfer.createVecLeft()
        transfer.mult(x, l)
        tmp = l.duplicate()
        self.pc.apply(l, tmp)
        transfer.multTranspose(tmp, y)
        # Matrix-free would look a little like this
        # # restrict x to P1
        # low = Function(P1)
        # high = Function(V)
        # with high.dat.vec_ro as v:
        #     x.copy(v)
        # print v.array_r
        # op2.par_loop(get_transfer_kernel(V, P1),
        #              M.cell_set,
        #              low.dat(op2.INC, low.cell_node_map()[op2.i[0]],
        #                      flatten=True),
        #              high.dat(op2.READ, high.cell_node_map(),
        #                       flatten=True))
        # with low.dat.vec as v:
        #     print v.array_r
        #     tmp = v.duplicate()
        #     self.pc.apply(v, tmp)
        #     tmp.copy(v)
        # op2.par_loop(get_transfer_kernel(V, P1, restriction=False),
        #              M.cell_set,
        #              high.dat(op2.WRITE, high.cell_node_map()[op2.i[0]],
        #                       flatten=True),
        #              low.dat(op2.READ, low.cell_node_map(),
        #                      flatten=True))
        # with high.dat.vec_ro as v:
        #     v.copy(y)


A = assemble(a, bcs=bcs)

b = assemble(L)

solver = LinearSolver(A, options_prefix="")
u = Function(V, name="solution")

solver.solve(u, b)

exact = Function(V).interpolate(exact_expr)
diff = assemble(exact - u)
# diff.rename("difference")
# exact.rename("exact")
print norm(diff)

# print u.dat.data_ro
# File("out.pvd").write(diff, exact, u)
# File("out.pvd").write(u)
# print u.dat.data_ro
