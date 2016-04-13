from firedrake import *
import numpy
import collections
import itertools
import functools
from coffee.visitor import Visitor



class RewriteQuals(Visitor):

    def visit_object(self, o, *args, **kwargs):
        return o

    def visit_list(self, o, *args, **kwargs):
        return list(self.visit(e, *args, **kwargs) for e in o)

    visit_Node = Visitor.maybe_reconstruct

    def visit_FunDecl(self, o, *args, **kwargs):
        ops, okwargs = o.operands()
        try:
            ops[4].remove("static")
            ops[4].remove("inline")
        except ValueError:
            pass
        decls = ops[2][1:]
        for d in decls:
            d.typ = "double"
            sym = d.sym
            rank = sym.rank + (1, )
            d.sym = sym.reconstruct(sym.symbol, rank=rank)
        new = o.reconstruct(*ops, **okwargs)
        return new


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
        # section for plex vertices to firedrake vertices
        vtx_numbering = mesh._vertex_numbering
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
                                        for c in cells]))

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
                                    dtype=int)
            bc_mask = numpy.zeros(dof_patch.shape, dtype=bool)
            for i, c in enumerate(patch):
                for j, dof in enumerate(cell_node_map[c, :]):
                    dof_patch[i, j] = local[dof]
                    # Mask out global dirichlet bcs
                    if dof in self.bc_nodes:
                        bc_mask[i, j] = True
            glob_patch = numpy.empty(dof_patch.max() + 1, dtype=int)
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

            mask = numpy.empty(glob_patch.shape, dtype=bool)
            for i, j in numpy.ndindex(dof_patch.shape):
                mask[dof_patch[i, j]] = bc_mask[i, j]
            glob_patches.append(glob_patch)
            dof_patches.append(dof_patch)
            bc_masks.append(mask)

        for p, d, g, bcs in zip(patches, dof_patches, glob_patches, bc_masks):
            print p
            print d
            print g
            print bcs
            print

    def compile_kernels(self):
        from firedrake.tsfc_interface import compile_form
        from pyop2.compilation import load
        import ctypes
        kernels = compile_form(self.a, "subspace_form")
        v = RewriteQuals()
        compiled_kernels = []
        for k in kernels:
            # Don't want to think about mixed yet
            assert k.indices == (0, 0)
            kinfo = k.kinfo
            assert kinfo.integral_type == "cell"
            assert not kinfo.oriented
            assert len(kinfo.coefficient_map) == 0

            kernel = kinfo.kernel
            # Rewrite to remove static inline
            kernel._ast = v.visit(kernel._ast)
            kernel._code = None
            code = kernel.code()
            code = code.replace("static inline ", "#include <math.h>\n")
            fn = load(code, "c", kernel.name, argtypes=[ctypes.c_voidp] * len(kernel._ast.args),
                      restype=None)
            compiled_kernels.append((k, fn))
        self.kernels = compiled_kernels


from pyop2 import base as pyop2
from petsc4py import PETSc

class DenseSparsity(pyop2.Sparsity):
    def __init__(self):
        self.shape = (1, 1)
        pass


class DenseMat(pyop2.Mat):
    def __init__(self, nrows, ncols):
        mat = PETSc.Mat().createDense(((nrows, nrows), (ncols, ncols)),
                                      bsize=1,
                                      comm=PETSc.COMM_SELF)
        mat.setUp()
        self.sparsity = DenseSparsity()
        self.handle = mat

M = RectangleMesh(2, 2, 2.0, 2.0)
V = FunctionSpace(M, "CG", 2)
bcs = DirichletBC(V, 0, (1, 2, 3, 4))
u = TrialFunction(V)
v = TestFunction(V)
a = inner(grad(u), grad(v))*dx

SCP = SubspaceCorrectionPrec(a, bcs=None)

SCP.compile_kernels()

# Pk -> P1 restriction on reference element
# np.dot(np.dot(P2.dual.to_riesz(P1.get_nodal_basis()), P1.get_coeffs().T).T, P2_residual)
# Generally:
# np.linalg.solve(Pkmass, PkP1mass)
# from IPython import embed; embed()
