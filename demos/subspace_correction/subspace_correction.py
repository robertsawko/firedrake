from __future__ import absolute_import
from firedrake import *
from firedrake import utils
from firedrake.petsc import PETSc
from impl import sscutils
import numpy

import ufl
from ufl.algorithms import map_integrands, MultiFunction
from impl.patches import get_cell_facet_patches, get_dof_patches


class ArgumentReplacer(MultiFunction):
    def __init__(self, test, trial):
        self.args = {0: test, 1: trial}
        super(ArgumentReplacer, self).__init__()

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
        mesh = a.ufl_domain()
        self.mesh = mesh
        test, trial = a.arguments()
        V = test.function_space()
        assert V == trial.function_space()
        self.V = V
        if V.rank == 0:
            self.P1 = FunctionSpace(mesh, "CG", 1)
        elif V.rank == 1:
            assert len(V.shape) == 1
            self.P1 = VectorFunctionSpace(mesh, "CG", 1, dim=V.shape[0])
        else:
            raise NotImplementedError

        if bcs is None:
            self.bcs = ()
            bcs = numpy.zeros(0, dtype=numpy.int32)
        else:
            try:
                bcs = tuple(bcs)
            except TypeError:
                bcs = (bcs, )
            self.bcs = bcs
            bcs = numpy.unique(numpy.concatenate([bc.nodes for bc in bcs]))

        dof_section = V._dm.getDefaultSection()
        dm = mesh._plex
        cells, facets = get_cell_facet_patches(dm, mesh._cell_numbering)
        d, g, b = get_dof_patches(dm, dof_section,
                                  V.cell_node_map().values,
                                  bcs, cells, facets)
        self.cells = cells
        self.facets = facets
        self.dof_patches = d
        self.glob_patches = g
        self.bc_patches = b

    @utils.cached_property
    def P1_form(self):
        mapper = ArgumentReplacer(TestFunction(self.P1),
                                  TrialFunction(self.P1))
        return map_integrands.map_integrand_dags(mapper, self.a)

    @utils.cached_property
    def P1_bcs(self):
        bcs = []
        for bc in self.bcs:
            val = Function(self.P1)
            val.interpolate(as_ufl(bc.function_arg))
            bcs.append(DirichletBC(self.P1, val, bc.sub_domain, method=bc.method))
        return tuple(bcs)

    @utils.cached_property
    def P1_op(self):
        return assemble(self.P1_form, bcs=self.P1_bcs).M.handle

    @utils.cached_property
    def kernels(self):
        from firedrake.tsfc_interface import compile_form
        kernels = compile_form(self.a, "subspace_form")
        compiled_kernels = []
        for k in kernels:
            # Don't want to think about mixed yet
            assert k.indices == (0, 0)
            kinfo = k.kinfo
            assert kinfo.integral_type == "cell"
            assert not kinfo.oriented
            compiled_kernels.append(kinfo)
        assert len(compiled_kernels) == 1
        return tuple(compiled_kernels)

    @utils.cached_property
    def matrix_callable(self):
        return sscutils.matrix_callable(self.kernels, self.V, self.mesh.coordinates,
                                        *self.a.coefficients())

    @utils.cached_property
    def matrices(self):
        mats = []
        dim = V.dof_dset.cdim
        coords = self.mesh.coordinates
        carg = coords.dat._data.ctypes.data
        cmap = coords.cell_node_map()._values.ctypes.data
        coeffs = self.a.coefficients()
        args = []
        for n in self.kernels[0].coefficient_map:
            c = coeffs[n]
            args.append(c.dat._data.ctypes.data)
            args.append(c.cell_node_map()._values.ctypes.data)
        for i in range(len(self.dof_patches.offset) - 1):
            mat = PETSc.Mat().create(comm=PETSc.COMM_SELF)
            size = (self.glob_patches.offset[i+1] - self.glob_patches.offset[i])*dim
            mat.setSizes(((size, size), (size, size)),
                         bsize=dim)
            mat.setType(mat.Type.DENSE)
            mat.setOptionsPrefix("scp_")
            mat.setFromOptions()
            mat.setUp()
            marg = mat.handle
            mmap = self.dof_patches.value[self.dof_patches.offset[i]:].ctypes.data
            cells = self.cells.value[self.cells.offset[i]:].ctypes.data
            end = self.cells.offset[i+1] - self.cells.offset[i]
            self.matrix_callable(0, end, cells, marg, mmap, mmap, carg, cmap, *args)
            mat.assemble()
            rows = self.bc_patches.value[self.bc_patches.offset[i]:self.bc_patches.offset[i+1]]
            rows = numpy.dstack([dim*rows + i for i in range(dim)]).flatten()
            mat.zeroRowsColumns(rows)
            mats.append(mat)
        return tuple(mats)

    def transfer_kernel(self, restriction=True):
        """Compile a kernel that will map between Pk and P1.

        :kwarg restriction: If True compute a restriction operator, if
             False, a prolongation operator.
        :returns: a PyOP2 kernel.

        The prolongation maps a solution in P1 into Pk using the natural
        embedding.  The restriction maps a residual in the dual of Pk into
        the dual of P1 (it is the dual of the prolongation), computed
        using linearity of the test function.
        """
        # Mapping of a residual in Pk into a residual in P1
        from coffee import base as coffee
        from tsfc.coffee import generate as generate_coffee, SCALAR_TYPE
        from tsfc.kernel_interface import prepare_coefficient, prepare_arguments
        from gem import gem, impero_utils as imp
        import ufl
        import numpy

        Pk = self.V
        P1 = self.P1
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

        funargs = []
        Pke = Vin.fiat_element
        P1e = Vout.fiat_element

        assert Vin.shape == Vout.shape

        shape = (P1e.space_dimension(), ) + Vout.shape + (Pke.space_dimension(), ) + Vin.shape

        outarg = coffee.Decl(SCALAR_TYPE, coffee.Symbol("A", rank=shape))
        i = gem.Index()
        j = gem.Index()
        pre = [i]
        post = [j]
        extra = []
        for _ in Vin.shape:
            extra.append(gem.Index())
        indices = pre + extra + post + extra

        indices = tuple(indices)
        outgem = [gem.Indexed(gem.Variable("A", shape), indices)]

        funargs.append(outarg)

        exprs = [gem.Indexed(weights, (i, j))]

        ir = imp.compile_gem(outgem, exprs, indices)

        body = generate_coffee(ir, {})
        function = coffee.FunDecl("void", name, funargs, body,
                                  pred=["static", "inline"])

        return op2.Kernel(function, name=function.name)

    @utils.cached_property
    def transfer_op(self):
        sp = op2.Sparsity((self.P1.dof_dset,
                           self.V.dof_dset),
                          (self.P1.cell_node_map(),
                           self.V.cell_node_map()),
                          "P1_Pk_mapper")
        mat = op2.Mat(sp, PETSc.ScalarType)
        matarg = mat(op2.WRITE, (self.P1.cell_node_map(self.P1_bcs)[op2.i[0]],
                                 self.V.cell_node_map(self.bcs)[op2.i[1]]))
        # HACK HACK HACK, this seems like it might be a pyop2 bug
        sh = matarg._block_shape
        assert len(sh) == 1 and len(sh[0]) == 1 and len(sh[0][0]) == 2
        a, b = sh[0][0]
        nsh = (((a*self.P1.dof_dset.cdim, b*self.V.dof_dset.cdim), ), )
        matarg._block_shape = nsh
        op2.par_loop(self.transfer_kernel(), self.mesh.cell_set,
                     matarg)
        mat.assemble()
        mat._force_evaluation()
        return mat.handle


class PatchPC(object):

    def setUp(self, pc):
        A, P = pc.getOperators()
        ctx = P.getPythonContext()
        ksp = PETSc.KSP().create()
        pfx = pc.getOptionsPrefix()
        ksp.setOptionsPrefix(pfx + "sub_")
        ksp.setType(ksp.Type.PREONLY)
        ksp.setFromOptions()
        self.ksp = ksp
        self.ctx = ctx

    def view(self, pc, viewer=None):
        if viewer is not None:
            comm = viewer.comm
        else:
            comm = pc.comm

        PETSc.Sys.Print("Vertex-patch preconditioner, all subsolves identical", comm=comm)
        self.ksp.view(viewer)

    def apply(self, pc, x, y):
        y.set(0)
        # Apply y <- PC(x)
        tmp_ys = []
        ctx = self.ctx
        bsize = ctx.V.dim
        for i, m in enumerate(ctx.matrices):
            self.ksp.reset()
            self.ksp.setOperators(m, m)
            ly, b = m.createVecs()
            ly.set(0)
            b.set(0)
            patch_dofs = ctx.glob_patches.value[ctx.glob_patches.offset[i]:ctx.glob_patches.offset[i+1]]
            bc_dofs = ctx.bc_patches.value[ctx.bc_patches.offset[i]:ctx.bc_patches.offset[i+1]]
            b.array.reshape(-1, bsize)[:] = x.array_r.reshape(-1, bsize)[patch_dofs]
            b.array.reshape(-1, bsize)[bc_dofs] = 0
            self.ksp.solve(b, ly)
            tmp_ys.append(ly)

        for i, ly in enumerate(tmp_ys):
            patch_dofs = ctx.glob_patches.value[ctx.glob_patches.offset[i]:ctx.glob_patches.offset[i+1]]
            y.array.reshape(-1, bsize)[patch_dofs] += ly.array_r.reshape(-1, bsize)[:]


class P1PC(object):

    def setUp(self, pc):
        self.pc = PETSc.PC().create()
        self.pc.setOptionsPrefix(pc.getOptionsPrefix() + "lo_")
        A, P = pc.getOperators()
        ctx = P.getPythonContext()
        op = ctx.P1_op
        self.pc.setOperators(op, op)
        self.pc.setUp()
        self.pc.setFromOptions()
        self.transfer = ctx.transfer_op
        self.work1 = self.transfer.createVecLeft()
        self.work2 = self.transfer.createVecLeft()

    def view(self, pc, viewer=None):
        if viewer is not None:
            comm = viewer.comm
        else:
            comm = pc.comm

        PETSc.Sys.Print("Low-order P1, inner pc follows", comm=comm)
        self.pc.view(viewer)

    def apply(self, pc, x, y):
        y.set(0)
        self.work1.set(0)
        self.work2.set(0)
        self.transfer.mult(x, self.work1)
        self.pc.apply(self.work1, self.work2)
        self.transfer.multTranspose(self.work2, y)

# Works in 3D too!

import sys

if len(sys.argv) < 3:
    print "Usage: python subspace_correction.py L order [petsc_options]"
L = int(sys.argv[1])
k = int(sys.argv[2])
M = RectangleMesh(L, L, 2.0, 2.0)
M.coordinates.dat.data[:] -= 1
scalar = False
if scalar:
    V = FunctionSpace(M, "CG", k)
    bcval = 0
else:
    V = VectorFunctionSpace(M, "CG", k)
    bcval = (0, 0)

bcs = DirichletBC(V, bcval, (1, 2, 3, 4)) # , 5, 6))
u = TrialFunction(V)
v = TestFunction(V)
eps = 1
C = as_tensor([[1, 0],
               [0, eps]])
a = inner(C*grad(u), grad(v))*dx
coords = SpatialCoordinate(M)
x = variable(coords[0])
y = variable(coords[1])
sx = sin(pi*x)
sy = sin(pi*y)
cx = cos(pi*x)
cy = cos(pi*y)
xx = x*x
yy = y*y

if scalar:
    exact_expr = sin(pi*x)*sin(pi*y)*exp(-10*(xx + yy))
else:
    exact_expr = as_vector([sin(pi*x)*sin(pi*y)*exp(-10*(xx + yy)), 0])

forcing = -(diff(diff(exact_expr, x), x) + diff(diff(exact_expr, y), y))

L = inner(forcing, v)*dx
u = Function(V, name="solution")

import time

start = time.time()
SCP = SubspaceCorrectionPrec(a, bcs=bcs)

print 'making patches took', time.time() - start
numpy.set_printoptions(linewidth=200, precision=5, suppress=True)

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


A = assemble(a, bcs=bcs)

b = assemble(L)

solver = LinearSolver(A, options_prefix="")

A, P = solver.ksp.getOperators()

# Need to remove this bit if don't use python pcs
P = PETSc.Mat().create()
P.setSizes(*A.getSizes())
P.setType(P.Type.PYTHON)
P.setPythonContext(SCP)
P.setUp()
P.setFromOptions()
solver.ksp.setOperators(A, P)

u = Function(V, name="solution")

solver.solve(u, b)

exact = Function(V).interpolate(exact_expr)
diff = assemble(exact - u)
print norm(diff)
