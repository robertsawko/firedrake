from __future__ import absolute_import
from firedrake import TestFunction, TrialFunction, DirichletBC, \
    FunctionSpace, VectorFunctionSpace, Function, assemble
from firedrake.utils import cached_property
from firedrake.petsc import PETSc
from pyop2 import op2
from mpi4py import MPI
from . import sscutils
import numpy

import ufl
from ufl.algorithms import map_integrands, MultiFunction
from .patches import get_cell_facet_patches, get_dof_patches, \
    g2l_begin, g2l_end, l2g_begin, l2g_end



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
                                  V.cell_node_map().values_with_halo,
                                  bcs, cells, facets)
        self.bc_nodes = bcs
        self.cells = cells
        self.facets = facets
        self.dof_patches = d
        self.glob_patches = g
        self.bc_patches = b

    @cached_property
    def P1_form(self):
        mapper = ArgumentReplacer(TestFunction(self.P1),
                                  TrialFunction(self.P1))
        return map_integrands.map_integrand_dags(mapper, self.a)

    @cached_property
    def P1_bcs(self):
        bcs = []
        for bc in self.bcs:
            val = Function(self.P1)
            val.interpolate(ufl.as_ufl(bc.function_arg))
            bcs.append(DirichletBC(self.P1, val, bc.sub_domain, method=bc.method))
        return tuple(bcs)

    @cached_property
    def P1_op(self):
        return assemble(self.P1_form, bcs=self.P1_bcs).M.handle

    @cached_property
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

    @cached_property
    def matrix_callable(self):
        return sscutils.matrix_callable(self.kernels, self.V, self.mesh.coordinates,
                                        *self.a.coefficients())

    @cached_property
    def matrices(self):
        mats = []
        dim = self.V.dof_dset.cdim
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

    @cached_property
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


def mpi_type(dtype, dim):
    try:
        tdict = MPI.__TypeDict__
    except AttributeError:
        tdict = MPI._typedict

    btype = tdict[dtype.char]
    if dim == 1:
        return btype
    typ = btype.Create_contiguous(dim)
    typ.Commit()
    return typ


class PatchPC(object):

    def setUp(self, pc):
        with PETSc.Log.Stage("PatchPC setup"):
            A, P = pc.getOperators()
            ctx = P.getPythonContext()
            self.ksps = []
            self.ctx = ctx
            V = ctx.V
            self._mpi_type = mpi_type(numpy.dtype(PETSc.ScalarType), V.dim)
            dm = V._dm
            self._sf = dm.getDefaultSF()

            local = PETSc.Vec().create(comm=PETSc.COMM_SELF)
            size = V.dof_dset.total_size * V.dim
            local.setSizes((size, size), bsize=V.dim)
            local.setUp()
            self._local = local

            # Now the patch vectors:
            self._bs = []
            self._ys = []
            self._bcs = []
            self._vscats = []
            for i, m in enumerate(ctx.matrices):
                ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
                pfx = pc.getOptionsPrefix()
                ksp.setOptionsPrefix(pfx + "sub_")
                ksp.setType(ksp.Type.PREONLY)
                ksp.setOperators(m, m)
                ksp.setFromOptions()
                self.ksps.append(ksp)
                size = (ctx.glob_patches[i].shape[0])*V.dim
                b = PETSc.Vec().create(comm=PETSc.COMM_SELF)
                b.setSizes((size, size), bsize=V.dim)
                b.setUp()
                indices = ctx.glob_patches[i]
                vscat = PETSc.Scatter().create(self._local,
                                               PETSc.IS().createBlock(V.dim,
                                                                      indices,
                                                                      comm=PETSc.COMM_SELF),
                                               b, None)
                self._vscats.append(vscat)
                self._bs.append(b)
                self._ys.append(b.duplicate())
                self._bcs.append(numpy.zeros((len(ctx.bc_patches[i]), V.dim), dtype=PETSc.ScalarType))

    def view(self, pc, viewer=None):
        if viewer is not None:
            comm = viewer.comm
        else:
            comm = pc.comm

        PETSc.Sys.Print("Vertex-patch preconditioner, all subsolves identical", comm=comm)
        self.ksps[0].view(viewer)

    def apply(self, pc, x, y):
        # x.copy(y)
        y.set(0)
        with PETSc.Log.Stage("PatchPC apply"):
            # Apply y <- PC(x)
            ctx = self.ctx
            local = self._local
            local.set(0)

            sf = self._sf
            mtype = self._mpi_type
            # Can't use DMGlobalToLocal because we need to pass non-scalar MPI_Type.
            g2l_begin(sf, x, local, mtype)
            g2l_end(sf, x, local, mtype)
            for i, m in enumerate(ctx.matrices):
                ly = self._ys[i]
                b = self._bs[i]
                vscat = self._vscats[i]
                vscat.begin(local, b, addv=PETSc.InsertMode.INSERT_VALUES,
                            mode=PETSc.ScatterMode.FORWARD)
                vscat.end(local, b, addv=PETSc.InsertMode.INSERT_VALUES,
                          mode=PETSc.ScatterMode.FORWARD)
                bcdofs = ctx.bc_patches[i]
                # Zero boundary values
                # TODO: Condense them out!
                b.setValuesBlocked(bcdofs, self._bcs[i],
                                   addv=PETSc.InsertMode.INSERT_VALUES)
                self.ksps[i].solve(b, ly)

            local.set(0)
            for i, ly in enumerate(self._ys):
                vscat = self._vscats[i]
                vscat.begin(ly, local, addv=PETSc.InsertMode.ADD_VALUES,
                            mode=PETSc.ScatterMode.REVERSE)
                vscat.end(ly, local, addv=PETSc.InsertMode.ADD_VALUES,
                          mode=PETSc.ScatterMode.REVERSE)
            l2g_begin(sf, local, y, mtype)
            l2g_end(sf, local, y, mtype)
            y.array.reshape(-1, self.ctx.V.dim)[self.ctx.bc_nodes] = x.array_r.reshape(-1, self.ctx.V.dim)[self.ctx.bc_nodes]


class P1PC(object):

    def setUp(self, pc):
        with PETSc.Log.Stage("P1PC setup"):
            self.pc = PETSc.PC().create()
            self.pc.setOptionsPrefix(pc.getOptionsPrefix() + "lo_")
            A, P = pc.getOperators()
            ctx = P.getPythonContext()
            self.ctx = ctx
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
        with PETSc.Log.Stage("P1PC apply"):
            y.set(0)
            self.work1.set(0)
            self.work2.set(0)
            self.transfer.mult(x, self.work1)
            self.pc.apply(self.work1, self.work2)
            self.transfer.multTranspose(self.work2, y)
            y.array.reshape(-1, self.ctx.V.dim)[self.ctx.bc_nodes] = x.array_r.reshape(-1, self.ctx.V.dim)[self.ctx.bc_nodes]