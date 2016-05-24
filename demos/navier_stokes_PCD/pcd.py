from firedrake import *
from firedrake.petsc import PETSc

# dummy class for holding context
class Bucket(object):
    pass


# creates the PC matrix object that sets all the
# context for the PCD preconditioner to the Schur complement
# approximation

# future work: can we get Fp as a matrix-free object?
class PCDMat(object):
    def __init__(self, args):
        # args is a dict that contains the Reynolds number
        self.Re = args['Re']

        return

    def createMatrix(self, ctx):
        prb = ctx.problem
        VW = prb.u.function_space()

        W = VW.sub(1)

        p = TrialFunction(W)
        q = TestFunction(W)

        a_p = p*q*dx

        # hack: small perturbation regularizes the matrix, I
        # should be setting an null space here instead.
        alpha = Constant(0.0001)
        k_p = inner(grad(p), grad(q))*dx  + alpha*p*q*dx

        fcps = prb.form_compiler_parameters

        Mp = assemble(a_p, bcs=[],
                      form_compiler_parameters=fcps,
                      nest=False)

        Kp = assemble(k_p, bcs=[],
                      form_compiler_parameters=fcps,
                      nest=False)

        u0 = split(ctx._x)[0]
        f_p = inner(dot(u0, grad(p)),q)*dx  + 1.0/self.Re * inner(grad(p), grad(q))*dx

        Fp = assemble(f_p, bcs=[],
                      form_compiler_parameters=fcps,
                      nest=False)

        Wdim = VW.sub(1).dof_dset.size

        S_hat = PETSc.Mat().create()
        S_hat.setSizes(((Wdim, None), (Wdim, None)))
        S_hat.setType(PETSc.Mat.Type.PYTHON)

       
        self.Mp = Mp
        self.f_p = f_p
        self.Fp = Fp
        self.Kp = Kp

        self.Mp.M._force_evaluation()
        self.Kp.M._force_evaluation()
        
        # the block itself needs these for the preconditioner code
        # to fish the out.
        shat_ctx = Bucket()
        shat_ctx.Fp = Fp
        shat_ctx.Mp = Mp
        shat_ctx.Kp = Kp

        
        S_hat.setPythonContext(shat_ctx)

        F = ctx._jac.M[0,0].handle
        M = PETSc.Mat().createNest([[F, None], [None, S_hat]],
                                   isrows=VW.dof_dset.field_ises,
                                   iscols=VW.dof_dset.field_ises)
        M.setUp()
        M.setFromOptions()

        return M
    
    def updateMatrix(self, P):
        # slurp the current Jacobian out of ctx into the top left block
        # and reassmble the Fp in the lower right block
        assemble(self.f_p, tensor=self.Fp)
        self.Fp.M._force_evaluation()
        P.assemble()
        return

    
# This is a preconditioner *only* for the
# Schur complement block.
class PCDPrec(object):
    
    def setUp(self, pc):
        optpre = pc.getOptionsPrefix()

        shat_ctx = pc.getOperators()[1].getPythonContext()

        Mp = shat_ctx.Mp._M.handle
        Kp = shat_ctx.Kp._M.handle

        MpSolver = PETSc.KSP().create()
        MpSolver.setOperators(Mp, Mp)
        MpSolver.setOptionsPrefix(optpre+"Mp_")
        MpSolver.setOperators(Mp, Mp)
        MpSolver.setUp()
        MpSolver.setFromOptions()
        self.MpSolver = MpSolver

        KpSolver = PETSc.KSP().create()
        KpSolver.setOperators(Kp, Kp)
        KpSolver.setOptionsPrefix(optpre+"Kp_")
        KpSolver.setOperators(Kp, Kp)
        KpSolver.setUp()
        KpSolver.setFromOptions()
        self.KpSolver = KpSolver

        self.tmp = [Mp.createVecRight() for i in (0, 1)]
        
        return

    def apply(self, pc, x, y):
        _, P = pc.getOperators()
        Fp = P.getPythonContext().Fp._M.handle
        
        self.MpSolver.solve(x, self.tmp[0])
        Fp.mult(self.tmp[0], self.tmp[1])
        self.KpSolver.solve(self.tmp[1], y)
