# Works in 3D too!
from firedrake import *
from firedrake.petsc import PETSc
from impl import SubspaceCorrectionPrec
import sys
import numpy

if len(sys.argv) < 3:
    print "Usage: python subspace_correction.py L order [petsc_options]"

L = int(sys.argv[1])
k = int(sys.argv[2])

M = RectangleMesh(L, L, 2.0, 2.0)
M.coordinates.dat.data[:] -= 1


V = VectorFunctionSpace(M, "CG", k+1)
W = FunctionSpace(M, "CG", k)
VW = V * W

bcs = [DirichletBC(VW.sub(0), Constant((1, 0)), (4, )),  # Top
       DirichletBC(VW.sub(0), Constant((0, 0)), (1, 2, 3))]  # Other sides

u, p = TrialFunctions(VW)
v, q = TestFunctions(VW)

a = inner(grad(u), grad(v))*dx - p*div(v)*dx + div(u)*q*dx
aP = inner(grad(u), grad(v))*dx + p*q*dx

f1 = project(Expression(("0","0")), V)
f2 = project(Expression("0"), W)

ell = inner(f1, v)*dx + f2*q*dx

A = assemble(a, bcs=bcs)
Ap = assemble(aP, bcs=bcs)
b = assemble(ell)


solver = LinearSolver(A, P=None, options_prefix="")

A, P = solver.ksp.getOperators()

P00 = PETSc.Mat().create()
P00.setSizes(((V.dof_dset.size * V.dim, None),
              (V.dof_dset.size * V.dim, None)),
             bsize=(V.dim, V.dim))
P00.setType(P00.Type.PYTHON)

u = TrialFunction(V)
v = TestFunction(V)
p00 = inner(grad(u), grad(v))*dx

pbcs = [DirichletBC(V, bc.function_arg, bc.sub_domain) for bc in bcs]
SCP = SubspaceCorrectionPrec(p00, bcs=pbcs)
P00.setPythonContext(SCP)

P = PETSc.Mat().createNest([[P00, None],
                            [None, Ap.M[1, 1].handle]],
                           isrows=VW.dof_dset.field_ises,
                           iscols=VW.dof_dset.field_ises)

P.setUp()
P.setFromOptions()

solver.ksp.setOperators(A, P)

u = Function(VW, name="solution")

solver.solve(u, b)
