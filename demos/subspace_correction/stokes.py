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

bcval = (1, 0)
bcs = [DirichletBC(VW.sub(0), (1,0), (1,)),
       DirichletBC(VW.sub(0), (0,0), (2,3,4))]

u, p = TrialFunctions(VW)
v, q = TestFunctions(VW)

a = inner(grad(u), grad(v))*dx - p*div(v)*dx + div(u)*q*dx
aP = inner(grad(u), grad(v))*dx + p*q*dx

f1 = project(Expression(("0","0")), V)
f2 = project(Expression("0"), W)

ell = inner(f1, v)*dx + f2*q*dx

up = Function(VW)

solve(a==ell, up)

A = assemble(a, bcs=bcs)
Ap = assemble(aP, bcs=bcs)
b = assemble(ell)


solver = LinearSolver(A, P=Ap, options_prefix="")

A, P = solver.ksp.getOperators()

# Now: I think we need to:
# extract the (0,0) block from the matnest of P
# Set its Python context to SCP (which we need to create)
# set everything else from options?



             

