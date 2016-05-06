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

E = 1.0e9
nu = 0.3
mu = E/(2.0*(1.0 + nu))
lmbda = E*nu/((1.0 + nu)*(1.0 - 2.0*nu))

# Stress computation
def sigma(v):
    return 2.0*mu*sym(grad(v)) + lmbda*tr(sym(grad(v)))*Identity(len(v))


# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
a = inner(sigma(u), grad(v))*dx

f = Function(V)
from numpy import random
f.dat.data[:,:] = random.randn(*f.dat.data.shape)
L = inner(f, v)*dx

u = Function(V, name="solution")

import time

start = time.time()
SCP = SubspaceCorrectionPrec(a, bcs=bcs)

print 'making patches took', time.time() - start
numpy.set_printoptions(linewidth=200, precision=5, suppress=True)

A = assemble(a, bcs=bcs)

b = assemble(L)

solver = LinearSolver(A, options_prefix="")

A, P = solver.ksp.getOperators()

# Need to remove this bit if don't use python pcs
P = PETSc.Mat().create()
P.setSizes(A.getSizes(), bsize=A.getBlockSizes())
P.setType(P.Type.PYTHON)
P.setPythonContext(SCP)
P.setUp()
P.setFromOptions()
solver.ksp.setOperators(A, P)

u = Function(V, name="solution")

with PETSc.Log.Stage("Make patch problems"):
    with PETSc.Log.Event("Total time"):
        SCP.matrices

with PETSc.Log.Stage("SCP Solve"):
    solver.solve(u, b)


