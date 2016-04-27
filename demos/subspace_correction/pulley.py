from firedrake import *
from firedrake.petsc import PETSc
from impl import SubspaceCorrectionPrec
import sys
import numpy

if len(sys.argv) < 2:
    print "Usage: python pulley.py order [petsc_options]"

mesh = Mesh("pulley.msh")

dm = mesh._plex

dm.createLabel("boundary_ids")

sec = dm.getCoordinateSection()
coords = dm.getCoordinates()
def inner_surface(x):
    r = 3.75 - x[2]*0.17
    return (x[0]*x[0] + x[1]*x[1]) < r*r

for f in dm.getStratumIS("exterior_facets", 1).indices:
    p, _ = dm.getTransitiveClosure(f)
    p = p[-3:]
    innerblah = True
    
    for v in p:
        x = dm.vecGetClosure(sec, coords, v)
        if not inner_surface(x):
            innerblah = False
            break
    if innerblah:
        dm.setLabelValue("boundary_ids", f, 1)
    else:
        dm.setLabelValue("boundary_ids", f, 2)

mesh.init()

print "Done initializing the mesh"

# Carry on, subdomain id "1" is on the inner wheel surface
# Subdomain id "2" is the rest of the mesh surface


k = int(sys.argv[1])

V = VectorFunctionSpace(mesh, "CG", k)
bcval = (0, 0, 0)

bcs = DirichletBC(V, bcval, (1,))
u = TrialFunction(V)
v = TestFunction(V)


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

# Rotation rate and mass density
omega = 300.0
rho = 10.0


# Loading due to centripetal acceleration (rho*omega^2*x_i)
fexp = Expression(("rho*omega*omega*x[0]", "rho*omega*omega*x[1]", "0.0"),
                  omega=omega, rho=rho)
f = project(fexp, V)
#f = Function(V)

L = inner(f, v)*dx

u0 = Function(V, name="solution")

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
P.setSizes(*A.getSizes())
P.setType(P.Type.PYTHON)
P.setPythonContext(SCP)
P.setUp()
P.setFromOptions()
solver.ksp.setOperators(A, P)

solver.solve(u0, b)


