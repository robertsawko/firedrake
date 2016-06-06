from firedrake import *
import nlvs
from firedrake.petsc import PETSc
from firedrake.function import Function as FDFunction
import ufl

# Set up all the PDE stuff

M = UnitSquareMesh(100, 100)
V = VectorFunctionSpace(M, "CG", 2)
W = FunctionSpace(M, "CG", 1)
VW = V * W

up = Function(VW)
u, p = split(up)
v, q = TestFunctions(VW)

Re = Constant(50.0)

F = (
    1./Re * inner(grad(u), grad(v)) * dx
    + inner(grad(u)*u, v) * dx
    - p * div(v) * dx
    + div(u) * q * dx
)

bcs = [DirichletBC(VW.sub(0), Constant((1, 0)), (4, )),  # Top
       DirichletBC(VW.sub(0), Constant((0, 0)), (1, 2, 3))]  # Other sides


# Let's create the nonlinear variational problem since Firedrake is
# kind enough to do this for us with some bookkeeping under the hood.
prob = NonlinearVariationalProblem(F, up, bcs = bcs, nest=False)

# Notice the new use of "extra_args" to stuff the Reynolds number
# in.  This dictionary will get passed to the PCD Matrix constructor
# Also, we're setting the options_prefix to '' so that we have a short
# path to set things from an options file.
solver = nlvs.NonlinearVariationalSolver(prob, options_prefix='',
                                         extra_args={'Re': Re})

solver.solve()

# Write the velocity to file.
File("nse.pvd").write(up.split()[0])
