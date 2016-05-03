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

scalar = True
if scalar:
    V = FunctionSpace(M, "CG", k)
    bcval = 1
else:
    V = VectorFunctionSpace(M, "CG", k)
    bcval = (0, 0)

bcs = [DirichletBC(V, bcval, (1, 2, 3, 4))]
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
    exact_expr = sin(pi*x)*sin(pi*y)*exp(-10*(xx + yy)) + bcval
else:
    exact_expr = as_vector([sin(pi*x)*sin(pi*y)*exp(-10*(xx + yy)), 0])

forcing = -(diff(diff(exact_expr, x), x) + diff(diff(exact_expr, y), y))

L = inner(forcing, v)*dx
u = Function(V, name="solution")

SCP = SubspaceCorrectionPrec(a, bcs=bcs)

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

# if PETSc.COMM_WORLD.rank == 0:
#     print repr(SCP.facets)
# Need to remove this bit if don't use python pcs
P = PETSc.Mat().create()
P.setSizes(A.getSizes(), bsize=A.getBlockSizes())
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
