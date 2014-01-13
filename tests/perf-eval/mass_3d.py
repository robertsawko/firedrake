"""This demo program solves a Mass problem."""

from firedrake import *

def run_mass(x, degree=1):
    # Create mesh and define function space
    mesh = UnitCubeMesh(2 ** x, 2 ** x, 2 ** x)
    V = FunctionSpace(mesh, "CG", degree)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)
    a = v * u * dx
    L = v * f * dx

    # Compute solution
    assemble(a)
    assemble(L)
    x = Function(V)
    solve(a == L, x, solver_parameters={'ksp_type': 'cg'})

    return x
