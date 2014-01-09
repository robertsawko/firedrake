"""This demo program solves Burgers' equation"""

from firedrake import *

def run_burgers(x, degree=2):
    n = 30
    mesh = UnitSquareMesh(n, n)
    V = VectorFunctionSpace(mesh, "CG", degree)
    ic = project(Expression(["sin(2*pi*x[0])", "0"]), V)
    u = Function(ic)
    u_next = Function(V)
    v = TestFunction(V)
    nu = 0.0001
    timestep = 1.0/n
    F = (inner((u_next - u)/timestep,v) + inner(dot(u_next,grad(u_next)),v) + nu*inner(grad(u_next),grad(v)))*dx
    # bc = [DirichletBC(V, Expression([0.0]), 1),
    #       DirichletBC(V, Expression([0.0]), 2)]
    t = 0.0; end = 0.2
    while (t <= end):
        # solve(F == 0, u_next, bc)
        solve(F == 0, u_next)
        u.assign(u_next)
        t += float(timestep)

