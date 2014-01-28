"""This demo program solves an Advection-Diffusion problem"""

from firedrake import *

def run_advdiff(x, degree=1):
    dt = 0.0001
    mesh = ExtrudedMesh(UnitSquareMesh(2 ** x, 2 ** x), layers=11, layer_height=1.0)
    
    T = FunctionSpace(mesh, "CG", degree)
    V = VectorFunctionSpace(mesh, "CG", degree)
    
    p = TrialFunction(T)
    q = TestFunction(T)
    t = Function(T)
    u = Function(V)
    a = Function(T)

    diffusivity = 0.1

    M = p * q * dx

    adv_rhs = (q * t + dt * dot(grad(q), u) * t) * dx

    D = -dt * diffusivity * dot(grad(q), grad(p)) * dx

    diff = M - 0.5 * D
    diff_rhs = action(M + 0.5 * D, t)

    solve(M == adv_rhs, t)
    solve(D == diff_rhs, t)

    return t
