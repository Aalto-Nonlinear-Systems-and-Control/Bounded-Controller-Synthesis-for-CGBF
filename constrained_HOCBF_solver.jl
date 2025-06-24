using SumOfSquares
using DynamicPolynomials
using Mosek
using MosekTools
using JuMP

include("utils.jl")
using .taylorApproxModule: taylor_cos, taylor_sin

function hocbf_solver()

    # Indexed polynomial variables x[1] x[2]
    @polyvar x[1:4]

    # Parameters to control shape
    R = 3.5  # Radius-like parameter
    a = 2.0  # Controls the curvature along the y-axis
    b = 1.5  # Controls horizontal tilt

    du = 3
    ds = 8

    xi0 = 1e-8

    # System dynamics
    f0 = x[4] * taylor_cos(x[3], 8)
    f1 = x[4] * taylor_sin(x[3], 7)

    f = [f0, f1, 0, 0] # 4-element vector (Julia treats as column vector)
    g = [0 0; 0 0; 0 1; 1 0] # 4x2 matrix

    # Expression of value function h
    h = a*(R - x[2])^2 - b*x[1] - (x[1]^4 + x[2]^4 - R^2)^2 

    jacobian_h = reshape([differentiate(h, x[i]) for i in 1:4], 1, 4)
    Lfh = jacobian_h * reshape(f, 4, 1) # Size = (1, 1)

    jacobian_Lfh = reshape([differentiate(Lfh[1, 1], x[i]) for i in 1:4], 1, 4)
    Lf2h = jacobian_Lfh * reshape(f, 4, 1) # Size = (1, 1)
    LgLfh = jacobian_Lfh * g # Size = (1, 2)


    # Create monomials for control inputs
    monos_ux = monomials(x, 0:du)

    model = SOSModel(Mosek.Optimizer)

    # Define control polynomial variables
    @variable(model, u1, Poly(monos_ux))
    @variable(model, u2, Poly(monos_ux))
    @variable(model, alpha1)
    @variable(model, alpha2)
    @variable(model, delta)

    u = reshape([u1; u2], 2, 1)
    LgLfhu = LgLfh * u

    # Auxiliary polynomials
    # monos_s = monomials(x, 0:ds)
    # @variable(model, s0, Polynomial(monos_s))

    # SOS constraints
    # BUG quadratic constraint!!
    @constraint(model, Lf2h[1, 1] + LgLfhu[1, 1] + alpha1 * Lfh[1, 1] + alpha1 * alpha2 * h + delta >= 0)
    @constraint(model, delta >= 0)
    @constraint(model, alpha1 - xi0 >= 0)
    @constraint(model, alpha2 - xi0 >= 0)
    # @constraint(model, s0 >= 0)

    # Set objective function (Minimise delta)
    @objective(model, Min, delta)

    # Solve the optimisation problem
    optimize!(model)

    u1_poly = string(value(u1))
    u2_poly = string(value(u2))
    alpha2 = value(alpha2)

    return u1_poly, u2_poly, alpha2
end


u1, u2 = hocbf_solver()

println(u1)
println(u2)