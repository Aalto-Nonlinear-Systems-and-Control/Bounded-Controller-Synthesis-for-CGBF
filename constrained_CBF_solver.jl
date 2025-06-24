using SumOfSquares
using DynamicPolynomials
using Mosek
using MosekTools
using JuMP

include("utils.jl")
using .taylorApproxModule: taylor_cos, taylor_sin

function cbf_solver()

    # Indexed polynomial variables x, y, theta
    @polyvar x[1:3] 

    # Parameters to control shape
    R = 3.5  # Radius-like parameter
    a = 2.0  # Controls the curvature along the y-axis
    b = 1.5  # Controls horizontal tilt

    du = 3
    ds = 8

    xi0 = 1e-8

    # System dynamics
    gcos = taylor_cos(x[3], 8)
    gsin = taylor_sin(x[3], 7)

    f = [0, 0, 0] # 4-element vector (Julia treats as column vector)
    g = [gcos 0; gsin 0; 0 1] # 4x2 matrix

    # Expression of value function h
    h = a*(R - x[2])^2 - b*x[1] - (x[1]^4 + x[2]^4 - R^2)^2 

    jacobian_h = reshape([differentiate(h, x[i]) for i in 1:3], 1, 3)
    Lfh = jacobian_h * reshape(f, 3, 1) # Size = (1, 1)
    Lgh = jacobian_h * reshape(g, 3, 2)

    # Create monomials for control inputs
    monos_ux = monomials(x, 0:du)

    model = SOSModel(Mosek.Optimizer)

    # Define control polynomial variables
    @variable(model, u1, Poly(monos_ux))
    @variable(model, u2, Poly(monos_ux))
    @variable(model, alpha)
    @variable(model, delta)

    u = reshape([u1; u2], 2, 1)
    Lghu = Lgh * u

    # Auxiliary polynomials
    # monos_s = monomials(x, 0:ds)
    # @variable(model, s0, Poly(monos_s))

    # SOS constraints
    @constraint(model, Lfh[1, 1] + Lghu[1, 1] + alpha * h + delta >= 0)
    @constraint(model, delta >= 0)
    @constraint(model, alpha - xi0 >= 0)
    # @constraint(model, s0 >= 0)

    # Set objective function (Minimise delta)
    @objective(model, Min, delta)

    # Solve the optimisation problem
    optimize!(model)

    u1_poly = string(value(u1))
    u2_poly = string(value(u2))
    alpha = value(alpha)

    return u1_poly, u2_poly, alpha
end


# u1, u2 = cbf_solver()

# println(u1)
# println(u2)