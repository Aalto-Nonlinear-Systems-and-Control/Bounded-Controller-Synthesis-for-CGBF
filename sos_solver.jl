using SumOfSquares
using DynamicPolynomials
using Mosek
using MosekTools
using JuMP

include("utils.jl")
using .utilsModule: string2poly, str2expr
using .taylorApproxModule: taylor_cos, taylor_sin

function sos_solver(; h_exp, ds, du)
    """
    Args:
        ; (For kayword argumants syntax)
        h_exp: Julia math expression for safe region (h > 0)
        ds: Degree of auxiliary polynomials
        du: Degree of control polynomials
    """
    # Define polynomial variables
    @polyvar x y
    vars = [x, y]

    h = string2poly(h_exp, x, y)

    # Parameters
    xi0 = 1e-8

    # Create monomials for control inputs
    monos_ux = monomials(vars, 0:du)

    # Initialise SOS program
    model = SOSModel(Mosek.Optimizer)

    # Define control polynomial variables
    @variable(model, u1, Poly(monos_ux))
    @variable(model, u2, Poly(monos_ux))
    @variable(model, lambda)
    @variable(model, delta)

    # Lie derivative of h along the contorl vector field
    Lhu = differentiate(h, vars[1])*u1 + differentiate(h, vars[2])*u2

    # Auxiliary polynomials
    monos_s = monomials(vars, 0:ds)
    @variable(model, s0, Poly(monos_s))
    @variable(model, s1, Poly(monos_s))

    # TAG SOS constraints
    # @constraint(model, Lhu - lambda*h + delta - s0*h - s1*g >= 0)
    @constraint(model, Lhu - lambda*h + delta - s0*h >= 0)
    @constraint(model, delta >= 0)
    @constraint(model, lambda - xi0 >= 0)
    @constraint(model, s0 >= 0)
    @constraint(model, s1 >= 0)

    # Set objective function (maximise delta)
    @objective(model, Min, delta)

    # Solve the optimisation problem
    optimize!(model)

    u1_poly = string(value(u1))
    u2_poly = string(value(u2))

    return u1_poly, u2_poly

end

function create_poly(expr_str, x, mu)
    # Parse the expression
    expr = Meta.parse(expr_str)
    
    # Evaluate in a local scope with defined variables (mu is a JuMP variable, do not define it here)
    return eval(quote
        let x = $x, mu = $mu, taylor_cos = taylor_cos, taylor_sin = taylor_sin
            $expr
        end
    end)
end


function sos_solver2(; psi_gamma_mu, psi, ku1_num, ku2_num, ku_den, u1_bound, u2_bound, dmu, ds, )
    @polyvar x[1:4]

    model = SOSModel(Mosek.Optimizer)

    monos_mux = monomials(x, 0:dmu)
    monos_ms = monomials(x, 0:ds)
    @variable(model, mu, Poly(monos_mux))
    @variable(model, delta)
    @variable(model, s11, Poly(monos_ms))
    @variable(model, s12, Poly(monos_ms))

    psi_gamma_mu = create_poly(psi_gamma_mu, x, mu)
    psi = create_poly(psi, x, mu)
    ku1_num = create_poly(ku1_num, x, mu)
    ku2_num = create_poly(ku2_num, x, mu)
    ku_den = create_poly(ku_den, x, mu)

    # TAG add constraints
    @constraint(model, mu >= 0)
    @constraint(model, psi_gamma_mu + delta >= 0)
    @constraint(model, u1_bound * ku_den^2 - ku1_num * ku_den - s11 * psi >= 0)
    @constraint(model, u1_bound * ku_den^2 + ku1_num * ku_den - s12 * psi >= 0)
    @constraint(model, s11 >= 0)
    @constraint(model, s12 >= 0)
    # @constraint(model, u2_bound * ku_den^2 - ku2_num * ku_den - 0 * psi_gamma >= 0)
    # @constraint(model, u2_bound * ku_den^2 + ku2_num * ku_den - 0 * psi_gamma >= 0)

    @objective(model, Min, delta)

    optimize!(model)
    mu_poly = string(value(mu))
    # println(mu_poly)
    result = replace(mu_poly, r"x\[(\d+)\]" => s"x\1")
    return(result)

end
