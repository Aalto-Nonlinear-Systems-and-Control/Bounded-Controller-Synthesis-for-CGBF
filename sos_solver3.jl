using SumOfSquares
using DynamicPolynomials
using Mosek
using MosekTools
using JuMP

include("utils.jl")
using .utilsModule: string2poly, str2expr
using .taylorApproxModule: taylor_cos, taylor_sin


model = SOSModel(Mosek.Optimizer)

    monos_mux = monomials(x, 0:dmu)
    monos_ms = monomials(x, 0:ds)
    @variable(model, mu, Poly(monos_mux))
    @variable(model, delta)
    @variable(model, s11, Poly(monos_ms))
    @variable(model, s12, Poly(monos_ms))

function create_poly(expr_str, x)
    # Parse the expression
    expr = Meta.parse(expr_str)
    
    # Evaluate in a local scope with defined variables (mu is a JuMP variable, do not define it here)
    return eval(quote
        let x = $x, taylor_cos = taylor_cos, taylor_sin = taylor_sin
            $expr
        end
    end)
end


function sos_solver2(; psi_gamma_mu, psi, ku1_num, ku2_num, ku_den, u1_bound, u2_bound, dmu, ds, )
    @polyvar x[1:4]


    # psi_gamma_mu = create_poly(psi_gamma_mu, x)
    # psi = create_poly(psi, x)
    # ku1_num = create_poly(ku1_num, x)
    # ku2_num = create_poly(ku2_num, x)
    # ku_den = create_poly(ku_den, x)

    # TAG add constraints
    @constraint(model, mu >= 0)
    @constraint(model, create_poly(psi_gamma_mu, x) + delta >= 0)
    # @constraint(model, u1_bound * ku_den^2 - ku1_num * ku_den - s11 * psi >= 0)
    # @constraint(model, u1_bound * ku_den^2 + ku1_num * ku_den - s12 * psi >= 0)
    # @constraint(model, s11 >= 0)
    # @constraint(model, s12 >= 0)
    # @constraint(model, u2_bound * ku_den^2 - ku2_num * ku_den - 0 * psi_gamma >= 0)
    # @constraint(model, u2_bound * ku_den^2 + ku2_num * ku_den - 0 * psi_gamma >= 0)

    @objective(model, Min, delta)

    optimize!(model)
    mu_poly = string(value(mu))
    println(mu_poly)

end