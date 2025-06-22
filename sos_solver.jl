using SumOfSquares
using DynamicPolynomials
using Mosek
using MosekTools
using JuMP


function string_to_poly(str, vars...)
    # Create a local scope with polynomial variables
    temp_module = Module()
    for v in vars
        Core.eval(temp_module, :($(Symbol(string(v))) = $v))
    end
    return Core.eval(temp_module, Meta.parse(str))
end


function sos_solver(; h_exp, g_exp, ds, du)
    """
    Args:
        ; (For kayword argumants syntax)
        h_exp: Julia math expression for safe region (h > 0)
        g_exp: Julia math expression for target region (g > 0)
        ds: Degree of auxiliary polynomials
        du: Degree of control polynomials
    """
    # Define polynomial variables
    @polyvar x y
    vars = [x, y]

    h = string_to_poly(h_exp, x, y)
    g = string_to_poly(g_exp, x, y)

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
    @constraint(model, Lhu - lambda*h + delta - s0*h - s1*g >= 0)
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