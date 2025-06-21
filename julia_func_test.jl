using SumOfSquares
using DynamicPolynomials
using Mosek
using MosekTools
using JuMP

function string_to_poly(str, vars...)
    # Create a local scope with polynomial variables
    local_dict = Dict(Symbol(string(v)) => v for v in vars)
    return eval(Meta.parse(str))
end


function sos_solver(h_exp, g_exp)
    """
    Args:
        h_exp: Julia math expression for safe region (h > 0)
        g_exp: Julia math expression for target region (g > 0)
    """
    # Define polynomial variables
    @polyvar x y
    vars = [x, y]

    # Safe region
    h = string_to_poly(h_exp, x, y)
    
    # Goal region
    g = string_to_poly(g_exp, x, y)
    return h, g

end