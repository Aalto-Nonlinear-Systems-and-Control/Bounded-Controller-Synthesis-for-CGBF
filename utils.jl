

module utilsModule

using Plots, Symbolics
using Symbolics: substitute, build_function
using DynamicPolynomials: AbstractPolynomial, variables

# For debugging
using Debugger
using Infiltrator

export plotPoly, add_streamlines!

function plotPoly(polyExpr, vars, x_range, y_range, colour, level_values=[0])
    # Create coordinate grids
    x_vals = range(x_range[1], x_range[2], length=400)
    y_vals = range(y_range[1], y_range[2], length=400)
    
    # Evaluate polynomial at grid points
    # For DynamicPolynomials, use the correct evaluation syntax
    zGrid = zeros(Float64, length(y_vals), length(x_vals))
    
    for (i, y_val) in enumerate(y_vals)
        for (j, x_val) in enumerate(x_vals)
            # Method 1: Direct substitution using => syntax
            zGrid[i, j] = polyExpr(vars[1] => x_val, vars[2] => y_val)
        end
    end
    
    # Create contour plot
    contour!(x_vals, y_vals, zGrid, 
             levels=level_values,
             linewidth=2,
             linecolor=colour,
             fillcolor=colour,
             fillalpha=0.4,
             xlabel="x",
             ylabel="y",
             title="level set of the symbolic polynomial",
             grid=true,
             aspect_ratio=:equal)
end

function expr2fun(poly)
    """
    Convert polynomial expression to function
    Handles DynamicPolynomials.Polynomial objects from JuMP
    """
    if isa(poly, Function)
        return poly
    elseif isa(poly, String)
        # Parse string expression and create function
        try
            # Replace common mathematical functions and operators for Julia syntax
            julia_expr = replace(poly, "^" => ".^")
            julia_expr = replace(julia_expr, "*" => ".*")
            julia_expr = replace(julia_expr, "/" => "./")
            
            # Create function string
            func_str = "(x, y) -> " * julia_expr
            return eval(Meta.parse(func_str))
        catch e
            error("Failed to parse string expression: $poly")
        end
    elseif isa(poly, AbstractPolynomial)
        # Handle DynamicPolynomials.Polynomial objects (from JuMP value())
        vars = variables(poly)
        if length(vars) == 0
            # Constant polynomial
            return (x, y) -> poly()
        elseif length(vars) == 1
            # Single variable - assume it's either x or y
            var = first(vars)
            return (x, y) -> poly(var => x)  # or could be y, depends on your setup
        else
            # Multiple variables - assume first two are x, y (or whatever your variables are)
            var1, var2 = vars[1:2]
            return (x, y) -> poly(var1 => x, var2 => y)
        end
    else
        error("Unsupported input type: $(typeof(poly)). Expected Function, String, or AbstractPolynomial")
    end
end

function string2poly(str, vars...)
    # Create a local scope with polynomial variables
    temp_module = Module()
    for v in vars
        Core.eval(temp_module, :($(Symbol(string(v))) = $v))
    end
    return Core.eval(temp_module, Meta.parse(str))
end

function add_streamlines!(existing_plot, P, Q, xrange=[-5, 5], yrange=[-5, 5], density=30)
    """
    Add streamlines to an existing Plots.jl plot object
    """
    P_func = expr2fun(P)
    Q_func = expr2fun(Q)

    # Generate starting points for streamlines
    n_starts = Int(round(sqrt(density)))
    x_starts = range(xrange[1], xrange[2], length=n_starts)
    y_starts = range(yrange[1], yrange[2], length=n_starts)

    # Compute and plot each streamline
    for x0 in x_starts, y0 in y_starts
        x_stream, y_stream = compute_streamline(x0, y0, P_func, Q_func, xrange, yrange)
        if length(x_stream) > 1
            # Plot the streamline path
            plot!(existing_plot, x_stream, y_stream, color=RGB(146/255, 149/255, 145/255), linewidth=0.5, alpha=0.7, label="") # Use label = "" to get rid of the legend

            # Add an arrow near the end of the streamline
            idx = round(Int, length(x_stream) * 0.8) # 80% along the streamline
            if idx < length(x_stream)
                x_arrow = x_stream[idx]
                y_arrow = y_stream[idx]
                u = x_stream[idx + 1] - x_stream[idx]
                v = y_stream[idx + 1] - y_stream[idx]

                scale_factor = 0.5

                #@infiltrate # For debugging, execution will pause here
                u_scaled = u * scale_factor
                v_scaled = v * scale_factor

                quiver!(existing_plot, [x_arrow], [y_arrow],
                        quiver=([u_scaled], [v_scaled]),
                        color=:black,
                        alpha=0.5,
                        lw=0.5,
                        arrow=true,
                        label="")
            end
        end
    end

    return existing_plot
end

function compute_streamline(x0, y0, P_func, Q_func, xrange, yrange, max_steps=2000, step_size=1e-5)
    """
    Compute a single stream line starting from (x0, y0)
    """
    x_stream = [x0]
    y_stream = [y0]

    x, y = x0, y0

    for i in 1:max_steps
        # Check bounds
        if x<xrange[1]||x>xrange[2]||y<yrange[1]||y>yrange[2]
            break
        end

        push!(x_stream, x)
        push!(y_stream, y)

        # Compute derivatives
        try
            dx = P_func(x, y) * step_size
            dy = Q_func(x, y) * step_size

            # Check for very small velocities
            if abs(dx) < 1e-6 && abs(dy) < 1e-6
                break
            end

            x += dx
            y += dy

        catch
            break
        end
    end

    return x_stream, y_stream
end

end


module taylorApproxModule

using DynamicPolynomials

function taylor_sin(x, order::Int)
    """
        taylor_sin(x, order::Int)

    Returns a polynomial approximation of sin(x) using Taylor expansion
    up to the specified odd order (e.g., 1, 3, 5, ...).
    """
    @assert isodd(order) "Order for sin(x) must be an odd number."
    approx = 0
    for n in 0:((order - 1) ÷ 2)
        term = (-1)^n * x^(2n + 1) / factorial(2n + 1)
        approx += term
    end
    return approx
end

function taylor_cos(x, order::Int)
    """
        taylor_cos(x, order::Int)

    Returns a polynomial approximation of cos(x) using Taylor expansion
    up to the specified even order (e.g., 0, 2, 4, ...).
    """
    @assert iseven(order) "Order for cos(x) must be an even number."
    approx = 0
    for n in 0:(order ÷ 2)
        term = (-1)^n * x^(2n) / factorial(2n)
        approx += term
    end
    return approx
end

end