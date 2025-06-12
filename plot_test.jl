using DynamicPolynomials
using Plots

@polyvar x y
vars = [x, y]
polyExpr = x^2 + y^2 - 1 # Circle

x_min = -2; x_max = 6
y_min = -3; y_max = 3

x_range = [x_min, x_max]
y_range = [y_min, y_max]


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

p = plot()

contour!(x_vals, y_vals, zGrid, levels=[0], color=:blue, linewidth = 2)

display(p)
println("Plot should be visible. Press Enter to continue...")
readline()