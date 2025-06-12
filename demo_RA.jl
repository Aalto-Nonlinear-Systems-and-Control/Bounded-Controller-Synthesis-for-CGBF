using SumOfSquares
using DynamicPolynomials
using Mosek
using MosekTools
using JuMP
using Plots
using Dates

# For debugging
using Debugger
using Infiltrator

include("utils.jl")
using .utilsModule: plotPoly, add_streamlines!

# Define polynomial variables
@polyvar x y
vars = [x, y]

# Parameters
xi0 = 1e-8
ds = 8
du = 3

# Create monomials for control inputs
monos_ux = monomials(vars, 0:du)

# Initialise SOS program
model = SOSModel(Mosek.Optimizer)

# Define control polynomial variables
@variable(model, u1, Poly(monos_ux))
@variable(model, u2, Poly(monos_ux))
@variable(model, lambda)
@variable(model, delta)

# Safe set and target set (h > 0)
h = -(4*(x-2) - 2*y^3)^2 + 0.8*y^3 + 10

# Goal region (g < 0)
g = (((x-2) - 3.8)^2 / 1.2^2) + ((y - 1.9)^2 / 0.4^2) - 1

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

u1_poly = value(u1)
u2_poly = value(u2)
lambda_poly = value(lambda)
delta_poly = value(delta)

println("u1_poly:----------------------------------------------------------")
println(u1_poly)
println("u2_poly:----------------------------------------------------------")
println(u2_poly)
println("lambda: ", lambda_poly)
println("delta: ", delta_poly)
println("delta/lambda: ", delta_poly/lambda_poly)

# TAG Plots

x_min = -2; x_max = 6
y_min = -3; y_max = 3

p = plot()

plotPoly(-g, vars, [x_min, x_max], [y_min, y_max], :green, [0])
plotPoly(h, vars, [x_min, x_max], [y_min, y_max], :blue, [0])

add_streamlines!(p, u1_poly, u2_poly, [x_min, x_max], [y_min, y_max], 144)

# Save the figure
timestamp = Dates.format(now(), "yyyy-mm-dd_HHMM")
filename = "quiver_plot_" * timestamp * ".png"
savefig(p, "./figures/" * filename)

display(p)
println("Plot should be visible. Press Enter to continue...")
readline()