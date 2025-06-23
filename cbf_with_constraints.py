import juliacall
from juliacall import Main as jl
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Load julia functions
jl.include("constrained_CBF_solver.jl")

# State and output variables
x = sp.symbols("x:4")
y = sp.symbols("y:2")

# Parameters to control shape
R = 3.5  # Radius-like parameter
a = 2.0  # Controls the curvature along the y-axis
b = 1.5  # Controls horizontal tilt

# Define the safe and target region
psi = a*(R - y[1])**2 - b*y[0] - (y[0]**4 + y[1]**4 - R**2)**2 # Safe region

# Define the system dynamics (Dubins car)
f0 = x[3] * sp.cos(x[2])
f1 = x[3] * sp.sin(x[2])

f = sp.Matrix([f0, f1, 0, 0]) # Automatically treats as a 4*1 column vector
g = sp.Matrix([[0, 0], [0, 0], [1, 0], [0, 1]]) # Automatically treats as a 4*2 matrix



dyn_cl = f + g @ kx
dyn_cl_f = sp.lambdify(x, dyn_cl, "numpy") # Dynamics of the closed-loop system as a function of x