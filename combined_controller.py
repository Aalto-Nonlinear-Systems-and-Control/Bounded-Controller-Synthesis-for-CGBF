import juliacall
from juliacall import Main as jl
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Load julia functions
jl.include("julia_func_test.jl")

# Parameters to control shape
R = 3.5  # Radius-like parameter
a = 2.0  # Controls the curvature along the y-axis
b = 1.5  # Controls horizontal tilt

# Safe set and target set (h > 0)
h_exp = "a*(R - y)^2 - b*x - (x^4 + y^4 - R^2)^2"
# Replace variables with their values
h_exp = h_exp.replace('R', str(R))
h_exp = h_exp.replace('a', str(a))
h_exp = h_exp.replace('b', str(b))

# Goal region (g < 0)
g_exp = "((x + 0.5)^2 / 1.0^2) + ((y - (2.1-4.0))^2 / 0.5^2) - 1"

k1_0, k1_1 = jl.sos_solver(h_exp, g_exp)

x = sp.symbols("x:4")

f0 = x[3] * sp.cos(x[2])
f1 = x[3] * sp.sin(x[2])


f = sp.Matrix([f0, f1, 0, 0]) # Automatically treats as a 4*1 column vector
g = sp.Matrix([[0, 0], [0, 0], [1, 0], [0, 1]]) # Automatically treats as a 4*2 matrix

k1 = sp.Matrix([k1_0, k1_1]) # Automatically treats as a 2*1 column vector
