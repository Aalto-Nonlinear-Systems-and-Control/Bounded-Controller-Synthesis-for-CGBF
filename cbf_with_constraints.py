import juliacall
from juliacall import Main as jl
from utils import julia_indexed_str2py_expr, traj_plot
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Load julia functions
jl.include("constrained_HOCBF_solver.jl")

# State and output variables
x = sp.symbols("x:4")
# x = sp.symbols("x:3") # x, y, theta
y = sp.symbols("y:2")

# Parameters to control shape
R = 3.5  # Radius-like parameter
a = 2.0  # Controls the curvature along the y-axis
b = 1.5  # Controls horizontal tilt

# Define the safe and target region
psi = a*(R - y[1])**2 - b*y[0] - (y[0]**4 + y[1]**4 - R**2)**2 # Safe region
# psi = a**2 - y[0]**2 - y[1]**2

# Define the system dynamics (Dubins car, 2 relative degree)
f0 = x[3] * sp.cos(x[2])
f1 = x[3] * sp.sin(x[2])

f = sp.Matrix([f0, f1, 0, 0]) # Automatically treats as a 4*1 column vector
g = sp.Matrix([[0, 0], [0, 0], [0, 1], [1, 0]]) # Automatically treats as a 4*2 matrix

# Define the system dynamics (Dubins car)

# f = sp.Matrix([0, 0, 0]) # Automatically treats as a 4*1 column vector
# g = sp.Matrix([[sp.cos(x[2]), 0], [sp.sin(x[2]), 0], [0, 1]]) # Automatically treats as a 4*2 matrix

# TODO
u1, u2, alpha = jl.cbf_solver()

u1_sp = julia_indexed_str2py_expr(u1, x)
u2_sp = julia_indexed_str2py_expr(u2, x)

kx = sp.Matrix([u1_sp, u2_sp])
dyn_cl = f + g * kx
dyn_cl[-1] = [0 * x[2]]
dyn_cl_f = sp.lambdify(x, dyn_cl, "numpy") # Dynamics of the closed-loop system as a function of x

psi_x = psi.subs({y[0]:x[0], y[1]:x[1]})
psi_fx = sp.lambdify(x, psi_x, "numpy")

# Lfh = sp.Matrix([psi_x]).jacobian(x) @ f # Shape = (1, 1)
# psi1_x =  Lfh + alpha2 * sp.Matrix([psi_x]) # Shape = (1, 1)
# psi1_fx = sp.lambdify(x, psi1_x, "numpy")

np.random.seed(6)
num_points = 100
pts = np.random.random((4, num_points)) * 4 - 2

psi_vals = psi_fx(*pts)
# psi1_vals = psi1_fx(*pts).squeeze()
index = np.nonzero(psi_vals >= 0)

pts_init = pts[:, index].squeeze(axis = 1)

dt = 1e-4

pts_x_traj = [pts_init]
pts_y_traj = [pts_init[[0, 1], :]]

for i in range(50000):
    pts_cur = pts_x_traj[-1]

    results = [dyn_cl_f(*pts_cur[:, i]) for i in range(pts_cur.shape[1])]
    pts_x_next = dt * np.column_stack(results) + pts_cur
    pts_y_next = pts_x_next[[0, 1], :]

    pts_x_traj.append(pts_x_next)
    pts_y_traj.append(pts_y_next)

traj_x = np.stack(pts_x_traj)
traj_y = np.stack(pts_y_traj)

traj_plot(
    pts_init=pts_init,
    traj_x=traj_x,
    traj_y=traj_y,
    psi=sp.lambdify(y, psi, "numpy")
)