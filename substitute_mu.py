import juliacall
from juliacall import Main as jl
from utils import py_expr2julia_str, julia_str2py_expr, traj_plot, u_plot, convert_vars_to_indexed
import sympy as sp
import numpy as np

# Load julia functions
jl.include("sos_solver.jl")

# Parameters to control shape
R = 3.5  # Radius-like parameter
a = 2.0  # Controls the curvature along the y-axis
b = 1.5  # Controls horizontal tilt

lambda_ = 1.8513e-05

# Input and state variale
x = sp.symbols("x:4")
y = sp.symbols("y:2")

mu = 2.1963299190601216e-12*x[3] + 1.0494758035052001e-8*x[3]**2 - 2.6349834520276613e-13*x[2]*x[3] - 9.205003648583509e-13*x[1]*x[3] - 2.3092720349850776e-12*x[0]*x[3] - 4.111749612007152e-8*x[0]*x[3]**2 + 2.6356644744999853e-12*x[0]*x[2]*x[3] + 2.690346217980559e-13*x[0]*x[1]*x[3] - 3.1876126450730262e-12*x[0]**2*x[3] - 1.0858905303014626e-12*x[0]*x[3]**3 + 7.589537445854428e-14*x[0]*x[2]*x[3]**2 + 2.6323060376706904e-13*x[0]*x[2]**2*x[3] - 2.407067429789732e-12*x[0]*x[1]*x[3]**2 + 3.0785388342225183e-13*x[0]*x[1]*x[2]*x[3] + 2.9216623105405364e-12*x[0]*x[1]**2*x[3] + 2.3708486538374352e-8*x[0]**2*x[3]**2 - 1.9397495169429723e-12*x[0]**2*x[2]*x[3] - 2.1118420172551907e-12*x[0]**2*x[1]*x[3] + 4.219407301583722e-12*x[0]**3*x[3] + 4.391767159662844e-13*x[0]*x[2]*x[3]**3 - 3.94574203925093e-8*x[0]*x[2]**2*x[3]**2 + 5.876990679797354e-13*x[0]*x[2]**3*x[3] + 4.2485692923954145e-15*x[0]*x[1]*x[2]*x[3]**2 + 2.648257911733019e-13*x[0]*x[1]*x[2]**2*x[3] - 6.168275307217552e-15*x[0]*x[1]**2*x[2]*x[3] + 1.5847080406389663e-13*x[0]**2*x[2]*x[3]**2 - 4.874497745906312e-13*x[0]**2*x[2]**2*x[3] - 7.33228123411078e-13*x[0]**2*x[1]*x[2]*x[3] - 7.394605272359176e-13*x[0]**3*x[2]*x[3]

# Define the safe and target region
psi = a*(R - y[1])**2 - b*y[0] - (y[0]**4 + y[1]**4 - R**2)**2 # Safe region
phi = ((y[0] + 0.5)**2 / 1.0**2) + ((y[1] - (2.1-4.0))**2 / 0.5**2) - 1 # Target region

h_exp = py_expr2julia_str(psi)
g_exp = py_expr2julia_str(phi)

k1_0, k1_1 = jl.sos_solver(
    h_exp = h_exp, 
    ds = 8, 
    du = 3)

k1_0_sp = julia_str2py_expr(k1_0, y)
k1_1_sp = julia_str2py_expr(k1_1, y)

# TAG Define the system model (Dubins car)
f0 = x[3] * sp.cos(x[2])
f1 = x[3] * sp.sin(x[2])


f = sp.Matrix([f0, f1, 0, 0]) # Automatically treats as a 4*1 column vector
g = sp.Matrix([[0, 0], [0, 0], [1, 0], [0, 1]]) # Automatically treats as a 4*2 matrix

k1 = sp.Matrix([k1_0_sp, k1_1_sp]) # Automatically treats as a 2*1 column vector

# Parameters to control shape (h function: Safe set)
psi_y = sp.Matrix([psi])

# Define the output
h = sp.Matrix([x[0], x[1]]) # 2*1 column vector

Lfh = h.jacobian(x) @ f

F = Lfh.jacobian(x) @ f
G = Lfh.jacobian(x) @ g

Dy_psi = psi_y.jacobian(y)

item1 = -F
item2 = mu * Dy_psi.T
item3 = k1.jacobian(sp.Matrix([y[0], y[1]])) @ Lfh
item4 = lambda_/2 * (Lfh - k1)
item2 = item2.subs({y[0]:x[0], y[1]:x[1]})
item3 = item3.subs({y[0]:x[0], y[1]:x[1]})
item4 = item4.subs({y[0]:x[0], y[1]:x[1]})

ku = sp.inv_quick(G) @ (item1 + item2 + item3 + item4) # Shape = (2, 1)

dyn_cl = f + g * ku
# dyn_cl = sp.simplify(dyn_cl)

dyn_cl_f = sp.lambdify(x, dyn_cl, "numpy")
ku_f = sp.lambdify(x, ku, "numpy")

psi_gamma = sp.Matrix([psi]) - 1/(2*mu) * (Lfh - k1).T @ (Lfh - k1) # Shape = (1, 1)

# TAG Convert from h(y) to h(x)
phi_x = phi.subs({y[0]:x[0], y[1]:x[1]})
psi_x = psi.subs({y[0]:x[0], y[1]:x[1]})
# TODO send to sos
psi_gamma_x = psi_gamma.subs({y[0]:x[0], y[1]:x[1]}) # Shape = (1, 1)


np.random.seed(6)
num_points = 500
pts = np.random.random((4, num_points)) * 4 - 2

psi_fx = sp.lambdify(x, psi_x, "numpy")
phi_fx = sp.lambdify(x, phi_x, "numpy")
psi_gamma_fx = sp.lambdify(x, psi_gamma_x, "numpy")

vals_psi_gamma = psi_gamma_fx(*pts).squeeze()
psi_vals = psi_fx(*pts)
phi_vals = phi_fx(*pts)

# TAG Find pts that inside the safe set and outside the target set
index = np.nonzero((psi_vals >= 0) & (vals_psi_gamma >= 0) & (phi_vals > 0))
# index = np.nonzero((psi_vals >= 0) & (phi_vals > 0))

pts_init = pts[:, index].squeeze(axis = 1)

dt = 1e-4

ku_traj = []
pts_x_traj = [pts_init]
pts_y_traj = [pts_init[[0, 1], :]]

time_range = 10000
for i in range(time_range):
    pts_cur = pts_x_traj[-1]

    pts_x_next = (dt * dyn_cl_f(*pts_cur)).squeeze(axis=1) + pts_cur
    pts_y_next = pts_x_next[[0, 1], :]
    ku_next = ku_f(*pts_cur).squeeze(axis=1)

    pts_x_traj.append(pts_x_next)
    pts_y_traj.append(pts_y_next)
    ku_traj.append(ku_next)

traj_x = np.stack(pts_x_traj)
traj_y = np.stack(pts_y_traj)
traj_ku = np.stack(ku_traj)

traj_plot(
    pts_init=pts_init,
    traj_x=traj_x,
    traj_y=traj_y,
    psi=sp.lambdify(y, psi, "numpy")
    # phi=sp.lambdify(y, phi, "numpy")
)

u_plot(
    traj_ku=traj_ku,
    time_range=time_range
)