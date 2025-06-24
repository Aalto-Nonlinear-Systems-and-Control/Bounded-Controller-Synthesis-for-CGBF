from combined_controller import *

np.random.seed(6)
num_points = 100
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