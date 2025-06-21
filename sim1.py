import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib import cm

from example_k1 import k1_0, k1_1, y

if __name__ == "__main__":
    x = sp.symbols("x:4")

    f0 = x[3] * sp.cos(x[2])
    f1 = x[3] * sp.sin(x[2])


    f = sp.Matrix([f0, f1, 0, 0]) # Automatically treats as a 4*1 column vector
    g = sp.Matrix([[0, 0], [0, 0], [1, 0], [0, 1]]) # Automatically treats as a 4*2 matrix

    k1 = sp.Matrix([k1_0, k1_1]) # Automatically treats as a 2*1 column vector


    # Parameters to control shape
    R = 3.5  # Radius-like parameter
    a = 2.0  # Controls the curvature along the y-axis
    b = 1.5  # Controls horizontal tilt
    psi_y = sp.Matrix([a*(R - y[1])**2 - b*y[0] - (y[0]**4 + y[1]**4 - R**2)**2])

    # Define the output
    h = sp.Matrix([x[0], x[1]]) # 2*1 column vector

    Lfh = h.jacobian(x) @ f

    F = Lfh.jacobian(x) @ f
    G = Lfh.jacobian(x) @ g


    Dy_psi = psi_y.jacobian(y)

    mu_1 = 5
    lambda_ = 1.8513e-05

    item1 = -F
    item2 = mu_1 * Dy_psi.T
    item3 = k1.jacobian(sp.Matrix([y[0], y[1]])) @ Lfh
    item4 = lambda_/2 * (Lfh - k1)
    item2 = item2.subs({y[0]:x[0], y[1]:x[1]})
    item3 = item3.subs({y[0]:x[0], y[1]:x[1]})
    item4 = item4.subs({y[0]:x[0], y[1]:x[1]})

    ku = sp.inv_quick(G) @ (item1 + item2 + item3 + item4)

    dyn_cl = f + g * ku
    # dyn_cl = sp.simplify(dyn_cl)

    dyn_cl_f = sp.lambdify(x, dyn_cl, "numpy")

    psi = a*(R - y[1])**2 - b*y[0] - (y[0]**4 + y[1]**4 - R**2)**2 # Safe region
    phi = ((y[0] + 0.5)**2 / 1.0**2) + ((y[1] - (2.1-4.0))**2 / 0.5**2) - 1 # Targer region
    psi_gamma = sp.Matrix([psi]) - 1/(2*mu_1) * (Lfh - k1).T @ (Lfh - k1)

    psi_gamma_x = psi_gamma.subs({y[0]:x[0], y[1]:x[1]})
    phi_x = phi.subs({y[0]:x[0], y[1]:x[1]})
    psi_x = psi.subs({y[0]:x[0], y[1]:x[1]})

    np.random.seed(6)
    pts = np.random.random((4, 25000)) * 4 - 2

    psi_fx = sp.lambdify(x, psi_x, "numpy")
    phi_fx = sp.lambdify(x, phi_x, "numpy")
    psi_gamma_fx = sp.lambdify(x, psi_gamma_x, "numpy")

    vals_psi_gamma = psi_gamma_fx(*pts).squeeze()
    psi_vals = psi_fx(*pts)
    phi_vals = phi_fx(*pts)

    # Find pts that inside the safe set and outside the target set
    index = np.nonzero((psi_vals >= 0) & (phi_vals > 0) & (vals_psi_gamma >= 0)) # HACK
    # index = np.nonzero((psi_vals >= 0) & (phi_vals > 0))

    pts_init = pts[:, index].squeeze(axis = 1)
    
    dt = 1e-5

    pts_x_traj = [pts_init]
    pts_y_traj = [pts_init[[0, 1], :]]

    for i in range(4800):
        pts_cur = pts_x_traj[-1]

        pts_x_next = (dt * dyn_cl_f(*pts_cur)).squeeze(axis=1) + pts_cur
        pts_y_next = pts_x_next[[0, 1], :]

        pts_x_traj.append(pts_x_next)
        pts_y_traj.append(pts_y_next)

    traj_x = np.stack(pts_x_traj)
    traj_y = np.stack(pts_y_traj)

    px = 1/plt.rcParams["figure.dpi"]
    fig, ax = plt.subplots(figsize=(640*px, 600*px), layout="constrained")
    fig.set_dpi(150)

    # Trajectory simulation

    ny1 = np.linspace(-2.5, 2.5, 500)
    ny2 = np.linspace(-2.5, 2.5, 500)

    Y1, Y2 = np.meshgrid(ny1, ny2)
    phi_fy = sp.lambdify(y, phi, "numpy")
    psi_fy = sp.lambdify(y, psi, "numpy")

    Z_phi = phi_fy(Y1, Y2)
    Z_psi = psi_fy(Y1, Y2)

    track = ax.contourf(
        Y1,
        Y2,
        Z_psi,
        levels =  np.linspace(0, 80, 30),
        alpha = 0.4,
        # colors = [(0.5, 0.5, 0.5)],
        cmap = "viridis",
        zorder = 1
    )
    cbar = fig.colorbar(track)
    cbar.set_label(r"Value of $\psi(\boldsymbol{y})$")

    ax.contourf(
        Y1,
        Y2,
        Z_phi,
        levels = [-np.inf, 0],
        alpha = 0.8,
        colors = "skyblue",
        # cmap = "viridis",
        zorder = 2,
    )

    for i in range(traj_y.shape[-1]):
        plt.plot(
            traj_y[:, 0, i],
            traj_y[:, 1, i],
            "black",
            linewidth = 1,
            alpha=0.4,
            zorder = 2
        )

    plt.scatter(pts_init[0, :], pts_init[1, :], s = 1.5, c = "blue", alpha = 0.8, zorder = 3)
    plt.scatter(traj_y[-1, 0, :], traj_y[-1, 1, :], s = 1.5, c = "red", zorder = 3)

    
    plt.axis("equal")
    plt.autoscale(tight=True)

    # Extract heading angles
    theta_init = pts_init[2, :]  # Extract theta from initial points
    dx_init = np.cos(theta_init)  # X-component of heading direction
    dy_init = np.sin(theta_init)  # Y-component of heading direction

    # Quiver plot to visualize heading direction
    plt.quiver(
        pts_init[0, :], pts_init[1, :],  # Position (x, y)
        dx_init, dy_init,  # Direction components
        angles="xy", scale_units="xy", scale=10, color="blue", alpha=0.7, width=0.002,
        zorder = 3
    )

    theta_fnl = traj_x[-1, 2, :]
    dx_fnl = np.cos(theta_fnl)
    dy_fnl = np.sin(theta_fnl)

    plt.quiver(
        traj_y[-1, 0, :], traj_y[-1, 1, :],  # Position (x, y)
        dx_fnl, dy_fnl,  # Direction components
        angles="xy", scale_units="xy", scale=10, color="red", alpha=0.7, width=0.002,
        zorder = 3
    )

    plt.xlabel("$y_1$")
    plt.ylabel("$y_2$")
    plt.title("Reach-Avoid Simulation for Dubins Car on a Track Field")
    plt.grid()
    plt.show()


    # traj_psi_gamma_vals = []

    # for i in range(traj_x.shape[-1]):
    #     this_traj_x = traj_x[..., i]
    #     this_psi_gamma_vals = psi_gamma_fx(
    #         this_traj_x[:, 0], this_traj_x[:, 1], this_traj_x[:, 2], this_traj_x[:, 3]
    #     ).squeeze()
    #     traj_psi_gamma_vals.append(this_psi_gamma_vals)

    # traj_psi_gamma_vals = np.stack(traj_psi_gamma_vals)
    # print(traj_psi_gamma_vals.shape)

    # t = np.arange(traj_psi_gamma_vals.shape[-1])
    # print(t.shape)

    # px = 1 / plt.rcParams["figure.dpi"]
    # fig, ax = plt.subplots(figsize=(640 * px, 600 * px), layout="constrained")
    # fig.set_dpi(150)

    # for i in range(traj_psi_gamma_vals.shape[0]):
    #     plt.plot(t, traj_psi_gamma_vals[i, ...], "black", alpha=0.5, linewidth=0.5)

    # # plt.axis("equal")
    # plt.autoscale(tight=True)

    # plt.show()