import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def py_expr2julia_str(py_expr, var_mapping=None):
    """
    Convert a Python mathematical expression to a Julia expression string.
    
    Args:
        py_expr (expr): Python expression as string
        var_mapping (dict): Optional mapping of Python variables to Julia variables
                          Default maps y[0] -> x, y[1] -> y, etc.
    
    Returns:
        str: Julia expression string
    """
    
    # Default variable mapping if none provided
    if var_mapping is None:
        var_mapping = {
            'y0': 'x',
            'y1': 'y',
            'y2': 'z',
            # Add more mappings as needed
        }
    
    # Convert to string if not already
    julia_string = str(py_expr)
    
    # Replace variable mappings (do this first, before ** replacement)
    for py_var, julia_var in var_mapping.items():
        julia_string = julia_string.replace(py_var, julia_var)
    
    # Replace ** with ^ for exponentiation
    julia_string = julia_string.replace('**', '^')
    
    return julia_string

def julia_str2py_expr(julia_string, vars, var_mapping=None):
    """
    Convert a julia expression string to python mathematical expression

    Args:
        julia_string (string): Julia expression string
        vars (sp.Symbol): Defined sympy variables
        var_mapping (dict): Optional mapping of julia variables to python variables

    Returns:
        expr: Python mathematical expression 
    """

    py_expr = sp.sympify(julia_string)
    
    # Create substitution mapping: x -> y[0], y -> y[1]
    if var_mapping is None:
        var_mapping = {sp.Symbol('x'): vars[0], 
                       sp.Symbol('y'): vars[1]}

    # Apply substitution
    py_expr = py_expr.subs(var_mapping)

    return py_expr

def traj_plot(pts_init, traj_x, traj_y, psi, phi):
    """
    Trajectory and level set plot of python + julia solved SOS programming problem.

    Args:
        pts_init (np(4, n)): Initial state variables points (x)
        traj_x (np): Trajectory of the state variable over simulation steps
        traj_y (np): Trajectory of the output over simulation steps
        psi (multi-var func): Function for safe region
        phi (multi-var func): Function for target region
    """
    px = 1/plt.rcParams["figure.dpi"]
    fig, ax = plt.subplots(figsize=(640*px, 600*px), layout="constrained")
    fig.set_dpi(150)

    # Trajectory simulation

    ny1 = np.linspace(-2.5, 2.5, 500)
    ny2 = np.linspace(-2.5, 2.5, 500)

    Y1, Y2 = np.meshgrid(ny1, ny2)
    # phi_fy = sp.lambdify(y, phi, "numpy")
    # psi_fy = sp.lambdify(y, psi, "numpy")

    Z_psi = psi(Y1, Y2) # Safe region (h > 0)
    Z_phi = phi(Y1, Y2) # Target region

    # Safe region plot
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

    # Target region plot
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