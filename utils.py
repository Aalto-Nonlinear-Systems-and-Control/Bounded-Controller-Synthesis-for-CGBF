import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import re

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

def julia_indexed_str2py_expr(expr_str: str, symbol_list: tuple, var_name: str = 'x') -> sp.Expr:
    """
    Converts a string with indexed variables (e.g., 'x[1]^2 + x[2]') into a SymPy expression,
    using an existing list/tuple of SymPy symbols like x = symbols("x:4").
    
    Args:
        expr_str (str): The input expression string using ^ for exponentiation and indexed variables.
        symbol_list (tuple): A tuple/list of sympy symbols (e.g., from symbols("x:4")).
        var_name (str): The base name of the variable, default is 'x'.

    Returns:
        sympy.Expr: The parsed SymPy expression.
    """
    # Step 1: Replace '^' with '**'
    expr_str = expr_str.replace('^', '**')
    
    # Step 2: Replace x[i] with actual symbol names like x0, x1, etc.
    for i, sym in enumerate(symbol_list):
        expr_str = re.sub(rf'{re.escape(var_name)}\[{i+1}\]', str(sym), expr_str)

    # Build local dictionary for sympify
    local_dict = {str(sym): sym for sym in symbol_list}

    # Parse the expression
    expr = sp.sympify(expr_str, locals=local_dict)

    return expr

def convert_vars_to_indexed(expr):
    # Replace x followed by digits with x[digits]
    return re.sub(r'\bx(\d+)\b', r'x[\1]', expr)

def traj_plot(pts_init, traj_x, traj_y, psi, phi=None):
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
    if(phi != None):
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
    if (phi != None):
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

def u_plot(traj_ku, time_range):

    px = 1/plt.rcParams["figure.dpi"]
    fig, ax = plt.subplots(figsize=(640*px, 600*px), layout="constrained")
    fig.set_dpi(150)


    for i in range(traj_ku.shape[-1]):
        t_span = np.linspace(1, time_range, traj_ku.shape[0])
        if i == 0:
            plt.plot(
                t_span,
                traj_ku[:, 0, i],
                "red",
                linewidth = 1,
                alpha=0.4,
                zorder = 2,
                label='u_1'
            )

            plt.plot(
                t_span,
                traj_ku[:, 1, i],
                "blue",
                linewidth = 1,
                alpha=0.4,
                zorder = 2,
                label='u_2'
            )
        else:
            plt.plot(
                t_span,
                traj_ku[:, 0, i],
                "red",
                linewidth = 1,
                alpha=0.4,
                zorder = 2
            )

            plt.plot(
                t_span,
                traj_ku[:, 1, i],
                "blue",
                linewidth = 1,
                alpha=0.4,
                zorder = 2
            )


    plt.xlabel("$t$")
    plt.ylabel("$k_u$")
    plt.title("Control signal range")
    plt.legend()
    plt.grid()
    plt.show()


def julia_to_python_syntax(julia_expr):
    """
    Convert Julia-based syntax string to Python-based syntax string.
    
    Main conversions:
    - x[1] -> x[0], x[2] -> x[1], etc. (1-based to 0-based indexing)
    - ^ -> ** (exponentiation operator)
    - Ensures proper function names are preserved (sin, cos, etc.)
    
    Args:
        julia_expr (str): Julia syntax expression
        
    Returns:
        str: Python syntax expression
    """
    # Start with the input expression
    python_expr = julia_expr
    
    # Convert array indexing from 1-based to 0-based
    # Pattern: x[number] where number >= 1
    def convert_indexing(match):
        var_name = match.group(1)
        index = int(match.group(2))
        return f"{var_name}[{index - 1}]"
    
    # Find all x[number] patterns and convert them
    python_expr = re.sub(r'(\w+)\[(\d+)\]', convert_indexing, python_expr)
    
    # Convert exponentiation operator ^ to **
    # We need to be careful not to convert ^ inside function names or other contexts
    # This pattern looks for ^ that's not preceded by letters (to avoid function names)
    python_expr = re.sub(r'\^', '**', python_expr)
    
    return python_expr

# Test with your example
if __name__ == "__main__":
    julia_example = """-1.5*x[1] + 24.5*(1 - 0.285714285714286*x[2])^2 - 150.0625*(0.0816326530612245*x[1]^4 + 0.0816326530612245*x[2]^4 - 1)^2 - 280.633678724912*(-0.008432149543646558*x[1]^3 + 0.01432104402446673*x[1]^2*x[2] + 0.03607029642698979*x[1]^2 - 0.02270366350736754*x[1]*x[2]^2 - 0.02429498207125128*x[1]*x[2] + 0.080941266683892946*x[1] + 0.1999436532796817*x[2]^3 - 0.28180870972485791*x[2]^2 - 0.7095379997166807*x[2] + 0.0422099762227606*x[4]*sin(x[3]) + 1)^2/mu - 70.95018831336112*(0.2913188694454152*x[1]^3 - 0.027242857365120245*x[1]^2*x[2] - 0.21411863865522523*x[1]^2 + 0.25707997547260699*x[1]*x[2]^2 - 0.31631707637709617*x[1]*x[2] - x[1] - 0.018008316782829921*x[2]^3 - 0.01830723222023598*x[2]^2 + 0.093151705351324937*x[2] + 0.083947588679374198*x[4]*cos(x[3]) + 0.7320804198097999)^2/mu"""
    
    python_result = julia_to_python_syntax(julia_example)
    print("Original Julia expression:")
    print(julia_example)
    print("\nConverted Python expression:")
    print(python_result)
    
    # Test with simpler examples
    print("\n" + "="*50)
    print("Additional test cases:")
    
    test_cases = [
        "x[1]^2 + x[2]^3",
        "sin(x[1]) + cos(x[2]^2)",
        "x[1]*x[2] + x[3]^4",
        "(x[1] + x[2])^2"
    ]
    
    for test in test_cases:
        converted = julia_to_python_syntax(test)
        print(f"'{test}' -> '{converted}'")