import numpy as np
import sympy as sp
import scipy.integrate
import traceback

def parse_system(system):
    """Parse ODE system using SymPy."""
    equations = [eq.strip() for eq in system.split(',')]
    expressions = []
    
    vars = []
    for eq in equations:
        var = eq.split('/')[0].strip()[1:]
        vars.append(var)
    
    for eq in equations:
        right = eq.split('=')[1].strip()
        expressions.append(sp.sympify(right))
    
    return sp.lambdify(vars, expressions, modules='numpy')

def integrate_ode(y0, times, system, events=None, debug=False):
    """Integrate an ODE system."""
    system_fn = parse_system(system)
    
    try:
        sol = scipy.integrate.solve_ivp(
            lambda t, y: system_fn(*y),
            (min(times), max(times)),
            y0,
            t_eval=times,
            events=events
        )
        return sol.y.T
            
    except Exception as e:
        if debug:
            import traceback
            print(traceback.format_exc())
        return None

# Function to solve different ODE systems
def solve_ho(omega, gamma, y0=np.array([1.0, 1.0]), t=np.linspace(0, 10, 100)):
    """Solve harmonic oscillator system with specific parameters"""
    template = "dx/dt = y, dy/dt = -{}*x - {}*y"
    system = template.format(omega**2, gamma)
    solution = integrate_ode(y0, t, system)
    
    if solution is not None:
        return {
            'params': (omega, gamma, y0[0], y0[1]),  # Include initial conditions in params
            'equations': system,
            'solution': solution,
            'time_points': t
        }
    return None

def solve_sinusoidal(amplitude, frequency, phase, use_cos=False, t=np.linspace(0, 10, 100)):
    """Solve for a pure sinusoidal function."""
    if use_cos:
        y = amplitude * np.cos(frequency * t + phase)
    else:
        y = amplitude * np.sin(frequency * t + phase)
    # Create a "solution" with position and velocity (derivative)
    solution = np.zeros((len(t), 2))
    solution[:, 0] = y
    # Calculate the derivative
    solution[:, 1] = amplitude * frequency * (np.cos(frequency * t + phase) if not use_cos else -np.sin(frequency * t + phase))
    
    return {
        'params': (amplitude, frequency, phase, use_cos),
        'equations': f"y = {amplitude} * {'cos' if use_cos else 'sin'}({frequency}t + {phase})",
        'solution': solution,
        'time_points': t
    }

def solve_linear(slope, intercept, t=np.linspace(0, 10, 100)):
    """Solve for a linear function."""
    y = slope * t + intercept
    # Create a "solution" with position and velocity
    solution = np.zeros((len(t), 2))
    solution[:, 0] = y
    # Derivative is constant for linear function
    solution[:, 1] = np.ones_like(t) * slope
    
    return {
        'params': (slope, intercept),
        'equations': f"y = {slope}t + {intercept}",
        'solution': solution,
        'time_points': t
    }

def solve_lotka_volterra(alpha, beta, delta, gamma, y0=np.array([1.0, 0.5]), t=np.linspace(0, 20, 200)):
    """Solve the Lotka-Volterra predator-prey model."""
    system = "dx/dt = {}*x - {}*x*y, dy/dt = -{}*y + {}*x*y".format(alpha, beta, delta, gamma)
    solution = integrate_ode(y0, t, system)
    
    if solution is not None:
        return {
            'params': (alpha, beta, delta, gamma, y0[0], y0[1]),  # Include initial conditions in params
            'equations': system,
            'solution': solution,
            'time_points': t
        }
    return None

def solve_fitzhugh_nagumo(a, b, tau, I, y0=np.array([0.0, 0.0]), t=np.linspace(0, 100, 1000)):
    """Solve the FitzHugh-Nagumo model for neuron dynamics."""
    system = "dv/dt = v - v^3/3 - w + {}, dw/dt = ({})*(v + {} - {}*w)".format(I, 1/tau, a, b)
    solution = integrate_ode(y0, t, system)
    
    if solution is not None:
        return {
            'params': (a, b, tau, I, y0[0], y0[1]),  # Include initial conditions in params
            'equations': system,
            'solution': solution,
            'time_points': t
        }
    return None

def solve_coupled_linear(alpha, beta, y0=np.array([1.0, 1.0]), t=np.linspace(0, 10, 100)):
    """Solve the coupled linear system dx/dt = alpha*y, dy/dt = beta*x."""
    
    # Define the system function directly (bypassing the parser which may have issues with negatives)
    def system_fn(t, y):
        return [alpha * y[1], beta * y[0]]
    
    try:
        sol = scipy.integrate.solve_ivp(
            system_fn,
            (min(t), max(t)),
            y0,
            t_eval=t
        )
        
        return {
            'params': (alpha, beta, y0[0], y0[1]),  # Include initial conditions in params
            'equations': f"dx/dt = {alpha}*y, dy/dt = {beta}*x",
            'solution': sol.y.T,
            'time_points': t
        }
    except Exception as e:
        print(f"Error solving coupled linear system: {e}")
        return None

def solve_van_der_pol(mu, y0=np.array([1.0, 0.0]), t=np.linspace(0, 20, 200)):
    """Solve the Van der Pol oscillator system."""
    
    def system_fn(t, y):
        return [y[1], mu * (1 - y[0]**2) * y[1] - y[0]]
    
    try:
        sol = scipy.integrate.solve_ivp(
            system_fn,
            (min(t), max(t)),
            y0,
            t_eval=t
        )
        
        return {
            'params': (mu, y0[0], y0[1]),
            'equations': f"dx/dt = y, dy/dt = {mu}(1-x²)y - x",
            'solution': sol.y.T,
            'time_points': t
        }
    except Exception as e:
        print(f"Error solving Van der Pol system: {e}")
        return None

def solve_duffing(alpha, beta, delta, gamma, omega, y0=np.array([0.0, 0.0]), t=np.linspace(0, 50, 500)):
    """Solve the Duffing oscillator system."""
    
    def system_fn(t, y):
        return [y[1], -delta * y[1] - alpha * y[0] - beta * y[0]**3 + gamma * np.cos(omega * t)]
    
    try:
        sol = scipy.integrate.solve_ivp(
            system_fn,
            (min(t), max(t)),
            y0,
            t_eval=t
        )
        
        return {
            'params': (alpha, beta, delta, gamma, omega, y0[0], y0[1]),
            'equations': f"dx/dt = y, dy/dt = -({delta})y - ({alpha})x - ({beta})x³ + ({gamma})cos({omega}t)",
            'solution': sol.y.T,
            'time_points': t
        }
    except Exception as e:
        print(f"Error solving Duffing system: {e}")
        return None

def solve_double_pendulum(g, m1, m2, l1, l2, y0=np.array([np.pi/2, 0, np.pi/2, 0]), t=np.linspace(0, 20, 200)):
    """Solve the double pendulum system."""
    
    def system_fn(t, y):
        theta1, omega1, theta2, omega2 = y
        
        # Pre-compute common terms
        delta = theta2 - theta1
        sin_delta = np.sin(delta)
        cos_delta = np.cos(delta)
        
        # Compute denominators
        den1 = (m1 + m2) * l1 - m2 * l1 * cos_delta**2
        den2 = (l2 / l1) * den1
        
        # Compute numerators
        num1 = m2 * l1 * omega1**2 * sin_delta * cos_delta + m2 * g * np.sin(theta2) * cos_delta + m2 * l2 * omega2**2 * sin_delta - (m1 + m2) * g * np.sin(theta1)
        num2 = -m2 * l2 * omega2**2 * sin_delta * cos_delta + (m1 + m2) * g * np.sin(theta1) * cos_delta - (m1 + m2) * l1 * omega1**2 * sin_delta - (m1 + m2) * g * np.sin(theta2)
        
        # Compute derivatives
        dtheta1_dt = omega1
        domega1_dt = num1 / den1
        dtheta2_dt = omega2
        domega2_dt = num2 / den2
        
        return [dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt]
    
    try:
        sol = scipy.integrate.solve_ivp(
            system_fn,
            (min(t), max(t)),
            y0,
            t_eval=t,
            method='RK45',
            rtol=1e-6,
            atol=1e-9
        )
        
        # Convert to cartesian coordinates for visualization
        theta1 = sol.y[0]
        theta2 = sol.y[2]
        
        x1 = l1 * np.sin(theta1)
        y1 = -l1 * np.cos(theta1)
        x2 = x1 + l2 * np.sin(theta2)
        y2 = y1 - l2 * np.cos(theta2)
        
        # Create solution array for ODEformer (using angular variables)
        angular_solution = sol.y.T
        
        # Create cartesian solution for visualization
        cartesian_solution = np.column_stack((x1, y1, x2, y2)).T
        
        return {
            'params': (g, m1, m2, l1, l2, y0[0], y0[1], y0[2], y0[3]),
            'equations': "Complex equations for double pendulum dynamics",
            'solution': angular_solution,
            'cartesian_solution': cartesian_solution,
            'time_points': t
        }
    except Exception as e:
        print(f"Error solving double pendulum system: {e}")
        traceback.print_exc()
        return None

def solve_lorenz(sigma, rho, beta, y0=np.array([1.0, 1.0, 1.0]), t=np.linspace(0, 50, 5000)):
    """Solve the Lorenz system."""
    
    def system_fn(t, y):
        return [sigma * (y[1] - y[0]), y[0] * (rho - y[2]) - y[1], y[0] * y[1] - beta * y[2]]
    
    try:
        sol = scipy.integrate.solve_ivp(
            system_fn,
            (min(t), max(t)),
            y0,
            t_eval=t,
            method='RK45',
            rtol=1e-6,
            atol=1e-9
        )
        
        return {
            'params': (sigma, rho, beta, y0[0], y0[1], y0[2]),
            'equations': f"dx/dt = {sigma}(y - x), dy/dt = x({rho} - z) - y, dz/dt = xy - {beta}z",
            'solution': sol.y.T,
            'time_points': t,
            'is_3d': True
        }
    except Exception as e:
        print(f"Error solving Lorenz system: {e}")
        return None

def compute_poincare_section(solution, axis='z', value=0.0, direction=1):
    """
    Compute Poincaré section for a dynamical system.
    
    Args:
        solution: Solution dictionary containing time_points and solution
        axis: Axis for the section ('x', 'y', or 'z' for 3D systems, or a tuple of indices)
        value: Value at which to take the section
        direction: Direction of crossing (1 for positive, -1 for negative, 0 for both)
        
    Returns:
        Arrays of intersection points
    """
    # Extract solution data
    data = solution['solution']
    
    # Map axis name to index for 3D systems
    if isinstance(axis, str):
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        axis_idx = axis_map.get(axis.lower(), 0)
    else:
        # If axis is already an index
        axis_idx = axis
    
    # Find crossings of the section
    crossings = []
    for i in range(1, len(data)):
        # Check if the trajectory crossed the section
        prev_val = data[i-1, axis_idx] - value
        curr_val = data[i, axis_idx] - value
        
        # Check crossing in the specified direction
        if direction > 0 and prev_val < 0 and curr_val >= 0:
            # Positive crossing
            crossings.append(i)
        elif direction < 0 and prev_val > 0 and curr_val <= 0:
            # Negative crossing
            crossings.append(i)
        elif direction == 0 and prev_val * curr_val <= 0 and prev_val != curr_val:
            # Any crossing (but avoid tangent points)
            crossings.append(i)
    
    # Extract the intersection points
    intersection_points = []
    for i in crossings:
        t0 = solution['time_points'][i-1]
        t1 = solution['time_points'][i]
        
        point0 = data[i-1]
        point1 = data[i]
        
        # Linear interpolation parameter
        alpha = (value - point0[axis_idx]) / (point1[axis_idx] - point0[axis_idx])
        
        # Compute intersection point
        intersection = point0 + alpha * (point1 - point0)
        
        # For 3D systems, remove the sectioning axis
        if len(point0) == 3:
            # Create a point with other two coordinates
            other_indices = [j for j in range(3) if j != axis_idx]
            section_point = intersection[other_indices]
            intersection_points.append(section_point)
        else:
            # For other systems, keep all coordinates
            intersection_points.append(intersection)
    
    # Convert to numpy array
    if intersection_points:
        return np.array(intersection_points)
    else:
        return np.array([])