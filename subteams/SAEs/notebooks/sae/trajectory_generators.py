import numpy as np

def generate_oscillator(n_points=100, 
                       period=10.0, 
                       amplitude=1.0, 
                       phase_shift=0.0, 
                       damping=0.0,
                       noise_level=0.0,
                       dimensions=2):
    """
    Generate a damped oscillator trajectory.
    
    Args:
        n_points: Number of time points
        period: Oscillation period
        amplitude: Oscillation amplitude
        phase_shift: Phase shift in radians
        damping: Damping coefficient
        noise_level: Standard deviation of additive noise
        dimensions: Number of dimensions (2 or 3)
        
    Returns:
        Dictionary containing trajectory, time points, and equation
    """
    time_points = np.linspace(0, period * 2, n_points)
    omega = 2 * np.pi / period
    
    trajectory = np.zeros((n_points, dimensions))
    
    # First dimension: sine wave
    trajectory[:, 0] = amplitude * np.exp(-damping * time_points) * np.sin(omega * time_points + phase_shift)
    
    # Second dimension: cosine wave (90 degrees phase shift)
    trajectory[:, 1] = amplitude * np.exp(-damping * time_points) * np.cos(omega * time_points + phase_shift)
    
    # Third dimension if needed
    if dimensions == 3:
        # For 3D, add a vertical oscillation at half the frequency
        trajectory[:, 2] = amplitude * 0.5 * np.exp(-damping * time_points) * np.sin(omega * time_points / 2 + phase_shift)
    
    # Add noise if specified
    if noise_level > 0:
        trajectory += np.random.normal(0, noise_level, trajectory.shape)
        
    # Create equation string
    if damping == 0:
        eq_str = f"x(t) = {amplitude:.2f}·sin(2π·t/{period:.2f} + {phase_shift:.2f})"
    else:
        eq_str = f"x(t) = {amplitude:.2f}·e^(-{damping:.2f}·t)·sin(2π·t/{period:.2f} + {phase_shift:.2f})"
    
    return {
        'trajectory': trajectory,
        'time_points': time_points,
        'equation': eq_str
    }


def generate_spiral(n_points=100,
                   period=10.0,
                   radius_start=0.5,
                   radius_end=2.0,
                   height_start=0,
                   height_end=1.0,
                   revolutions=3,
                   noise_level=0.0):
    """
    Generate a 3D spiral trajectory.
    
    Args:
        n_points: Number of time points
        period: Time period
        radius_start: Starting radius
        radius_end: Ending radius
        height_start: Starting height
        height_end: Ending height
        revolutions: Number of complete revolutions
        noise_level: Standard deviation of additive noise
        
    Returns:
        Dictionary containing trajectory, time points, and equation
    """
    time_points = np.linspace(0, period, n_points)
    t_norm = time_points / period
    
    # Calculate radius and height as functions of time
    radius = radius_start + (radius_end - radius_start) * t_norm
    height = height_start + (height_end - height_start) * t_norm
    
    # Calculate angle based on revolutions
    theta = 2 * np.pi * revolutions * t_norm
    
    trajectory = np.zeros((n_points, 3))
    trajectory[:, 0] = radius * np.cos(theta)  # x-coordinate
    trajectory[:, 1] = radius * np.sin(theta)  # y-coordinate
    trajectory[:, 2] = height  # z-coordinate
    
    # Add noise if specified
    if noise_level > 0:
        trajectory += np.random.normal(0, noise_level, trajectory.shape)
        
    # Create equation string
    eq_str = (f"x(t) = r(t)·cos(θ(t)), y(t) = r(t)·sin(θ(t)), z(t) = h(t)\n"
             f"where r(t) = {radius_start:.2f} + {radius_end-radius_start:.2f}·t/{period:.2f}, "
             f"θ(t) = 2π·{revolutions}·t/{period:.2f}, "
             f"h(t) = {height_start:.2f} + {height_end-height_start:.2f}·t/{period:.2f}")
    
    return {
        'trajectory': trajectory,
        'time_points': time_points,
        'equation': eq_str
    }


def generate_lorenz(n_points=500,
                    total_time=40.0,
                    sigma=10.0,
                    rho=28.0,
                    beta=8/3,
                    initial_state=None,
                    noise_level=0.0):
    """
    Generate a Lorenz attractor trajectory.
    
    Args:
        n_points: Number of time points
        total_time: Total simulation time
        sigma, rho, beta: Lorenz system parameters
        initial_state: Initial state [x0, y0, z0]
        noise_level: Standard deviation of additive noise
        
    Returns:
        Dictionary containing trajectory, time points, and equation
    """
    def lorenz_deriv(state, t0, sigma=sigma, rho=rho, beta=beta):
        """Compute the derivative of the Lorenz system."""
        x, y, z = state
        return [
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z
        ]
    
    time_points = np.linspace(0, total_time, n_points)
    dt = time_points[1] - time_points[0]
    
    # Default initial state if none provided
    if initial_state is None:
        initial_state = [1.0, 1.0, 1.0]
    
    trajectory = np.zeros((n_points, 3))
    trajectory[0] = initial_state
    
    # Simple Euler integration
    for i in range(1, n_points):
        x, y, z = trajectory[i-1]
        dx, dy, dz = lorenz_deriv(trajectory[i-1], 0)
        trajectory[i] = [
            x + dx * dt,
            y + dy * dt,
            z + dz * dt
        ]
    
    # Add noise if specified
    if noise_level > 0:
        trajectory += np.random.normal(0, noise_level, trajectory.shape)
        
    # Create equation string
    eq_str = (f"dx/dt = {sigma:.1f}·(y - x)\n"
             f"dy/dt = x·({rho:.1f} - z) - y\n"
             f"dz/dt = x·y - {beta:.1f}·z")
    
    return {
        'trajectory': trajectory,
        'time_points': time_points,
        'equation': eq_str
    }


def generate_van_der_pol(n_points=200,
                        total_time=20.0,
                        mu=1.0,
                        initial_state=None,
                        noise_level=0.0):
    """
    Generate a Van der Pol oscillator trajectory.
    
    Args:
        n_points: Number of time points
        total_time: Total simulation time
        mu: Parameter controlling nonlinearity
        initial_state: Initial state [x0, v0]
        noise_level: Standard deviation of additive noise
        
    Returns:
        Dictionary containing trajectory, time points, and equation
    """
    def van_der_pol_deriv(state, t0, mu=mu):
        """Compute the derivative of the Van der Pol oscillator."""
        x, v = state
        return [
            v,
            mu * (1 - x**2) * v - x
        ]
    
    time_points = np.linspace(0, total_time, n_points)
    dt = time_points[1] - time_points[0]
    
    # Default initial state if none provided
    if initial_state is None:
        initial_state = [2.0, 0.0]
    
    trajectory = np.zeros((n_points, 2))
    trajectory[0] = initial_state
    
    # Simple Euler integration
    for i in range(1, n_points):
        x, v = trajectory[i-1]
        dx, dv = van_der_pol_deriv(trajectory[i-1], 0)
        trajectory[i] = [
            x + dx * dt,
            v + dv * dt
        ]
    
    # Add noise if specified
    if noise_level > 0:
        trajectory += np.random.normal(0, noise_level, trajectory.shape)
        
    # Create equation string
    eq_str = f"d²x/dt² - {mu:.1f}·(1-x²)·dx/dt + x = 0"
    
    return {
        'trajectory': trajectory,
        'time_points': time_points,
        'equation': eq_str
    }


def generate_double_pendulum(n_points=300,
                           total_time=10.0,
                           L1=1.0,
                           L2=1.0,
                           m1=1.0,
                           m2=1.0,
                           g=9.8,
                           initial_state=None,
                           noise_level=0.0):
    """
    Generate a double pendulum trajectory.
    
    Args:
        n_points: Number of time points
        total_time: Total simulation time
        L1, L2: Lengths of pendulum rods
        m1, m2: Masses of pendulum bobs
        g: Gravitational acceleration
        initial_state: Initial state [theta1, omega1, theta2, omega2]
        noise_level: Standard deviation of additive noise
        
    Returns:
        Dictionary containing trajectory, time points, and equation
    """
    def double_pendulum_deriv(state, t, L1=L1, L2=L2, m1=m1, m2=m2, g=g):
        """Compute the derivative of the double pendulum system."""
        theta1, omega1, theta2, omega2 = state
        
        # Compute derivatives
        delta = theta2 - theta1
        den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta) * np.cos(delta)
        den2 = (L2 / L1) * den1
        
        dtheta1 = omega1
        dtheta2 = omega2
        
        domega1 = ((m2 * L1 * omega1**2 * np.sin(delta) * np.cos(delta)
                   + m2 * g * np.sin(theta2) * np.cos(delta)
                   + m2 * L2 * omega2**2 * np.sin(delta)
                   - (m1 + m2) * g * np.sin(theta1))
                   / den1)
        
        domega2 = ((-m2 * L2 * omega2**2 * np.sin(delta) * np.cos(delta)
                   + (m1 + m2) * g * np.sin(theta1) * np.cos(delta)
                   - (m1 + m2) * L1 * omega1**2 * np.sin(delta)
                   - (m1 + m2) * g * np.sin(theta2))
                   / den2)
        
        return [dtheta1, domega1, dtheta2, domega2]
    
    time_points = np.linspace(0, total_time, n_points)
    dt = time_points[1] - time_points[0]
    
    # Default initial state if none provided
    if initial_state is None:
        initial_state = [np.pi/2, 0, np.pi/4, 0]
    
    # Full state trajectory (angles and angular velocities)
    state_trajectory = np.zeros((n_points, 4))
    state_trajectory[0] = initial_state
    
    # Simple Euler integration
    for i in range(1, n_points):
        deriv = double_pendulum_deriv(state_trajectory[i-1], 0)
        state_trajectory[i] = state_trajectory[i-1] + np.array(deriv) * dt
    
    # Convert to Cartesian coordinates
    x1 = L1 * np.sin(state_trajectory[:, 0])
    y1 = -L1 * np.cos(state_trajectory[:, 0])
    x2 = x1 + L2 * np.sin(state_trajectory[:, 2])
    y2 = y1 - L2 * np.cos(state_trajectory[:, 2])
    
    # Create final trajectory with bob positions
    trajectory = np.zeros((n_points, 4))
    trajectory[:, 0] = x1
    trajectory[:, 1] = y1
    trajectory[:, 2] = x2
    trajectory[:, 3] = y2
    
    # Add noise if specified
    if noise_level > 0:
        trajectory += np.random.normal(0, noise_level, trajectory.shape)
        
    # Create equation string (simplified)
    eq_str = "Double pendulum with parameters:\n"
    eq_str += f"L1={L1:.1f}, L2={L2:.1f}, m1={m1:.1f}, m2={m2:.1f}"
    
    return {
        'trajectory': trajectory,
        'time_points': time_points,
        'equation': eq_str,
        'state_trajectory': state_trajectory  # Include state trajectory for reference
    }