import streamlit as st
import numpy as np
np.infty=np.inf
np.random.seed(42)

import matplotlib.pyplot as plt
import torch
import sympy as sp
import scipy.integrate
import sys
import os
from pathlib import Path
import time
import importlib
from matplotlib.figure import Figure
import traceback
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Check for dimensionality reduction libraries
sklearn_available = importlib.util.find_spec("sklearn") is not None
umap_available = importlib.util.find_spec("umap") is not None

if sklearn_available and umap_available:
    import sklearn.manifold
    import umap
else:
    missing_libs = []
    if not sklearn_available:
        missing_libs.append("scikit-learn")
    if not umap_available:
        missing_libs.append("umap-learn")

# Import the activation collector
from activation_collector import *

# Add needed paths to system path
st.sidebar.header("Setup")
model_dir = st.sidebar.text_input("Model directory", "../odeformer/")
sae_dir = st.sidebar.text_input("SAE directory", "../sae/")

# Add the paths
if st.sidebar.button("Add Paths") or 'paths_added' not in st.session_state:
    if model_dir and Path(model_dir).exists():
        if model_dir not in sys.path:
            sys.path.append(model_dir)
        if f"{model_dir}/odeformer" not in sys.path:
            sys.path.append(f"{model_dir}/odeformer")
        if f"{model_dir}/odeformer/model" not in sys.path:
            sys.path.append(f"{model_dir}/odeformer/model")
        if f"{model_dir}/odeformer/envs" not in sys.path:
            sys.path.append(f"{model_dir}/odeformer/envs")
        st.sidebar.success(f"Added {model_dir} to sys.path")
        st.session_state.paths_added = True
    else:
        st.sidebar.error(f"Directory {model_dir} not found")
        
    if sae_dir and Path(sae_dir).exists():
        if sae_dir not in sys.path:
            sys.path.append(sae_dir)
        st.sidebar.success(f"Added {sae_dir} to sys.path")
        st.session_state.paths_added = True
    else:
        st.sidebar.error(f"Directory {sae_dir} not found")

# Initialize session state variables
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'using_real_activations' not in st.session_state:
    st.session_state.using_real_activations = False
if 'current_params' not in st.session_state:
    st.session_state.current_params = None
if 'current_solution' not in st.session_state:
    st.session_state.current_solution = None
if 'current_activations' not in st.session_state:
    st.session_state.current_activations = None
if 'current_latent_features' not in st.session_state:
    st.session_state.current_latent_features = None
if 'patches_installed' not in st.session_state:
    st.session_state.patches_installed = False
if 'time_point' not in st.session_state:
    st.session_state.time_point = 41
if 'feature_idx' not in st.session_state:
    st.session_state.feature_idx = 727
if 'system_type' not in st.session_state:
    st.session_state.system_type = "Harmonic Oscillator"
if 'current_sae_path' not in st.session_state:
    st.session_state.current_sae_path = None
if 'learned_equation' not in st.session_state:
    st.session_state.learned_equation = None
# New session state variables for activation selection
if 'activation_site' not in st.session_state:
    st.session_state.activation_site = 'RESIDUAL'
if 'activation_component' not in st.session_state:
    st.session_state.activation_component = 'encoder.transformer.residual1'
if 'all_collected_activations' not in st.session_state:
    st.session_state.all_collected_activations = None
if 'sae_paths' not in st.session_state:
    st.session_state.sae_paths = {
        "Residual Layer 1": "../sae/sae.encoder.outer.residual1_20250308_230626/checkpoints/sae_best_encoder.outer.residual1.pt",
        "Residual Layer 2": "../sae/sae.encoder.outer.residual2_20250309_122318/checkpoints/sae_best_encoder.outer.residual2.pt",
        "Residual Layer 3": "../sae/sae.encoder.outer.residual3_20250309_213613/checkpoints/sae_best_encoder.outer.residual3.pt",
        "Custom Path": ""
    }
# Initialize parameters for the new systems
if 'van_der_pol_params' not in st.session_state:
    st.session_state.van_der_pol_params = {
        'mu': 1.0,
        'x0': 1.0,
        'y0': 0.0,
    }
if 'duffing_params' not in st.session_state:
    st.session_state.duffing_params = {
        'alpha': 1.0,
        'beta': 5.0,
        'delta': 0.02,
        'gamma': 8.0,
        'omega': 0.5,
        'x0': 0.0,
        'y0': 0.0,
    }
if 'double_pendulum_params' not in st.session_state:
    st.session_state.double_pendulum_params = {
        'g': 9.8,
        'm1': 1.0,
        'm2': 1.0,
        'l1': 1.0,
        'l2': 1.0,
        'theta1': np.pi/2,
        'omega1': 0.0,
        'theta2': np.pi/2,
        'omega2': 0.0,
    }
if 'lorenz_params' not in st.session_state:
    st.session_state.lorenz_params = {
        'sigma': 10.0,
        'rho': 28.0,
        'beta': 8/3,
        'x0': 1.0,
        'y0': 1.0,
        'z0': 1.0,
    }
if 'poincare_params' not in st.session_state:
    st.session_state.poincare_params = {
        'axis': 'z',
        'value': 0.0,
        'direction': 1,
    }
    
# Try to import mishax for instrumentation
mishax_available = False
try:
    from mishax import ast_patcher, safe_greenlet
    mishax_available = True
    st.sidebar.success("✅ Found mishax libraries - instrumentation available")
except ImportError:
    st.sidebar.warning("⚠️ mishax libraries not found - using synthetic activations")
    st.sidebar.markdown("""
    To enable full instrumentation, install mishax:
    ```
    pip install mishax
    ```
    """)

# Check for dimensionality reduction libraries
if sklearn_available and umap_available:
    st.sidebar.success("✅ Found dimensionality reduction libraries")
else:
    missing_libs_str = ", ".join(missing_libs) if 'missing_libs' in locals() else "required libraries"
    st.sidebar.warning(f"⚠️ Missing libraries for dimensionality reduction: {missing_libs_str}")
    st.sidebar.markdown(f"""
    To enable t-SNE and UMAP visualizations, install:
    ```
    pip install {' '.join(missing_libs) if 'missing_libs' in locals() else "scikit-learn umap-learn"}
    ```
    """)

# Function to compute dimensionality reduction with caching
@st.cache_data
def compute_dimensionality_reduction(data, method, n_components, **kwargs):
    """
    Compute dimensionality reduction with caching.
    
    Args:
        data: Input data (n_samples, n_features)
        method: 'tsne' or 'umap'
        n_components: 2 or 3
        **kwargs: Additional parameters for the method
        
    Returns:
        Reduced dimensionality data
    """
    if method == 'tsne':
        reducer = sklearn.manifold.TSNE(
            n_components=n_components,
            random_state=42,
            **kwargs
        )
    elif method == 'umap':
        reducer = umap.UMAP(
            n_components=n_components,
            random_state=42,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown method: {method}")
        
    return reducer.fit_transform(data)

# Only proceed with model imports and loading if paths are added
if st.sidebar.button("Load Models") or st.session_state.models_loaded:
    try:
        # Import required modules
        from odeformer.model import SymbolicTransformerRegressor
        st.session_state.sae_module = importlib.import_module("sae")
        SparseAutoencoder = st.session_state.sae_module.SparseAutoencoder
        
        # Load ODEformer model
        @st.cache_resource
        def load_odeformer_model():
            model = SymbolicTransformerRegressor(from_pretrained=True)
            model_args = {'beam_size': 20, 'beam_temperature': 0.1}
            model.set_model_args(model_args)
            return model
            
        # Load SAE model
        @st.cache_resource
        def load_sae_model(model_path):
            sae_model = SparseAutoencoder(input_dim=256, latent_dim=1280)
            sae_model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            sae_model.eval()
            return sae_model
        
        # Try to load models
        try:
            # Only install patches once
            if not st.session_state.patches_installed:
                install()
                st.session_state.patches_installed = True
                
            model = load_odeformer_model()
            st.sidebar.success("ODEformer model loaded successfully")
            
            # SAE model selection
            st.sidebar.subheader("SAE Model Selection")
            sae_option = st.sidebar.selectbox(
                "Select SAE Model",
                list(st.session_state.sae_paths.keys()),
                index=0
            )

            # If custom path is selected, show text input
            if sae_option == "Custom Path":
                custom_path = st.sidebar.text_input("Custom SAE Path", 
                                                  value=st.session_state.sae_paths["Custom Path"])
                st.session_state.sae_paths["Custom Path"] = custom_path
                sae_path = custom_path
            else:
                sae_path = st.session_state.sae_paths[sae_option]
                
            # Display the selected path for reference
            st.sidebar.caption(f"Path: {sae_path}")

            if Path(sae_path).exists():
                sae_model = load_sae_model(sae_path)
                st.sidebar.success("SAE model loaded successfully")
                st.session_state.models_loaded = True
                
                # Check if SAE model changed and clear cached data if needed
                if st.session_state.current_sae_path != sae_path:
                    st.session_state.current_sae_path = sae_path
                    st.session_state.current_latent_features = None
                    st.session_state.current_activations = None
                    st.sidebar.info("SAE model changed - cache cleared")
            else:
                st.sidebar.error(f"SAE model not found at {sae_path}")
                
            # Store models in session state for later access
            st.session_state.model = model
            st.session_state.sae_model = sae_model
            
        except Exception as e:
            st.sidebar.error(f"Error loading ODEformer model: {str(e)}")
            st.sidebar.code(traceback.format_exc())
    
    except Exception as e:
        st.sidebar.error(f"Error loading models: {str(e)}")

# Add activation selection UI to the sidebar
if st.session_state.models_loaded:
    st.sidebar.header("Activation Settings")
    
    # If we've collected activations before, get the available sites
    available_sites = []
    if st.session_state.all_collected_activations:
        # Convert site keys to strings to ensure compatibility
        available_sites = [str(site) for site in st.session_state.all_collected_activations.keys()]
    
    # If no activations collected yet, use only the known available sites
    if not available_sites:
        available_sites = ['RESIDUAL', 'ATTN_OUTPUT']
    
    # Site selection dropdown
    site_index = 0
    if st.session_state.activation_site in available_sites:
        site_index = available_sites.index(st.session_state.activation_site)
        
    selected_site = st.sidebar.selectbox(
        "Activation Site", 
        available_sites,
        index=site_index
    )
    
    # Component selection dropdown - only show components that exist for this site
    available_components = []
    
    if st.session_state.all_collected_activations:
        # Find the matching site key - handle both string and enum cases
        matching_site_key = None
        for site_key in st.session_state.all_collected_activations.keys():
            if str(site_key) == selected_site:
                matching_site_key = site_key
                break
        
        if matching_site_key is not None:
            # Get available components for this site
            available_components = list(st.session_state.all_collected_activations[matching_site_key].keys())
    
    # If no components are available for this site, show an empty component dropdown
    component_index = 0
    component_disabled = False
    
    if not available_components:
        available_components = ["No components available for this site"]
        component_disabled = True
    elif st.session_state.activation_component in available_components:
        component_index = available_components.index(st.session_state.activation_component)
        
    selected_component = st.sidebar.selectbox(
        "Component", 
        available_components,
        index=component_index,
        disabled=component_disabled
    )
    
    # Only enable custom component input if we're not showing a placeholder
    use_custom = False
    if not component_disabled:
        use_custom = st.sidebar.checkbox("Use custom component", False)
    
    if use_custom:
        custom_component = st.sidebar.text_input("Custom Component", st.session_state.activation_component)
        selected_component = custom_component
    
    # Check if selection changed (don't update if we're in the disabled state)
    if (not component_disabled and
        (selected_site != st.session_state.activation_site or 
        selected_component != st.session_state.activation_component)):
        # Update session state
        st.session_state.activation_site = selected_site
        st.session_state.activation_component = selected_component
        # Clear cached activations and features
        st.session_state.current_activations = None
        st.session_state.current_latent_features = None
        st.sidebar.info("Activation selection changed - recollecting activations")

# ODE system functions
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

# New systems
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

# Function to get activations for the currently selected site and component
def get_activations(model, solution):
    """Get activations for the currently selected site and component directly from a single model run"""
    site_name = st.session_state.activation_site
    component = st.session_state.activation_component
    
    # Check if we already have all_collected_activations in the session state
    if st.session_state.all_collected_activations is not None:
        print("Using cached activations from session state")
        all_activations = st.session_state.all_collected_activations
    else:
        # Run the model only once to collect all activations
        print("Starting single model run to collect all activations...")
        all_activations, _ = collect_activations_during_fit(
            model, solution['time_points'], solution['solution']
        )
        
        # Store all collected activations for future reference
        if all_activations:
            st.session_state.all_collected_activations = all_activations
            # Don't print duplicate "Collection complete" message
            
            # Print component details for debugging
            for site in all_activations:
                print(f"Site '{site}' has components: {list(all_activations[site].keys())}")
    
    # Extract the specific activations we need from the all_activations dictionary
    specific_activations = None
    if all_activations:
        if site_name in all_activations:
            if component in all_activations[site_name]:
                shapes = list(all_activations[site_name][component].keys())
                if shapes:
                    tensor_list = all_activations[site_name][component][shapes[0]]
                    if tensor_list:
                        tensor = tensor_list[0]
                        if tensor.dim() == 3:
                            specific_activations = tensor[0].numpy()  # Extract batch dim
                        else:
                            specific_activations = tensor.numpy()
        
        # If extraction failed, try the default component without running model again
        if specific_activations is None and (site_name != 'RESIDUAL' or component != 'encoder.transformer.residual1'):
            st.warning(f"Failed to extract activations for {site_name}.{component}. Trying defaults.")
            default_component = 'encoder.transformer.residual1'
            
            # Try to extract from already collected activations
            if 'RESIDUAL' in all_activations and default_component in all_activations['RESIDUAL']:
                shapes = list(all_activations['RESIDUAL'][default_component].keys())
                if shapes:
                    tensor_list = all_activations['RESIDUAL'][default_component][shapes[0]]
                    if tensor_list:
                        tensor = tensor_list[0]
                        if tensor.dim() == 3:
                            specific_activations = tensor[0].numpy()
                        else:
                            specific_activations = tensor.numpy()
                            
                if specific_activations is not None:
                    # Update session state to reflect the actual values used
                    st.session_state.activation_site = 'RESIDUAL'
                    st.session_state.activation_component = default_component
    
    return specific_activations

# Function to process activation through SAE
def apply_sae(sae_model, activations):
    """Apply SAE to get latent features"""
    inputs = torch.tensor(activations, dtype=torch.float32)
    _, latent = sae_model(inputs)
    return latent.squeeze(0).detach().numpy()

# Function to get learned equations from the model
def get_learned_equations(model, solution):
    """
    Get the learned equations from the ODEformer model.
    
    Args:
        model: The trained ODEformer model
        solution: The solution dictionary containing time_points and solution data
        
    Returns:
        A dictionary with learned equations and metadata
    """
    try:
        # Extract time points and trajectory data
        times = solution['time_points']
        trajectories = solution['solution']
        
        # Make predictions with the model
        learned_eqs = model.predict(times, trajectories)
        
        # Get the top predicted equation
        if learned_eqs and len(learned_eqs) > 0:
            top_eq = learned_eqs[0] if isinstance(learned_eqs, list) else learned_eqs
            
            # Return in a dictionary format
            return {
                'equation_str': str(top_eq),
                'full_results': learned_eqs,
                'success': True
            }
        else:
            return {
                'equation_str': "No equations predicted",
                'full_results': None,
                'success': False
            }
    except Exception as e:
        print(f"Error getting learned equations: {str(e)}")
        return {
            'equation_str': f"Error: {str(e)}",
            'full_results': None,
            'success': False
        }

# Main Streamlit app
st.title("Dynamic Systems Feature Explorer")

if st.session_state.models_loaded:
    # System selection
    st.sidebar.header("System Selection")
    system_type = st.sidebar.selectbox(
        "Select System Type",
        ["Harmonic Oscillator", "Sinusoidal Function", "Linear Function", 
         "Lotka-Volterra", "FitzHugh-Nagumo", "Coupled Linear System",
         "Van der Pol Oscillator", "Duffing Oscillator", "Double Pendulum", "Lorenz System"],
        index=["Harmonic Oscillator", "Sinusoidal Function", "Linear Function", 
               "Lotka-Volterra", "FitzHugh-Nagumo", "Coupled Linear System",
               "Van der Pol Oscillator", "Duffing Oscillator", "Double Pendulum", "Lorenz System"].index(st.session_state.system_type) 
              if st.session_state.system_type in ["Harmonic Oscillator", "Sinusoidal Function", "Linear Function", 
                                            "Lotka-Volterra", "FitzHugh-Nagumo", "Coupled Linear System",
                                            "Van der Pol Oscillator", "Duffing Oscillator", "Double Pendulum", "Lorenz System"] else 0
    )
    st.session_state.system_type = system_type

    # Initial conditions and time span
    st.sidebar.header("Time Settings")
    t_start = st.sidebar.number_input("Start Time", value=0.0, step=1.0)
    
    # Adjust end time based on system type
    if system_type == "FitzHugh-Nagumo":
        t_end_default = 100.0
    elif system_type in ["Lotka-Volterra", "Van der Pol Oscillator", "Double Pendulum"]:
        t_end_default = 20.0
    elif system_type in ["Duffing Oscillator", "Lorenz System"]:
        t_end_default = 50.0
    else:
        t_end_default = 10.0
        
    t_end = st.sidebar.number_input("End Time", value=t_end_default, step=1.0, min_value=t_start + 0.1)
    
    # Adjust number of points based on system type
    if system_type in ["Duffing Oscillator", "Lorenz System"]:
        t_points_default = 500
    elif system_type == "FitzHugh-Nagumo":
        t_points_default = 1000
    else:
        t_points_default = 200
        
    t_points = st.sidebar.slider("Number of Time Points", 10, 5000, t_points_default, 10)
    times = np.linspace(t_start, t_end, t_points)

    # Parameter inputs based on system selection
    st.sidebar.header("System Parameters")
    
    if system_type == "Harmonic Oscillator":
        st.sidebar.subheader("Equation Parameters")
        # Use columns for sliders and text inputs
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            omega = st.slider("Natural Frequency (ω)", 0.1, 50.0, st.session_state.get('ho_omega', 1.0), 0.1)
            st.session_state.ho_omega = omega
            
        with col2:
            omega = st.number_input("ω value", 0.1, 50.0, st.session_state.ho_omega, 0.1, key="ho_omega_input")
            st.session_state.ho_omega = omega
            
        col1, col2 = st.sidebar.columns(2)
        with col1:
            gamma = st.slider("Damping Coefficient (γ)", 0.0, 50.0, st.session_state.get('ho_gamma', 0.5), 0.1)
            st.session_state.ho_gamma = gamma
            
        with col2:
            gamma = st.number_input("γ value", 0.0, 50.0, st.session_state.ho_gamma, 0.1, key="ho_gamma_input")
            st.session_state.ho_gamma = gamma
        
        st.sidebar.subheader("Initial Conditions")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            x0 = st.slider("Initial Position (x₀)", -50.0, 50.0, st.session_state.get('ho_x0', 1.0), 0.1)
            st.session_state.ho_x0 = x0
            
        with col2:
            x0 = st.number_input("x₀ value", -50.0, 50.0, st.session_state.ho_x0, 0.1, key="ho_x0_input")
            st.session_state.ho_x0 = x0
            
        col1, col2 = st.sidebar.columns(2)
        with col1:
            v0 = st.slider("Initial Velocity (v₀)", -50.0, 50.0, st.session_state.get('ho_v0', 0.0), 0.1)
            st.session_state.ho_v0 = v0
            
        with col2:
            v0 = st.number_input("v₀ value", -50.0, 50.0, st.session_state.ho_v0, 0.1, key="ho_v0_input")
            st.session_state.ho_v0 = v0
        
        # Create parameters object for current system
        system_params = (omega, gamma, x0, v0)
        
    elif system_type == "Sinusoidal Function":
        st.sidebar.subheader("Function Parameters")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            amplitude = st.slider("Amplitude (A)", 0.1, 50.0, st.session_state.get('sin_amplitude', 1.0), 0.1)
            st.session_state.sin_amplitude = amplitude
            
        with col2:
            amplitude = st.number_input("A value", 0.1, 50.0, st.session_state.sin_amplitude, 0.1, key="sin_amp_input")
            st.session_state.sin_amplitude = amplitude
            
        col1, col2 = st.sidebar.columns(2)
        with col1:
            frequency = st.slider("Frequency (ω)", 0.1, 50.0, st.session_state.get('sin_frequency', 1.0), 0.1)
            st.session_state.sin_frequency = frequency
            
        with col2:
            frequency = st.number_input("ω value", 0.1, 50.0, st.session_state.sin_frequency, 0.1, key="sin_freq_input")
            st.session_state.sin_frequency = frequency
            
        col1, col2 = st.sidebar.columns(2)
        with col1:
            phase = st.slider("Phase (φ, radians)", 0.0, 2*np.pi, st.session_state.get('sin_phase', 0.0), 0.1)
            st.session_state.sin_phase = phase
            
        with col2:
            phase = st.number_input("φ value", 0.0, 2*np.pi, st.session_state.sin_phase, 0.1, key="sin_phase_input")
            st.session_state.sin_phase = phase
            
        use_cos = st.sidebar.checkbox("Use Cosine instead of Sine", st.session_state.get('sin_use_cos', False))
        st.session_state.sin_use_cos = use_cos
        
        # Create parameters object for current system
        system_params = (amplitude, frequency, phase, use_cos)
        
    elif system_type == "Linear Function":
        st.sidebar.subheader("Function Parameters")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            slope = st.slider("Slope (m)", -50.0, 50.0, st.session_state.get('lin_slope', 1.0), 0.1)
            st.session_state.lin_slope = slope
            
        with col2:
            slope = st.number_input("m value", -50.0, 50.0, st.session_state.lin_slope, 0.1, key="lin_slope_input")
            st.session_state.lin_slope = slope
            
        col1, col2 = st.sidebar.columns(2)
        with col1:
            intercept = st.slider("Intercept (b)", -50.0, 50.0, st.session_state.get('lin_intercept', 0.0), 0.1)
            st.session_state.lin_intercept = intercept
            
        with col2:
            intercept = st.number_input("b value", -50.0, 50.0, st.session_state.lin_intercept, 0.1, key="lin_intercept_input")
            st.session_state.lin_intercept = intercept
        
        # Create parameters object for current system
        system_params = (slope, intercept)
        
    elif system_type == "Lotka-Volterra":
        st.sidebar.subheader("Equation Parameters")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            alpha = st.slider("Prey Growth Rate (α)", 0.1, 20.0, st.session_state.get('lv_alpha', 0.5), 0.1)
            st.session_state.lv_alpha = alpha
            
        with col2:
            alpha = st.number_input("α value", 0.1, 20.0, st.session_state.lv_alpha, 0.1, key="lv_alpha_input")
            st.session_state.lv_alpha = alpha
            
        col1, col2 = st.sidebar.columns(2)
        with col1:
            beta = st.slider("Predation Rate (β)", 0.01, 10.0, st.session_state.get('lv_beta', 0.2), 0.01)
            st.session_state.lv_beta = beta
            
        with col2:
            beta = st.number_input("β value", 0.01, 10.0, st.session_state.lv_beta, 0.01, key="lv_beta_input")
            st.session_state.lv_beta = beta
            
        col1, col2 = st.sidebar.columns(2)
        with col1:
            delta = st.slider("Predator Death Rate (δ)", 0.1, 20.0, st.session_state.get('lv_delta', 0.5), 0.1)
            st.session_state.lv_delta = delta
            
        with col2:
            delta = st.number_input("δ value", 0.1, 20.0, st.session_state.lv_delta, 0.1, key="lv_delta_input")
            st.session_state.lv_delta = delta
            
        col1, col2 = st.sidebar.columns(2)
        with col1:
            gamma = st.slider("Predator Growth from Prey (γ)", 0.01, 10.0, st.session_state.get('lv_gamma', 0.1), 0.01)
            st.session_state.lv_gamma = gamma
            
        with col2:
            gamma = st.number_input("γ value", 0.01, 10.0, st.session_state.lv_gamma, 0.01, key="lv_gamma_input")
            st.session_state.lv_gamma = gamma
        
        st.sidebar.subheader("Initial Conditions")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            prey0 = st.slider("Initial Prey Population (x₀)", 0.1, 50.0, st.session_state.get('lv_prey0', 1.0), 0.1)
            st.session_state.lv_prey0 = prey0
            
        with col2:
            prey0 = st.number_input("x₀ value", 0.1, 50.0, st.session_state.lv_prey0, 0.1, key="lv_prey0_input")
            st.session_state.lv_prey0 = prey0
            
        col1, col2 = st.sidebar.columns(2)
        with col1:
            predator0 = st.slider("Initial Predator Population (y₀)", 0.1, 50.0, st.session_state.get('lv_predator0', 0.5), 0.1)
            st.session_state.lv_predator0 = predator0
            
        with col2:
            predator0 = st.number_input("y₀ value", 0.1, 50.0, st.session_state.lv_predator0, 0.1, key="lv_predator0_input")
            st.session_state.lv_predator0 = predator0
        
        # Create parameters object for current system
        system_params = (alpha, beta, delta, gamma, prey0, predator0)
        
    elif system_type == "FitzHugh-Nagumo":
        st.sidebar.subheader("Equation Parameters")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            a = st.slider("Parameter a", -10.0, 10.0, st.session_state.get('fn_a', 0.7), 0.1)
            st.session_state.fn_a = a
            
        with col2:
            a = st.number_input("a value", -10.0, 10.0, st.session_state.fn_a, 0.1, key="fn_a_input")
            st.session_state.fn_a = a
            
        col1, col2 = st.sidebar.columns(2)
        with col1:
            b = st.slider("Parameter b", 0.1, 10.0, st.session_state.get('fn_b', 0.8), 0.1)
            st.session_state.fn_b = b
            
        with col2:
            b = st.number_input("b value", 0.1, 10.0, st.session_state.fn_b, 0.1, key="fn_b_input")
            st.session_state.fn_b = b
            
        col1, col2 = st.sidebar.columns(2)
        with col1:
            tau = st.slider("Time Scale (τ)", 1.0, 200.0, st.session_state.get('fn_tau', 12.5), 0.5)
            st.session_state.fn_tau = tau
            
        with col2:
            tau = st.number_input("τ value", 1.0, 200.0, st.session_state.fn_tau, 0.5, key="fn_tau_input")
            st.session_state.fn_tau = tau
            
        col1, col2 = st.sidebar.columns(2)
        with col1:
            I = st.slider("Input Current (I)", 0.0, 20.0, st.session_state.get('fn_I', 0.5), 0.1)
            st.session_state.fn_I = I
            
        with col2:
            I = st.number_input("I value", 0.0, 20.0, st.session_state.fn_I, 0.1, key="fn_I_input")
            st.session_state.fn_I = I
        
        st.sidebar.subheader("Initial Conditions")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            v0 = st.slider("Initial Membrane Potential (v₀)", -20.0, 20.0, st.session_state.get('fn_v0', 0.0), 0.1)
            st.session_state.fn_v0 = v0
            
        with col2:
            v0 = st.number_input("v₀ value", -20.0, 20.0, st.session_state.fn_v0, 0.1, key="fn_v0_input")
            st.session_state.fn_v0 = v0
            
        col1, col2 = st.sidebar.columns(2)
        with col1:
            w0 = st.slider("Initial Recovery Variable (w₀)", -20.0, 20.0, st.session_state.get('fn_w0', 0.0), 0.1)
            st.session_state.fn_w0 = w0
            
        with col2:
            w0 = st.number_input("w₀ value", -20.0, 20.0, st.session_state.fn_w0, 0.1, key="fn_w0_input")
            st.session_state.fn_w0 = w0
        
        # Create parameters object for current system
        system_params = (a, b, tau, I, v0, w0)
        
    elif system_type == "Coupled Linear System":
        st.sidebar.subheader("Equation Parameters")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            alpha = st.slider("Alpha (α) coefficient", -50.0, 50.0, st.session_state.get('cl_alpha', 1.0), 0.1)
            st.session_state.cl_alpha = alpha
            
        with col2:
            alpha = st.number_input("α value", -50.0, 50.0, st.session_state.cl_alpha, 0.1, key="cl_alpha_input")
            st.session_state.cl_alpha = alpha
            
        col1, col2 = st.sidebar.columns(2)
        with col1:
            beta = st.slider("Beta (β) coefficient", -50.0, 50.0, st.session_state.get('cl_beta', -1.0), 0.1)
            st.session_state.cl_beta = beta
            
        with col2:
            beta = st.number_input("β value", -50.0, 50.0, st.session_state.cl_beta, 0.1, key="cl_beta_input")
            st.session_state.cl_beta = beta
        
        st.sidebar.subheader("Initial Conditions")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            x0 = st.slider("Initial x value (x₀)", -50.0, 50.0, st.session_state.get('cl_x0', 1.0), 0.1)
            st.session_state.cl_x0 = x0
            
        with col2:
            x0 = st.number_input("x₀ value", -50.0, 50.0, st.session_state.cl_x0, 0.1, key="cl_x0_input")
            st.session_state.cl_x0 = x0
            
        col1, col2 = st.sidebar.columns(2)
        with col1:
            y0 = st.slider("Initial y value (y₀)", -50.0, 50.0, st.session_state.get('cl_y0', 0.0), 0.1)
            st.session_state.cl_y0 = y0
            
        with col2:
            y0 = st.number_input("y₀ value", -50.0, 50.0, st.session_state.cl_y0, 0.1, key="cl_y0_input")
            st.session_state.cl_y0 = y0
        
        # Create parameters object for current system
        system_params = (alpha, beta, x0, y0)
        
    elif system_type == "Van der Pol Oscillator":
        st.sidebar.subheader("Equation Parameters")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            mu = st.slider("Nonlinearity Parameter (μ)", 0.1, 50.0, st.session_state.van_der_pol_params['mu'], 0.1)
            st.session_state.van_der_pol_params['mu'] = mu
            
        with col2:
            mu = st.number_input("μ value", 0.1, 50.0, st.session_state.van_der_pol_params['mu'], 0.1)
            st.session_state.van_der_pol_params['mu'] = mu
        
        st.sidebar.subheader("Initial Conditions")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            x0 = st.slider("Initial Position (x₀)", -50.0, 50.0, st.session_state.van_der_pol_params['x0'], 0.1)
            st.session_state.van_der_pol_params['x0'] = x0
            
        with col2:
            x0 = st.number_input("x₀ value", -50.0, 50.0, st.session_state.van_der_pol_params['x0'], 0.1)
            st.session_state.van_der_pol_params['x0'] = x0
            
        col1, col2 = st.sidebar.columns(2)
        with col1:
            y0 = st.slider("Initial Velocity (y₀)", -50.0, 50.0, st.session_state.van_der_pol_params['y0'], 0.1)
            st.session_state.van_der_pol_params['y0'] = y0
            
        with col2:
            y0 = st.number_input("y₀ value", -50.0, 50.0, st.session_state.van_der_pol_params['y0'], 0.1)
            st.session_state.van_der_pol_params['y0'] = y0
        
        # Create parameters object
        system_params = (mu, x0, y0)
        
    elif system_type == "Duffing Oscillator":
        st.sidebar.subheader("Equation Parameters")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            alpha = st.slider("Linear Stiffness (α)", -50.0, 50.0, st.session_state.duffing_params['alpha'], 0.1)
            st.session_state.duffing_params['alpha'] = alpha
            
        with col2:
            alpha = st.number_input("α value", -50.0, 50.0, st.session_state.duffing_params['alpha'], 0.1)
            st.session_state.duffing_params['alpha'] = alpha
            
        col1, col2 = st.sidebar.columns(2)
        with col1:
            beta = st.slider("Nonlinear Stiffness (β)", -50.0, 50.0, st.session_state.duffing_params['beta'], 0.1)
            st.session_state.duffing_params['beta'] = beta
            
        with col2:
            beta = st.number_input("β value", -50.0, 50.0, st.session_state.duffing_params['beta'], 0.1)
            st.session_state.duffing_params['beta'] = beta
            
        col1, col2 = st.sidebar.columns(2)
        with col1:
            delta = st.slider("Damping (δ)", 0.0, 50.0, st.session_state.duffing_params['delta'], 0.01)
            st.session_state.duffing_params['delta'] = delta
            
        with col2:
            delta = st.number_input("δ value", 0.0, 50.0, st.session_state.duffing_params['delta'], 0.01)
            st.session_state.duffing_params['delta'] = delta
            
        col1, col2 = st.sidebar.columns(2)
        with col1:
            gamma = st.slider("Forcing Amplitude (γ)", 0.0, 80.0, st.session_state.duffing_params['gamma'], 0.1)
            st.session_state.duffing_params['gamma'] = gamma
            
        with col2:
            gamma = st.number_input("γ value", 0.0, 80.0, st.session_state.duffing_params['gamma'], 0.1)
            st.session_state.duffing_params['gamma'] = gamma
            
        col1, col2 = st.sidebar.columns(2)
        with col1:
            omega = st.slider("Forcing Frequency (ω)", 0.01, 50.0, st.session_state.duffing_params['omega'], 0.01)
            st.session_state.duffing_params['omega'] = omega
            
        with col2:
            omega = st.number_input("ω value", 0.01, 50.0, st.session_state.duffing_params['omega'], 0.01)
            st.session_state.duffing_params['omega'] = omega
        
        st.sidebar.subheader("Initial Conditions")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            x0 = st.slider("Initial Position (x₀)", -50.0, 50.0, st.session_state.duffing_params['x0'], 0.1)
            st.session_state.duffing_params['x0'] = x0
            
        with col2:
            x0 = st.number_input("x₀ value", -50.0, 50.0, st.session_state.duffing_params['x0'], 0.1)
            st.session_state.duffing_params['x0'] = x0
            
        col1, col2 = st.sidebar.columns(2)
        with col1:
            y0 = st.slider("Initial Velocity (y₀)", -50.0, 50.0, st.session_state.duffing_params['y0'], 0.1)
            st.session_state.duffing_params['y0'] = y0
            
        with col2:
            y0 = st.number_input("y₀ value", -50.0, 50.0, st.session_state.duffing_params['y0'], 0.1)
            st.session_state.duffing_params['y0'] = y0
        
        # Create parameters object
        system_params = (alpha, beta, delta, gamma, omega, x0, y0)
        
    elif system_type == "Double Pendulum":
        st.sidebar.subheader("Physical Parameters")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            g = st.slider("Gravity (g)", 1.0, 100.0, st.session_state.double_pendulum_params['g'], 0.1)
            st.session_state.double_pendulum_params['g'] = g
            
        with col2:
            g = st.number_input("g value", 1.0, 100.0, st.session_state.double_pendulum_params['g'], 0.1)
            st.session_state.double_pendulum_params['g'] = g
            
        col1, col2 = st.sidebar.columns(2)
        with col1:
            m1 = st.slider("Mass 1 (m₁)", 0.1, 10.0, st.session_state.double_pendulum_params['m1'], 0.1)
            st.session_state.double_pendulum_params['m1'] = m1
            
        with col2:
            m1 = st.number_input("m₁ value", 0.1, 10.0, st.session_state.double_pendulum_params['m1'], 0.1)
            st.session_state.double_pendulum_params['m1'] = m1
            
        col1, col2 = st.sidebar.columns(2)
        with col1:
            m2 = st.slider("Mass 2 (m₂)", 0.1, 10.0, st.session_state.double_pendulum_params['m2'], 0.1)
            st.session_state.double_pendulum_params['m2'] = m2
            
        with col2:
            m2 = st.number_input("m₂ value", 0.1, 10.0, st.session_state.double_pendulum_params['m2'], 0.1)
            st.session_state.double_pendulum_params['m2'] = m2
            
        col1, col2 = st.sidebar.columns(2)
        with col1:
            l1 = st.slider("Length 1 (l₁)", 0.1, 10.0, st.session_state.double_pendulum_params['l1'], 0.1)
            st.session_state.double_pendulum_params['l1'] = l1
            
        with col2:
            l1 = st.number_input("l₁ value", 0.1, 10.0, st.session_state.double_pendulum_params['l1'], 0.1)
            st.session_state.double_pendulum_params['l1'] = l1
            
        col1, col2 = st.sidebar.columns(2)
        with col1:
            l2 = st.slider("Length 2 (l₂)", 0.1, 10.0, st.session_state.double_pendulum_params['l2'], 0.1)
            st.session_state.double_pendulum_params['l2'] = l2
            
        with col2:
            l2 = st.number_input("l₂ value", 0.1, 10.0, st.session_state.double_pendulum_params['l2'], 0.1)
            st.session_state.double_pendulum_params['l2'] = l2
            
        st.sidebar.subheader("Initial Conditions")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            theta1 = st.slider("Initial Angle 1 (θ₁, rad)", -np.pi, np.pi, st.session_state.double_pendulum_params['theta1'], 0.1)
            st.session_state.double_pendulum_params['theta1'] = theta1
            
        with col2:
            theta1 = st.number_input("θ₁ value", -np.pi, np.pi, st.session_state.double_pendulum_params['theta1'], 0.1)
            st.session_state.double_pendulum_params['theta1'] = theta1
            
        col1, col2 = st.sidebar.columns(2)
        with col1:
            omega1 = st.slider("Initial Angular Velocity 1 (ω₁, rad/s)", -10.0, 10.0, st.session_state.double_pendulum_params['omega1'], 0.1)
            st.session_state.double_pendulum_params['omega1'] = omega1
            
        with col2:
            omega1 = st.number_input("ω₁ value", -10.0, 10.0, st.session_state.double_pendulum_params['omega1'], 0.1)
            st.session_state.double_pendulum_params['omega1'] = omega1
            
        col1, col2 = st.sidebar.columns(2)
        with col1:
            theta2 = st.slider("Initial Angle 2 (θ₂, rad)", -np.pi, np.pi, st.session_state.double_pendulum_params['theta2'], 0.1)
            st.session_state.double_pendulum_params['theta2'] = theta2
            
        with col2:
            theta2 = st.number_input("θ₂ value", -np.pi, np.pi, st.session_state.double_pendulum_params['theta2'], 0.1)
            st.session_state.double_pendulum_params['theta2'] = theta2
            
        col1, col2 = st.sidebar.columns(2)
        with col1:
            omega2 = st.slider("Initial Angular Velocity 2 (ω₂, rad/s)", -10.0, 10.0, st.session_state.double_pendulum_params['omega2'], 0.1)
            st.session_state.double_pendulum_params['omega2'] = omega2
            
        with col2:
            omega2 = st.number_input("ω₂ value", -10.0, 10.0, st.session_state.double_pendulum_params['omega2'], 0.1)
            st.session_state.double_pendulum_params['omega2'] = omega2
        
        # Create parameters object
        system_params = (g, m1, m2, l1, l2, theta1, omega1, theta2, omega2)
        
    elif system_type == "Lorenz System":
        st.sidebar.subheader("Equation Parameters")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            sigma = st.slider("Sigma (σ)", 0.1, 100.0, st.session_state.lorenz_params['sigma'], 0.1)
            st.session_state.lorenz_params['sigma'] = sigma
            
        with col2:
            sigma = st.number_input("σ value", 0.1, 100.0, st.session_state.lorenz_params['sigma'], 0.1)
            st.session_state.lorenz_params['sigma'] = sigma
            
        col1, col2 = st.sidebar.columns(2)
        with col1:
            rho = st.slider("Rho (ρ)", 0.1, 100.0, st.session_state.lorenz_params['rho'], 0.1)
            st.session_state.lorenz_params['rho'] = rho
            
        with col2:
            rho = st.number_input("ρ value", 0.1, 100.0, st.session_state.lorenz_params['rho'], 0.1)
            st.session_state.lorenz_params['rho'] = rho
            
        col1, col2 = st.sidebar.columns(2)
        with col1:
            beta = st.slider("Beta (β)", 0.1, 100.0, st.session_state.lorenz_params['beta'], 0.1)
            st.session_state.lorenz_params['beta'] = beta
            
        with col2:
            beta = st.number_input("β value", 0.1, 100.0, st.session_state.lorenz_params['beta'], 0.1)
            st.session_state.lorenz_params['beta'] = beta
            
        st.sidebar.subheader("Initial Conditions")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            x0 = st.slider("Initial x₀", -50.0, 50.0, st.session_state.lorenz_params['x0'], 0.1)
            st.session_state.lorenz_params['x0'] = x0
            
        with col2:
            x0 = st.number_input("x₀ value", -50.0, 50.0, st.session_state.lorenz_params['x0'], 0.1)
            st.session_state.lorenz_params['x0'] = x0
            
        col1, col2 = st.sidebar.columns(2)
        with col1:
            y0 = st.slider("Initial y₀", -50.0, 50.0, st.session_state.lorenz_params['y0'], 0.1)
            st.session_state.lorenz_params['y0'] = y0
            
        with col2:
            y0 = st.number_input("y₀ value", -50.0, 50.0, st.session_state.lorenz_params['y0'], 0.1)
            st.session_state.lorenz_params['y0'] = y0
            
        col1, col2 = st.sidebar.columns(2)
        with col1:
            z0 = st.slider("Initial z₀", -50.0, 50.0, st.session_state.lorenz_params['z0'], 0.1)
            st.session_state.lorenz_params['z0'] = z0
            
        with col2:
            z0 = st.number_input("z₀ value", -50.0, 50.0, st.session_state.lorenz_params['z0'], 0.1)
            st.session_state.lorenz_params['z0'] = z0
            
        # Poincaré section parameters for 3D systems
        st.sidebar.subheader("Poincaré Section")
        axis_options = ["x", "y", "z"]
        axis_index = axis_options.index(st.session_state.poincare_params['axis']) if st.session_state.poincare_params['axis'] in axis_options else 2
        
        st.session_state.poincare_params['axis'] = st.sidebar.selectbox(
            "Section Axis", 
            axis_options,
            index=axis_index
        )
            
        col1, col2 = st.sidebar.columns(2)
        with col1:
            value = st.slider("Section Value", -50.0, 50.0, st.session_state.poincare_params['value'], 0.1)
            st.session_state.poincare_params['value'] = value
            
        with col2:
            value = st.number_input("Value", -50.0, 50.0, st.session_state.poincare_params['value'], 0.1)
            st.session_state.poincare_params['value'] = value
            
        direction_options = [(1, "Positive"), (-1, "Negative"), (0, "Both")]
        direction_index = [d[0] for d in direction_options].index(st.session_state.poincare_params['direction']) if st.session_state.poincare_params['direction'] in [d[0] for d in direction_options] else 0
        
        st.session_state.poincare_params['direction'] = st.sidebar.selectbox(
            "Crossing Direction", 
            direction_options,
            index=direction_index,
            format_func=lambda x: x[1]
        )[0]
            
        # Create parameters object
        system_params = (sigma, rho, beta, x0, y0, z0)
    
    # Solve the selected system based on parameters
    def solve_selected_system():
        if system_type == "Harmonic Oscillator":
            return solve_ho(omega, gamma, y0=np.array([x0, v0]), t=times)
        
        elif system_type == "Sinusoidal Function":
            return solve_sinusoidal(amplitude, frequency, phase, use_cos, t=times)
        
        elif system_type == "Linear Function":
            return solve_linear(slope, intercept, t=times)
        
        elif system_type == "Lotka-Volterra":
            return solve_lotka_volterra(alpha, beta, delta, gamma, 
                                      y0=np.array([prey0, predator0]), t=times)
        
        elif system_type == "FitzHugh-Nagumo":
            return solve_fitzhugh_nagumo(a, b, tau, I, 
                                        y0=np.array([v0, w0]), t=times)
        
        elif system_type == "Coupled Linear System":
            return solve_coupled_linear(alpha, beta, 
                                      y0=np.array([x0, y0]), t=times)
        
        elif system_type == "Van der Pol Oscillator":
            return solve_van_der_pol(mu, y0=np.array([x0, y0]), t=times)
        
        elif system_type == "Duffing Oscillator":
            return solve_duffing(alpha, beta, delta, gamma, omega,
                               y0=np.array([x0, y0]), t=times)
        
        elif system_type == "Double Pendulum":
            return solve_double_pendulum(g, m1, m2, l1, l2,
                                       y0=np.array([theta1, omega1, theta2, omega2]), t=times)
        
        elif system_type == "Lorenz System":
            return solve_lorenz(sigma, rho, beta,
                              y0=np.array([x0, y0, z0]), t=times)
        
        return None
        
    # Check if parameters have changed
    current_params = (system_type, tuple(times), system_params)
    params_changed = (st.session_state.current_params != current_params)
    
    # Solve the system if parameters changed or no solution exists
    if params_changed or st.session_state.current_solution is None:
        solution = solve_selected_system()
        st.session_state.current_params = current_params
        st.session_state.current_solution = solution
        
        # Clear existing activations and features since parameters changed
        st.session_state.current_activations = None
        st.session_state.current_latent_features = None
        st.session_state.learned_equation = None
    else:
        solution = st.session_state.current_solution
    
    if solution:
        # Collect activations
        if st.session_state.current_activations is None:
            with st.spinner(f"Collecting neural network activations for {st.session_state.activation_site}.{st.session_state.activation_component}..."):
                model = st.session_state.model
                activations = get_activations(model, solution)
                st.session_state.current_activations = activations
        else:
            activations = st.session_state.current_activations
        
        # Check if activations were successfully collected
        if activations is None:
            st.error("""
            ### No Activations Collected
            
            The instrumentation failed to collect activations from the model. This could be due to:
            
            1. Missing required libraries (mishax)
            2. Incompatible model structure 
            3. AST patching failed to find matching patterns
            
            Check the console output for more details.
            """)
            st.stop()  # Stop execution of the app
        
        # Apply SAE only if needed
        if st.session_state.current_latent_features is None:
            with st.spinner("Applying sparse autoencoder..."):
                sae_model = st.session_state.sae_model
                latent_features = apply_sae(sae_model, activations)
                st.session_state.current_latent_features = latent_features
        else:
            latent_features = st.session_state.current_latent_features
        
        # Get time point and feature index from session state or use defaults
        time_point = st.session_state.time_point
        feature_idx = st.session_state.feature_idx
        
        # Option to highlight top activations on solution chart
        highlight_on_solution = st.checkbox("Highlight time points with top feature activations on solution chart", True)
        top_n_global = st.slider("Number of top activations to highlight", 10, 1000, 100, step=10)
        
        # Find top activation times for highlighting
        top_activation_times = []
        top_activation_values = []
        if highlight_on_solution:
            # Find top N activations globally
            flattened = latent_features.flatten()
            threshold = np.sort(flattened)[-min(top_n_global, len(flattened))]
            
            # Find time points with activations over threshold and their values
            for t in range(latent_features.shape[0]):
                # Get maximum activation at this time point
                max_val = np.max(latent_features[t, :])
                if max_val >= threshold:
                    top_activation_times.append(t)
                    top_activation_values.append(max_val)
        
        # Plot the solution and activations
        st.subheader(f"{system_type} Solution and Feature Activations")
        st.write(f"Equation: {solution['equations']}")
        
        # Get the time points
        times = solution['time_points']
        
        # Create labels and derivatives based on the system type
        if system_type == "Harmonic Oscillator":
            var1_name = "Position (x)"
            var2_name = "Velocity (dx/dt)"
            var1 = solution['solution'][:, 0]
            var2 = solution['solution'][:, 1]
            # Calculate acceleration
            var3_name = "Acceleration (d²x/dt²)"
            var3 = -omega**2 * var1 - gamma * var2
            
        elif system_type == "Sinusoidal Function":
            var1_name = "Value (y)"
            var2_name = "Derivative (dy/dt)"
            var1 = solution['solution'][:, 0]
            var2 = solution['solution'][:, 1]
            var3_name = "Second Derivative (d²y/dt²)"
            # Calculate second derivative
            var3 = -amplitude * frequency**2 * (np.sin(frequency * times + phase) if not use_cos else np.cos(frequency * times + phase))
            
        elif system_type == "Linear Function":
            var1_name = "Value (y)"
            var2_name = "Derivative (dy/dt)"
            var1 = solution['solution'][:, 0]
            var2 = solution['solution'][:, 1]
            var3_name = "Second Derivative (d²y/dt²)"
            # Second derivative is zero for linear function
            var3 = np.zeros_like(times)
            
        elif system_type == "Lotka-Volterra":
            var1_name = "Prey Population (x)"
            var2_name = "Predator Population (y)"
            var1 = solution['solution'][:, 0]
            var2 = solution['solution'][:, 1]
            var3_name = None
            var3 = None
            
        elif system_type == "FitzHugh-Nagumo":
            var1_name = "Membrane Potential (v)"
            var2_name = "Recovery Variable (w)"
            var1 = solution['solution'][:, 0]
            var2 = solution['solution'][:, 1]
            var3_name = None
            var3 = None
            
        elif system_type == "Coupled Linear System":
            var1_name = "x"
            var2_name = "y"
            var1 = solution['solution'][:, 0]
            var2 = solution['solution'][:, 1]
            var3_name = None
            var3 = None
            
        elif system_type == "Van der Pol Oscillator":
            var1_name = "Position (x)"
            var2_name = "Velocity (y)"
            var1 = solution['solution'][:, 0]
            var2 = solution['solution'][:, 1]
            var3_name = None
            var3 = None
            
        elif system_type == "Duffing Oscillator":
            var1_name = "Position (x)"
            var2_name = "Velocity (y)"
            var1 = solution['solution'][:, 0]
            var2 = solution['solution'][:, 1]
            var3_name = None
            var3 = None
            
        elif system_type == "Double Pendulum":
            var1_name = "Angle 1 (θ₁)"
            var2_name = "Angular Velocity 1 (ω₁)"
            var3_name = "Angle 2 (θ₂)"
            var4_name = "Angular Velocity 2 (ω₂)"
            var1 = solution['solution'][:, 0]
            var2 = solution['solution'][:, 1]
            var3 = solution['solution'][:, 2]
            var4 = solution['solution'][:, 3]
            
        elif system_type == "Lorenz System":
            var1_name = "x"
            var2_name = "y" 
            var3_name = "z"
            var1 = solution['solution'][:, 0]
            var2 = solution['solution'][:, 1]
            var3 = solution['solution'][:, 2]
        
        # Create feature activation plot
        feature_values = latent_features[:, feature_idx]
        
        # Solution plot - full width
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 5))
        ax1.plot(times, var1, label=var1_name, linewidth=2)
        ax1.plot(times, var2, label=var2_name, linewidth=2)
        
        if var3 is not None and system_type not in ["Double Pendulum", "Lorenz System"]:
            ax1.plot(times, var3, label=var3_name, linewidth=1.5, linestyle='--')
            
        elif system_type == "Double Pendulum":
            ax1.plot(times, var3, label=var3_name, linewidth=1.5)
            ax1.plot(times, var4, label=var4_name, linewidth=1.5, linestyle='--')
            
        elif system_type == "Lorenz System":
            ax1.plot(times, var3, label=var3_name, linewidth=1.5)
            
        # Add colorized vertical lines for top activations if available
        if highlight_on_solution and top_activation_values:
            min_val = min(top_activation_values)
            max_val = max(top_activation_values)
            norm = plt.Normalize(min_val, max_val)
            cmap = plt.cm.cool  # Use a visually appealing colormap
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            
            # Add slim, color-coded lines
            for i, t_idx in enumerate(top_activation_times):
                t = times[t_idx]
                color = cmap(norm(top_activation_values[i]))
                ax1.axvline(x=t, color=color, alpha=0.6, linewidth=0.8)
            
            # Add a colorbar
            cbar = plt.colorbar(sm, ax=ax1)
            cbar.set_label('Activation Value')
            
            st.caption(f"Found {len(top_activation_times)} time points with top activations ≥ {threshold:.4f}")
        
        # Highlight the selected time point
        if 'time_point' in locals() and time_point < len(times):
            selected_time = times[time_point]
            ax1.axvline(x=selected_time, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
            ax1.plot([selected_time], [var1[time_point]], 'ro', markersize=6)
            ax1.plot([selected_time], [var2[time_point]], 'ro', markersize=6)
            if var3 is not None:
                ax1.plot([selected_time], [var3[time_point]], 'ro', markersize=6)
                if system_type == "Double Pendulum":
                    ax1.plot([selected_time], [var4[time_point]], 'ro', markersize=6)
        
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_title('System Solution')
        st.pyplot(fig1)
        
        # Feature activation plot - full width under solution plot
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 4))
        ax2.plot(solution['time_points'], feature_values)
        
        # Highlight selected time point
        selected_time = solution['time_points'][time_point]
        ax2.axvline(x=selected_time, color='r', linestyle='--', alpha=0.7)
        ax2.plot([selected_time], [feature_values[time_point]], 'ro', markersize=8)
        
        # Highlight top activation times if available
        if highlight_on_solution and len(top_activation_times) > 0:
            for t_idx in top_activation_times:
                t = solution['time_points'][t_idx]
                ax2.axvline(x=t, color='magenta', alpha=0.3, linewidth=1.5)
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel(f'Feature {feature_idx} Activation')
        ax2.set_title(f'Feature {feature_idx} Activation Over Time')
        ax2.grid(True)
        st.pyplot(fig2)
        
        # Always show phase portraits for all systems
        st.subheader("Phase Portrait")
        
        if system_type in ["Harmonic Oscillator", "Van der Pol Oscillator", "Duffing Oscillator"]:
            # Simple 2D phase portrait colored by feature activations
            fig_phase, ax_phase = plt.subplots(1, 1, figsize=(8, 8))
            
            # Create a colormap for feature activations
            norm = plt.Normalize(np.min(feature_values), np.max(feature_values))
            
            # Use scatter plot with color mapped to feature activation
            scatter = ax_phase.scatter(var1, var2, c=feature_values, cmap='viridis', 
                                    norm=norm, s=30, alpha=0.7)
            
            # Add a colorbar
            cbar = plt.colorbar(scatter, ax=ax_phase)
            cbar.set_label(f'Feature {feature_idx} Activation')
            
            # Also add a line to show trajectory
            ax_phase.plot(var1, var2, 'k-', alpha=0.3, linewidth=0.8)
            
            ax_phase.set_xlabel(var1_name)
            ax_phase.set_ylabel(var2_name)
            ax_phase.grid(True)
            ax_phase.set_title(f"{system_type} Phase Portrait (colored by feature activations)")
            
            # Mark starting point
            ax_phase.plot([var1[0]], [var2[0]], 'go', markersize=8, label='Start')
            
            # Mark current time point
            if time_point < len(var1):
                ax_phase.plot([var1[time_point]], [var2[time_point]], 'ro', markersize=8, label='Current')
                
            ax_phase.legend()
            st.pyplot(fig_phase)
            
        elif system_type in ["Coupled Linear System", "Lotka-Volterra", "FitzHugh-Nagumo"]:
            # Phase portrait colored by feature activations
            fig_phase, ax_phase = plt.subplots(1, 1, figsize=(8, 8))
            
            # Create a colormap for feature activations
            norm = plt.Normalize(np.min(feature_values), np.max(feature_values))
            
            # Use scatter plot with color mapped to feature activation
            scatter = ax_phase.scatter(var1, var2, c=feature_values, cmap='viridis', 
                                    norm=norm, s=30, alpha=0.7)
            
            # Add a colorbar
            cbar = plt.colorbar(scatter, ax=ax_phase)
            cbar.set_label(f'Feature {feature_idx} Activation')
            
            # Also add a line to show trajectory
            ax_phase.plot(var1, var2, 'k-', alpha=0.3, linewidth=0.8)
            
            ax_phase.set_xlabel(var1_name)
            ax_phase.set_ylabel(var2_name)
            ax_phase.grid(True)
            ax_phase.set_title(f"{system_type} Phase Portrait (colored by feature activations)")
            
            # Mark starting point
            ax_phase.plot([var1[0]], [var2[0]], 'go', markersize=8, label='Start')
            
            # Mark current time point
            if time_point < len(var1):
                ax_phase.plot([var1[time_point]], [var2[time_point]], 'ro', markersize=8, label='Current')
                
            ax_phase.legend()
            st.pyplot(fig_phase)
            
        elif system_type == "Double Pendulum":
            # Phase portrait with multiple representations
            fig_phase = plt.figure(figsize=(15, 8))
            
            # Create a colormap for feature activations
            norm = plt.Normalize(np.min(feature_values), np.max(feature_values))
            
            # 1. Theta1 vs Theta2
            ax1 = fig_phase.add_subplot(131)
            scatter1 = ax1.scatter(var1, var3, c=feature_values, cmap='viridis', 
                                  norm=norm, s=30, alpha=0.7)
            ax1.plot(var1, var3, 'k-', alpha=0.3, linewidth=0.5)  # Add trajectory line
            ax1.set_xlabel("Angle 1 (θ₁)")
            ax1.set_ylabel("Angle 2 (θ₂)")
            ax1.grid(True)
            ax1.set_title("θ₁ vs θ₂")
            # Add colorbar
            plt.colorbar(scatter1, ax=ax1, shrink=0.7)
            ax1.plot(var1[0], var3[0], 'go', markersize=8)
            if time_point < len(var1):
                ax1.plot(var1[time_point], var3[time_point], 'ro', markersize=8)
            
            # 2. Omega1 vs Omega2
            ax2 = fig_phase.add_subplot(132)
            scatter2 = ax2.scatter(var2, var4, c=feature_values, cmap='viridis', 
                                  norm=norm, s=30, alpha=0.7)
            ax2.plot(var2, var4, 'k-', alpha=0.3, linewidth=0.5)  # Add trajectory line
            ax2.set_xlabel("Angular Velocity 1 (ω₁)")
            ax2.set_ylabel("Angular Velocity 2 (ω₂)")
            ax2.grid(True)
            ax2.set_title("ω₁ vs ω₂")
            # Add colorbar
            plt.colorbar(scatter2, ax=ax2, shrink=0.7)
            ax2.plot(var2[0], var4[0], 'go', markersize=8)
            if time_point < len(var2):
                ax2.plot(var2[time_point], var4[time_point], 'ro', markersize=8)
            
            # 3. Cartesian trajectory if available
            if 'cartesian_solution' in solution:
                ax3 = fig_phase.add_subplot(133)
                cart_sol = solution['cartesian_solution']
                # Plot pendulum arms
                for t in range(0, len(times), max(1, len(times)//100)):  # Plot a subset of points
                    alpha = max(0.05, min(0.5, t/len(times)))
                    # Draw first pendulum arm
                    ax3.plot([0, cart_sol[0, t]], [0, cart_sol[1, t]], 'k-', alpha=alpha, linewidth=1)
                    # Draw second pendulum arm
                    ax3.plot([cart_sol[0, t], cart_sol[2, t]], [cart_sol[1, t], cart_sol[3, t]], 'k-', alpha=alpha, linewidth=1)
                
                # Plot the path of the second pendulum
                ax3.plot(cart_sol[2, :], cart_sol[3, :], 'b-', alpha=0.5, linewidth=1)
                
                # Highlight current position
                if time_point < cart_sol.shape[1]:
                    # Current arms as bold lines
                    ax3.plot([0, cart_sol[0, time_point]], [0, cart_sol[1, time_point]], 'r-', linewidth=2)
                    ax3.plot([cart_sol[0, time_point], cart_sol[2, time_point]], 
                             [cart_sol[1, time_point], cart_sol[3, time_point]], 'r-', linewidth=2)
                    
                    # Current positions as points
                    ax3.plot([cart_sol[0, time_point]], [cart_sol[1, time_point]], 'ro', markersize=8)
                    ax3.plot([cart_sol[2, time_point]], [cart_sol[3, time_point]], 'ro', markersize=8)
                
                ax3.set_xlabel("x position")
                ax3.set_ylabel("y position")
                ax3.grid(True)
                ax3.set_title("Cartesian Trajectory")
                ax3.set_aspect('equal')  # Equal aspect ratio
                
                # Set limits with some padding
                max_val = np.max(np.abs(cart_sol))
                ax3.set_xlim(-max_val*1.1, max_val*1.1)
                ax3.set_ylim(-max_val*1.1, max_val*1.1)
            
            plt.tight_layout()
            st.pyplot(fig_phase)
            
        elif system_type == "Lorenz System":
            # 3D Phase portrait colored by feature activations
            fig_phase = plt.figure(figsize=(10, 8))
            ax_phase = fig_phase.add_subplot(111, projection='3d')
            
            # Create a colormap for feature activations
            norm = plt.Normalize(np.min(feature_values), np.max(feature_values))
            
            # Plot points colored by feature activation
            scatter = ax_phase.scatter(var1, var2, var3, 
                                     c=feature_values, cmap='viridis', 
                                     norm=norm, s=20, alpha=0.7)
            
            # Add a colorbar
            cbar = fig_phase.colorbar(scatter, ax=ax_phase, shrink=0.6, pad=0.1)
            cbar.set_label(f'Feature {feature_idx} Activation')
            
            # Also plot the 3D trajectory as a thin line
            ax_phase.plot3D(var1, var2, var3, 'k-', linewidth=0.5, alpha=0.3)
            
            # Mark starting point
            ax_phase.scatter([var1[0]], [var2[0]], [var3[0]], color='green', s=100, label='Start')
            
            # Mark current time point
            if time_point < len(var1):
                ax_phase.scatter([var1[time_point]], [var2[time_point]], [var3[time_point]], 
                                color='red', s=100, label='Current')
                
            ax_phase.set_xlabel(var1_name)
            ax_phase.set_ylabel(var2_name)
            ax_phase.set_zlabel(var3_name)
            ax_phase.legend()
            ax_phase.set_title("Lorenz Attractor (colored by feature activations)")
            
            st.pyplot(fig_phase)
            
            # Poincaré section for Lorenz system
            st.subheader("Poincaré Section")
            
            # Compute Poincaré section
            section_axis = st.session_state.poincare_params['axis']
            section_value = st.session_state.poincare_params['value']
            section_direction = st.session_state.poincare_params['direction']
            
            poincare_points = compute_poincare_section(
                solution, 
                axis=section_axis, 
                value=section_value, 
                direction=section_direction
            )
            
            if len(poincare_points) > 0:
                fig_poincare, ax_poincare = plt.subplots(1, 1, figsize=(8, 8))
                
                # Get the other two axes (not the section axis)
                axis_indices = {'x': 0, 'y': 1, 'z': 2}
                section_idx = axis_indices[section_axis]
                other_indices = [i for i in range(3) if i != section_idx]
                other_names = [name for i, name in enumerate([var1_name, var2_name, var3_name]) if i != section_idx]
                
                # Plot the Poincaré section points
                ax_poincare.scatter(poincare_points[:, 0], poincare_points[:, 1], 
                                   s=20, alpha=0.7, c=np.arange(len(poincare_points)))
                
                ax_poincare.set_xlabel(other_names[0])
                ax_poincare.set_ylabel(other_names[1])
                ax_poincare.set_title(f"Poincaré Section at {section_axis}={section_value}")
                ax_poincare.grid(True)
                
                st.pyplot(fig_poincare)
                st.write(f"Found {len(poincare_points)} intersection points. Direction: {'Positive' if section_direction > 0 else 'Negative' if section_direction < 0 else 'Both'}")
            else:
                st.warning("No Poincaré section points found. Try adjusting the section parameters.")
        
        # Display learned equations alongside original equations
        st.subheader("Model Prediction")

        # Create columns for side-by-side comparison
        eq_col1, eq_col2 = st.columns(2)

        with eq_col1:
            st.markdown("#### Original Equation")
            st.markdown(f"```\n{solution['equations']}\n```")
            
            # Add explanation based on system type
            if system_type == "Harmonic Oscillator":
                st.markdown(f"""
                **Parameters:**
                - Natural Frequency (ω): {omega}
                - Damping Coefficient (γ): {gamma}
                - Initial Position (x₀): {x0}
                - Initial Velocity (v₀): {v0}
                """)
            elif system_type == "Sinusoidal Function":
                st.markdown(f"""
                **Parameters:**
                - Amplitude (A): {amplitude}
                - Frequency (ω): {frequency}
                - Phase (φ): {phase}
                - Function: {'Cosine' if use_cos else 'Sine'}
                """)
            elif system_type == "Linear Function":
                st.markdown(f"""
                **Parameters:**
                - Slope (m): {slope}
                - Intercept (b): {intercept}
                """)
            elif system_type == "Lotka-Volterra":
                st.markdown(f"""
                **Parameters:**
                - Prey Growth Rate (α): {alpha}
                - Predation Rate (β): {beta}
                - Predator Death Rate (δ): {delta}
                - Predator Growth from Prey (γ): {gamma}
                - Initial Prey: {prey0}
                - Initial Predator: {predator0}
                """)
            elif system_type == "FitzHugh-Nagumo":
                st.markdown(f"""
                **Parameters:**
                - Parameter a: {a}
                - Parameter b: {b}
                - Time Scale (τ): {tau}
                - Input Current (I): {I}
                - Initial v: {v0}
                - Initial w: {w0}
                """)
            elif system_type == "Coupled Linear System":
                st.markdown(f"""
                **Parameters:**
                - Alpha (α): {alpha}
                - Beta (β): {beta}
                - Initial x: {x0}
                - Initial y: {y0}
                """)
            elif system_type == "Van der Pol Oscillator":
                st.markdown(f"""
                **Parameters:**
                - Nonlinearity Parameter (μ): {mu}
                - Initial Position (x₀): {x0}
                - Initial Velocity (y₀): {y0}
                """)
            elif system_type == "Duffing Oscillator":
                st.markdown(f"""
                **Parameters:**
                - Linear Stiffness (α): {alpha}
                - Nonlinear Stiffness (β): {beta}
                - Damping (δ): {delta}
                - Forcing Amplitude (γ): {gamma}
                - Forcing Frequency (ω): {omega}
                - Initial Position (x₀): {x0}
                - Initial Velocity (y₀): {y0}
                """)
            elif system_type == "Double Pendulum":
                st.markdown(f"""
                **Parameters:**
                - Gravity (g): {g}
                - Mass 1 (m₁): {m1}
                - Mass 2 (m₂): {m2}
                - Length 1 (l₁): {l1}
                - Length 2 (l₂): {l2}
                - Initial Angle 1 (θ₁): {theta1:.2f} rad
                - Initial Angular Velocity 1 (ω₁): {omega1}
                - Initial Angle 2 (θ₂): {theta2:.2f} rad
                - Initial Angular Velocity 2 (ω₂): {omega2}
                """)
            elif system_type == "Lorenz System":
                st.markdown(f"""
                **Parameters:**
                - Sigma (σ): {sigma}
                - Rho (ρ): {rho}
                - Beta (β): {beta}
                - Initial x₀: {x0}
                - Initial y₀: {y0}
                - Initial z₀: {z0}
                """)

        with eq_col2:
            st.markdown("#### Learned Equation")
            
            # Add a button to run the prediction
            if st.button("Run ODEformer Prediction"):
                with st.spinner("Running model to learn equation..."):
                    # Get learned equations from model
                    learned_eq_result = get_learned_equations(model, solution)
                    
                    if learned_eq_result['success']:
                        # Format the equation nicely
                        st.markdown(f"```\n{learned_eq_result['equation_str']}\n```")
                        
                        # Store the result in session state
                        st.session_state.learned_equation = learned_eq_result
                        
                        # Add additional details from the result if available
                        if learned_eq_result['full_results'] and hasattr(learned_eq_result['full_results'], 'error'):
                            st.markdown(f"**Fit Error:** {learned_eq_result['full_results'].error:.6f}")
                    else:
                        st.error(learned_eq_result['equation_str'])
            else:
                # Show previous results if available
                if 'learned_equation' in st.session_state and st.session_state.learned_equation:
                    st.markdown(f"```\n{st.session_state.learned_equation['equation_str']}\n```")
                    
                    # Add additional details from the result if available
                    if (st.session_state.learned_equation['full_results'] and 
                        hasattr(st.session_state.learned_equation['full_results'], 'error')):
                        st.markdown(f"**Fit Error:** {st.session_state.learned_equation['full_results'].error:.6f}")
                else:
                    st.info("Click the button to run the ODEformer model and learn the equation for this system")

        # Add extra information from the model if available
        if 'learned_equation' in st.session_state and st.session_state.learned_equation and st.session_state.learned_equation['success']:
            with st.expander("Additional Model Details"):
                # Show beam search results if available
                if (st.session_state.learned_equation['full_results'] and 
                    isinstance(st.session_state.learned_equation['full_results'], list) and 
                    len(st.session_state.learned_equation['full_results']) > 1):
                    
                    st.markdown("#### Top Alternative Equations")
                    for i, eq in enumerate(st.session_state.learned_equation['full_results'][1:5], 2):  # Start from 2nd result
                        if hasattr(eq, 'error'):
                            st.markdown(f"{i}. `{eq}` (Error: {eq.error:.6f})")
                        else:
                            st.markdown(f"{i}. `{eq}`")
        
        # TIME POINT ANALYSIS SECTION
        st.subheader("Time Point Analysis")
        
        # Time point selection interface
        time_point = st.slider("Select Time Point", 0, latent_features.shape[0]-1, time_point)
        
        # Use horizontal layout for navigation buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("◀ Previous", key="dec_time"):
                time_point = max(0, time_point - 1)
        with col2:
            if st.button("Next ▶", key="inc_time"):
                time_point = min(latent_features.shape[0]-1, time_point + 1)
        with col3:
            time_input = st.number_input("Direct input:", min_value=0, 
                                    max_value=latent_features.shape[0]-1, 
                                    value=time_point,
                                    step=1)
            if time_input != time_point:
                time_point = time_input
        
        # Save time_point to session state
        st.session_state.time_point = time_point
        
        # System state information based on system type
        st.markdown("**System State:**")
        m_cols = st.columns(4)
        
        if system_type == "Harmonic Oscillator":
            position = solution['solution'][time_point, 0]
            velocity = solution['solution'][time_point, 1]
            acceleration = -omega**2 * position - gamma * velocity
            
            with m_cols[0]:
                st.metric("Position", f"{position:.4f}")
            with m_cols[1]:
                st.metric("Velocity", f"{velocity:.4f}")
            with m_cols[2]:
                st.metric("Acceleration", f"{acceleration:.4f}")
                
        elif system_type == "Sinusoidal Function":
            value = solution['solution'][time_point, 0]
            derivative = solution['solution'][time_point, 1]
            second_deriv = -amplitude * frequency**2 * (np.sin(frequency * times[time_point] + phase) if not use_cos else np.cos(frequency * times[time_point] + phase))
            
            with m_cols[0]:
                st.metric("Value", f"{value:.4f}")
            with m_cols[1]:
                st.metric("Derivative", f"{derivative:.4f}")
            with m_cols[2]:
                st.metric("Second Derivative", f"{second_deriv:.4f}")
                
        elif system_type == "Linear Function":
            value = solution['solution'][time_point, 0]
            derivative = solution['solution'][time_point, 1]
            
            with m_cols[0]:
                st.metric("Value", f"{value:.4f}")
            with m_cols[1]:
                st.metric("Derivative", f"{derivative:.4f}")
            with m_cols[2]:
                st.metric("Second Derivative", "0.0000")
                
        elif system_type == "Lotka-Volterra":
            prey = solution['solution'][time_point, 0]
            predator = solution['solution'][time_point, 1]
            
            with m_cols[0]:
                st.metric("Prey Population", f"{prey:.4f}")
            with m_cols[1]:
                st.metric("Predator Population", f"{predator:.4f}")
            with m_cols[2]:
                # Calculate rate of change
                prey_change = alpha * prey - beta * prey * predator
                st.metric("Prey Rate of Change", f"{prey_change:.4f}")
            
        elif system_type == "FitzHugh-Nagumo":
            v = solution['solution'][time_point, 0]
            w = solution['solution'][time_point, 1]
            
            with m_cols[0]:
                st.metric("Membrane Potential (v)", f"{v:.4f}")
            with m_cols[1]:
                st.metric("Recovery Variable (w)", f"{w:.4f}")
            with m_cols[2]:
                # Calculate rate of change
                v_change = v - v**3/3 - w + I
                st.metric("dv/dt", f"{v_change:.4f}")
                
        elif system_type == "Coupled Linear System":
            x = solution['solution'][time_point, 0]
            y = solution['solution'][time_point, 1]
            
            with m_cols[0]:
                st.metric("x value", f"{x:.4f}")
            with m_cols[1]:
                st.metric("y value", f"{y:.4f}")
            with m_cols[2]:
                # Calculate rates of change
                x_change = alpha * y
                y_change = beta * x
                st.metric("dx/dt", f"{x_change:.4f}")
                
        elif system_type == "Van der Pol Oscillator":
            x = solution['solution'][time_point, 0]
            y = solution['solution'][time_point, 1]
            
            with m_cols[0]:
                st.metric("Position (x)", f"{x:.4f}")
            with m_cols[1]:
                st.metric("Velocity (y)", f"{y:.4f}")
            with m_cols[2]:
                # Calculate rate of change
                y_change = mu * (1 - x**2) * y - x
                st.metric("dy/dt", f"{y_change:.4f}")
                
        elif system_type == "Duffing Oscillator":
            x = solution['solution'][time_point, 0]
            y = solution['solution'][time_point, 1]
            t_val = solution['time_points'][time_point]
            
            with m_cols[0]:
                st.metric("Position (x)", f"{x:.4f}")
            with m_cols[1]:
                st.metric("Velocity (y)", f"{y:.4f}")
            with m_cols[2]:
                # Calculate rate of change
                y_change = -delta * y - alpha * x - beta * x**3 + gamma * np.cos(omega * t_val)
                st.metric("dy/dt", f"{y_change:.4f}")
                
        elif system_type == "Double Pendulum":
            theta1 = solution['solution'][time_point, 0]
            omega1 = solution['solution'][time_point, 1]
            theta2 = solution['solution'][time_point, 2]
            omega2 = solution['solution'][time_point, 3]
            
            with m_cols[0]:
                st.metric("θ₁", f"{theta1:.4f}")
            with m_cols[1]:
                st.metric("ω₁", f"{omega1:.4f}")
            with m_cols[2]:
                st.metric("θ₂", f"{theta2:.4f}")
            with m_cols[3]:
                st.metric("ω₂", f"{omega2:.4f}")
                
        elif system_type == "Lorenz System":
            x = solution['solution'][time_point, 0]
            y = solution['solution'][time_point, 1]
            z = solution['solution'][time_point, 2]
            
            with m_cols[0]:
                st.metric("x", f"{x:.4f}")
            with m_cols[1]:
                st.metric("y", f"{y:.4f}")
            with m_cols[2]:
                st.metric("z", f"{z:.4f}")
            with m_cols[3]:
                # Calculate one of the rates of change
                x_change = sigma * (y - x)
                st.metric("dx/dt", f"{x_change:.4f}")
        
        # Show selected feature value
        st.metric(f"Feature {feature_idx} value", f"{feature_values[time_point]:.4f}")
        
        # FEATURE EXPLORATION SECTION
        st.subheader("Feature Exploration")
        
        # Feature selection interface
        feature_idx = st.slider("Select Feature Index", 0, latent_features.shape[1]-1, feature_idx)
        
        # Use horizontal layout for navigation buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("◀ Previous", key="dec_feature"):
                feature_idx = max(0, feature_idx - 1)
        with col2:
            if st.button("Next ▶", key="inc_feature"):
                feature_idx = min(latent_features.shape[1]-1, feature_idx + 1)
        with col3:
            feature_input = st.number_input("Direct input:", min_value=0, 
                                      max_value=latent_features.shape[1]-1, 
                                      value=feature_idx,
                                      step=1)
            if feature_input != feature_idx:
                feature_idx = feature_input
        
        # Save feature_idx to session state
        st.session_state.feature_idx = feature_idx
        
        # Raw activations option
        show_raw = st.checkbox("Show raw activations", False)
        if show_raw:
            # Add tabs for raw activations and activation explorer
            tab1, tab2 = st.tabs(["Raw Activations", "Activation Explorer"])
            
            with tab1:
                fig_raw, ax_raw = plt.subplots(1, 1, figsize=(12, 3))  # Full width
                ax_raw.plot(activations[time_point])
                ax_raw.set_xlabel('Neuron Index')
                ax_raw.set_ylabel('Activation Value')
                ax_raw.set_title(f'Raw Activations at Time Point {time_point} for {st.session_state.activation_site}.{st.session_state.activation_component}')
                ax_raw.grid(True)
                st.pyplot(fig_raw)
                
            with tab2:
                # Add activation explorer
                if st.session_state.all_collected_activations:
                    st.markdown("### Available Activation Sites and Components")
                    
                    # Create an expandable section for each site
                    for site in st.session_state.all_collected_activations:
                        components = list(st.session_state.all_collected_activations[site].keys())
                        with st.expander(f"Site: {site} ({len(components)} components)"):
                            for comp in components:
                                shapes = list(st.session_state.all_collected_activations[site][comp].keys())
                                shape_str = ", ".join([str(s) for s in shapes])
                                st.markdown(f"**{comp}**: Shapes: {shape_str}")
                                
                                # Add a button to select this site/component
                                col1, col2 = st.columns([3, 1])
                                with col2:
                                    if st.button("Select", key=f"select_{site}_{comp}"):
                                        st.session_state.activation_site = str(site)
                                        st.session_state.activation_component = comp
                                        st.session_state.current_activations = None
                                        st.session_state.current_latent_features = None
                                        st.experimental_rerun()
        
        # Top features bar chart
        st.markdown("### Top Features at Selected Time Point")
        
        # Get top features at this time point
        features_at_time = latent_features[time_point, :]
        sorted_indices = np.argsort(features_at_time)[::-1]
        top_n = st.slider("Number of top features to show", 10, 100, 50, step=5)
        top_indices = sorted_indices[:top_n]
        
        # Create bar chart with gradient coloring
        fig3, ax3 = plt.subplots(1, 1, figsize=(14, 4))
        
        # Create a colormap for the bars
        bar_colors = plt.cm.viridis(np.linspace(0, 1, len(top_indices)))
        
        # Create bars with thin spacing
        bars = ax3.bar(top_indices, features_at_time[top_indices], 
                      color=bar_colors, width=0.8, 
                      edgecolor='none', alpha=0.8)
        
        # Highlight the currently selected feature if it's in the top N
        if feature_idx in top_indices:
            feature_pos = np.where(top_indices == feature_idx)[0][0]
            bars[feature_pos].set_color('red')
            bars[feature_pos].set_edgecolor('black')
            bars[feature_pos].set_linewidth(1.5)
            st.caption(f"Feature {feature_idx} is highlighted in red")
        else:
            st.caption(f"Feature {feature_idx} is not among the top {top_n} features at this time point")
            
        ax3.set_xlabel('Feature Index')
        ax3.set_ylabel('Activation Value')
        ax3.set_title(f'Top {top_n} Feature Activations at Time Point {time_point}')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Use a cleaner background
        ax3.set_facecolor('#f8f9fa')
        fig3.patch.set_facecolor('#f8f9fa')
        
        st.pyplot(fig3)
        
        # Add top activations heatmap and table
        st.markdown("### All Features Heatmap")
        st.write("This heatmap shows all feature activations across all time points.")
        
        # Use same highlighting logic as before
        highlight_top_n = st.checkbox("Highlight top activations on heatmap", True)
        if highlight_top_n:
            # Already calculated threshold above
            if 'threshold' not in locals():
                flattened = latent_features.flatten()
                threshold = np.sort(flattened)[-min(top_n_global, len(flattened))]
            
            # Create a figure
            fig4, ax4 = plt.subplots(1, 1, figsize=(12, 4))
            
            # Get the base heatmap
            heatmap_data = latent_features.T
            hm = ax4.imshow(heatmap_data, aspect='auto', cmap='viridis')
            
            # Find top N activations globally
            rows, cols = np.where(heatmap_data >= threshold)
            values = heatmap_data[rows, cols]
            
            # Plot circles at top activation locations
            scatter = ax4.scatter(cols, rows, s=15, c='red', alpha=0.5, 
                                marker='o', edgecolors='white', linewidth=0.5)
            
            # Add annotation for the threshold value
            st.caption(f"Highlighting all activations >= {threshold:.4f}")
            
            # Add a counter showing how many highlighted points in each feature
            feature_counts = {}
            for r in rows:
                if r not in feature_counts:
                    feature_counts[r] = 0
                feature_counts[r] += 1
            
            current_feature_count = feature_counts.get(feature_idx, 0)
            st.caption(f"Feature {feature_idx} has {current_feature_count} highlighted activations")
            
            # Generate table data for the top activations
            top_activations_data = []
            for i in range(len(values)):
                f_idx = int(rows[i])
                t_idx = int(cols[i])
                actual_time = solution['time_points'][t_idx]
                
                # Get appropriate state variables based on system type
                if system_type == "Harmonic Oscillator":
                    var1_val = solution['solution'][t_idx, 0]  # position
                    var2_val = solution['solution'][t_idx, 1]  # velocity
                    data_entry = {
                        "Feature": f_idx,
                        "Time Point": t_idx,
                        "Time": f"{actual_time:.2f}",
                        "Activation": f"{values[i]:.4f}",
                        "Position": f"{var1_val:.4f}",
                        "Velocity": f"{var2_val:.4f}"
                    }
                elif system_type in ["Sinusoidal Function", "Linear Function"]:
                    var1_val = solution['solution'][t_idx, 0]  # value
                    var2_val = solution['solution'][t_idx, 1]  # derivative
                    data_entry = {
                        "Feature": f_idx,
                        "Time Point": t_idx,
                        "Time": f"{actual_time:.2f}",
                        "Activation": f"{values[i]:.4f}",
                        "Value": f"{var1_val:.4f}",
                        "Derivative": f"{var2_val:.4f}"
                    }
                elif system_type == "Lotka-Volterra":
                    var1_val = solution['solution'][t_idx, 0]  # prey
                    var2_val = solution['solution'][t_idx, 1]  # predator
                    data_entry = {
                        "Feature": f_idx,
                        "Time Point": t_idx,
                        "Time": f"{actual_time:.2f}",
                        "Activation": f"{values[i]:.4f}",
                        "Prey": f"{var1_val:.4f}",
                        "Predator": f"{var2_val:.4f}"
                    }
                elif system_type == "FitzHugh-Nagumo":
                    var1_val = solution['solution'][t_idx, 0]  # v
                    var2_val = solution['solution'][t_idx, 1]  # w
                    data_entry = {
                        "Feature": f_idx,
                        "Time Point": t_idx,
                        "Time": f"{actual_time:.2f}",
                        "Activation": f"{values[i]:.4f}",
                        "V": f"{var1_val:.4f}",
                        "W": f"{var2_val:.4f}"
                    }
                elif system_type == "Coupled Linear System":
                    var1_val = solution['solution'][t_idx, 0]  # x
                    var2_val = solution['solution'][t_idx, 1]  # y
                    data_entry = {
                        "Feature": f_idx,
                        "Time Point": t_idx,
                        "Time": f"{actual_time:.2f}",
                        "Activation": f"{values[i]:.4f}",
                        "X": f"{var1_val:.4f}",
                        "Y": f"{var2_val:.4f}"
                    }
                elif system_type == "Van der Pol Oscillator":
                    var1_val = solution['solution'][t_idx, 0]  # x
                    var2_val = solution['solution'][t_idx, 1]  # y
                    data_entry = {
                        "Feature": f_idx,
                        "Time Point": t_idx,
                        "Time": f"{actual_time:.2f}",
                        "Activation": f"{values[i]:.4f}",
                        "X": f"{var1_val:.4f}",
                        "Y": f"{var2_val:.4f}"
                    }
                elif system_type == "Duffing Oscillator":
                    var1_val = solution['solution'][t_idx, 0]  # x
                    var2_val = solution['solution'][t_idx, 1]  # y
                    data_entry = {
                        "Feature": f_idx,
                        "Time Point": t_idx,
                        "Time": f"{actual_time:.2f}",
                        "Activation": f"{values[i]:.4f}",
                        "X": f"{var1_val:.4f}",
                        "Y": f"{var2_val:.4f}"
                    }
                elif system_type == "Double Pendulum":
                    var1_val = solution['solution'][t_idx, 0]  # theta1
                    var2_val = solution['solution'][t_idx, 2]  # theta2
                    data_entry = {
                        "Feature": f_idx,
                        "Time Point": t_idx,
                        "Time": f"{actual_time:.2f}",
                        "Activation": f"{values[i]:.4f}",
                        "θ₁": f"{var1_val:.4f}",
                        "θ₂": f"{var2_val:.4f}"
                    }
                elif system_type == "Lorenz System":
                    var1_val = solution['solution'][t_idx, 0]  # x
                    var2_val = solution['solution'][t_idx, 1]  # y
                    var3_val = solution['solution'][t_idx, 2]  # z
                    data_entry = {
                        "Feature": f_idx,
                        "Time Point": t_idx,
                        "Time": f"{actual_time:.2f}",
                        "Activation": f"{values[i]:.4f}",
                        "X": f"{var1_val:.4f}",
                        "Y": f"{var2_val:.4f}",
                        "Z": f"{var3_val:.4f}"
                    }
                
                top_activations_data.append(data_entry)
            
            # Sort by activation value (descending)
            top_activations_data.sort(key=lambda x: float(x["Activation"]), reverse=True)
        else:
            # Regular heatmap without highlighting
            fig4, ax4 = plt.subplots(1, 1, figsize=(12, 4))
            hm = ax4.imshow(latent_features.T, aspect='auto', cmap='viridis')
            
        # Common plot elements
        ax4.set_xlabel('Time Point')
        ax4.set_ylabel('Feature Index')
        ax4.set_title('All Features Over Time')
        
        # Mark selected time point
        ax4.axvline(x=time_point, color='yellow', linestyle='--', alpha=0.7, linewidth=1)
        
        # Mark selected feature
        ax4.axhline(y=feature_idx, color='yellow', linestyle='--', alpha=0.7, linewidth=1)
            
        plt.colorbar(hm, ax=ax4, label='Activation Value')
        st.pyplot(fig4)
        
        # Show table of top activations if highlighting is enabled
        if highlight_top_n:
            # Display the table of top activations
            st.markdown("#### Top Activations Values")
            st.write(f"Showing all {len(top_activations_data)} activations above threshold {threshold:.4f}")
            
            # Use a dataframe for better formatting and sortability
            import pandas as pd
            df = pd.DataFrame(top_activations_data)
            
            # Allow sorting by clicking on column headers
            st.dataframe(df, use_container_width=True, height=min(400, len(df) * 35 + 38))
        
        # Add t-SNE and UMAP visualization section
        st.markdown("### Dimensionality Reduction Visualization")
        st.write("Visualize the latent feature vectors using dimensionality reduction techniques.")
        
        if sklearn_available and umap_available:
            dim_reduce_tabs = st.tabs(["t-SNE", "UMAP"])
            
            # Common color options for both tabs
            color_options = {
                "Time Point": np.arange(latent_features.shape[0]),
            }
            
            # Add system-specific color options
            if system_type == "Harmonic Oscillator":
                color_options["Position"] = solution['solution'][:, 0]
                color_options["Velocity"] = solution['solution'][:, 1]
            elif system_type in ["Sinusoidal Function", "Linear Function"]:
                color_options["Value"] = solution['solution'][:, 0]
                color_options["Derivative"] = solution['solution'][:, 1]
            elif system_type == "Lotka-Volterra":
                color_options["Prey Population"] = solution['solution'][:, 0]
                color_options["Predator Population"] = solution['solution'][:, 1]
            elif system_type == "FitzHugh-Nagumo":
                color_options["Membrane Potential"] = solution['solution'][:, 0]
                color_options["Recovery Variable"] = solution['solution'][:, 1]
            elif system_type == "Coupled Linear System":
                color_options["X Value"] = solution['solution'][:, 0]
                color_options["Y Value"] = solution['solution'][:, 1]
            elif system_type == "Van der Pol Oscillator":
                color_options["X Value"] = solution['solution'][:, 0]
                color_options["Y Value"] = solution['solution'][:, 1]
            elif system_type == "Duffing Oscillator":
                color_options["X Value"] = solution['solution'][:, 0]
                color_options["Y Value"] = solution['solution'][:, 1]
            elif system_type == "Double Pendulum":
                color_options["θ₁"] = solution['solution'][:, 0]
                color_options["ω₁"] = solution['solution'][:, 1]
                color_options["θ₂"] = solution['solution'][:, 2]
                color_options["ω₂"] = solution['solution'][:, 3]
            elif system_type == "Lorenz System":
                color_options["X Value"] = solution['solution'][:, 0]
                color_options["Y Value"] = solution['solution'][:, 1]
                color_options["Z Value"] = solution['solution'][:, 2]
            
            # Add the feature activation as a color option
            color_options[f"Feature {feature_idx} Activation"] = latent_features[:, feature_idx]
            
            # t-SNE Tab
            with dim_reduce_tabs[0]:
                st.subheader("t-SNE Visualization")
                
                tsne_col1, tsne_col2 = st.columns(2)
                
                with tsne_col1:
                    tsne_perplexity = st.slider("Perplexity", 5, 100, 30, step=5)
                    tsne_components = st.radio("Dimensions", [2, 3], horizontal=True)
                
                with tsne_col2:
                    # Set default color to feature activation instead of time point
                    default_color_index = list(color_options.keys()).index(f"Feature {feature_idx} Activation")
                    tsne_color_by = st.selectbox("Color by", list(color_options.keys()), index=default_color_index, key='tsne_color')
                    tsne_cmap = st.selectbox("Color map", ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'coolwarm'], index=0, key='tsne_cmap')
                
                if st.button("Generate t-SNE"):
                    with st.spinner("Computing t-SNE projection..."):
                        # Reshape latent features for t-SNE input (time_points, features)
                        tsne_input = latent_features.reshape(latent_features.shape[0], -1)
                        try:
                            tsne_result = compute_dimensionality_reduction(
                                tsne_input, 
                                method='tsne', 
                                n_components=tsne_components,
                                perplexity=tsne_perplexity
                            )
                            
                            # Create the plot
                            fig_tsne = plt.figure(figsize=(10, 8))
                            
                            if tsne_components == 2:
                                plt.scatter(
                                    tsne_result[:, 0], 
                                    tsne_result[:, 1], 
                                    c=color_options[tsne_color_by], 
                                    cmap=tsne_cmap
                                )
                                plt.colorbar(label=tsne_color_by)
                                plt.title(f't-SNE Visualization (perplexity={tsne_perplexity})')
                                plt.xlabel('t-SNE 1')
                                plt.ylabel('t-SNE 2')
                                
                                # Highlight the selected time point
                                plt.scatter(
                                    tsne_result[time_point, 0], 
                                    tsne_result[time_point, 1], 
                                    color='red', 
                                    s=100, 
                                    marker='x', 
                                    label=f'Time Point {time_point}'
                                )
                                plt.legend()
                            else:
                                # 3D plot for 3 components
                                ax = fig_tsne.add_subplot(111, projection='3d')
                                sc = ax.scatter(
                                    tsne_result[:, 0], 
                                    tsne_result[:, 1], 
                                    tsne_result[:, 2], 
                                    c=color_options[tsne_color_by], 
                                    cmap=tsne_cmap
                                )
                                plt.colorbar(sc, label=tsne_color_by)
                                ax.set_title(f't-SNE Visualization (perplexity={tsne_perplexity})')
                                ax.set_xlabel('t-SNE 1')
                                ax.set_ylabel('t-SNE 2')
                                ax.set_zlabel('t-SNE 3')
                                
                                # Highlight the selected time point
                                ax.scatter(
                                    tsne_result[time_point, 0], 
                                    tsne_result[time_point, 1], 
                                    tsne_result[time_point, 2], 
                                    color='red', 
                                    s=100, 
                                    marker='x', 
                                    label=f'Time Point {time_point}'
                                )
                                ax.legend()
                            
                            st.pyplot(fig_tsne)
                            
                            # Show data table with reduced dimensions
                            st.subheader("t-SNE Coordinates")
                            tsne_df = pd.DataFrame(tsne_result, columns=[f't-SNE {i+1}' for i in range(tsne_components)])
                            tsne_df['Time Point'] = np.arange(tsne_input.shape[0])
                            tsne_df[tsne_color_by] = color_options[tsne_color_by]
                            
                            st.dataframe(tsne_df)
                        
                        except Exception as e:
                            st.error(f"Error computing t-SNE: {str(e)}")
                            st.code(traceback.format_exc())
            
            # UMAP Tab
            with dim_reduce_tabs[1]:
                st.subheader("UMAP Visualization")
                
                umap_col1, umap_col2 = st.columns(2)
                
                with umap_col1:
                    umap_neighbors = st.slider("Number of Neighbors", 2, 100, 15, step=1)
                    umap_min_dist = st.slider("Minimum Distance", 0.01, 0.99, 0.1, step=0.01)
                    umap_components = st.radio("Dimensions", [2, 3], horizontal=True, key="umap_dim")
                
                with umap_col2:
                    # Set default color to feature activation instead of time point
                    default_color_index = list(color_options.keys()).index(f"Feature {feature_idx} Activation")
                    umap_color_by = st.selectbox("Color by", list(color_options.keys()), index=default_color_index, key='umap_color')
                    umap_cmap = st.selectbox("Color map", ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'coolwarm'], index=0, key='umap_cmap')
                
                if st.button("Generate UMAP"):
                    with st.spinner("Computing UMAP projection..."):
                        # Reshape latent features for UMAP input (time_points, features)
                        umap_input = latent_features.reshape(latent_features.shape[0], -1)
                        try:
                            umap_result = compute_dimensionality_reduction(
                                umap_input, 
                                method='umap', 
                                n_components=umap_components,
                                n_neighbors=umap_neighbors,
                                min_dist=umap_min_dist
                            )
                            
                            # Create the plot
                            fig_umap = plt.figure(figsize=(10, 8))
                            
                            if umap_components == 2:
                                plt.scatter(
                                    umap_result[:, 0], 
                                    umap_result[:, 1], 
                                    c=color_options[umap_color_by], 
                                    cmap=umap_cmap
                                )
                                plt.colorbar(label=umap_color_by)
                                plt.title(f'UMAP Visualization (n_neighbors={umap_neighbors}, min_dist={umap_min_dist})')
                                plt.xlabel('UMAP 1')
                                plt.ylabel('UMAP 2')
                                
                                # Highlight the selected time point
                                plt.scatter(
                                    umap_result[time_point, 0], 
                                    umap_result[time_point, 1], 
                                    color='red', 
                                    s=100, 
                                    marker='x', 
                                    label=f'Time Point {time_point}'
                                )
                                plt.legend()
                            else:
                                # 3D plot for 3 components
                                ax = fig_umap.add_subplot(111, projection='3d')
                                sc = ax.scatter(
                                    umap_result[:, 0], 
                                    umap_result[:, 1], 
                                    umap_result[:, 2], 
                                    c=color_options[umap_color_by], 
                                    cmap=umap_cmap
                                )
                                plt.colorbar(sc, label=umap_color_by)
                                ax.set_title(f'UMAP Visualization (n_neighbors={umap_neighbors}, min_dist={umap_min_dist})')
                                ax.set_xlabel('UMAP 1')
                                ax.set_ylabel('UMAP 2')
                                ax.set_zlabel('UMAP 3')
                                
                                # Highlight the selected time point
                                ax.scatter(
                                    umap_result[time_point, 0], 
                                    umap_result[time_point, 1], 
                                    umap_result[time_point, 2], 
                                    color='red', 
                                    s=100, 
                                    marker='x', 
                                    label=f'Time Point {time_point}'
                                )
                                ax.legend()
                            
                            st.pyplot(fig_umap)
                            
                            # Show data table with reduced dimensions
                            st.subheader("UMAP Coordinates")
                            umap_df = pd.DataFrame(umap_result, columns=[f'UMAP {i+1}' for i in range(umap_components)])
                            umap_df['Time Point'] = np.arange(umap_input.shape[0])
                            umap_df[umap_color_by] = color_options[umap_color_by]
                            
                            st.dataframe(umap_df)
                        
                        except Exception as e:
                            st.error(f"Error computing UMAP: {str(e)}")
                            st.code(traceback.format_exc())
        else:
            st.warning("""
            ⚠️ Dimensionality reduction libraries are not available. 
            
            To enable t-SNE and UMAP visualizations, install:
            
            ```
            pip install scikit-learn umap-learn
            ```
            """)
    else:
        st.error("Failed to solve the selected system with the given parameters.")
else:
    st.warning("Please add paths and load models using the sidebar controls.")

# Add some information about the app
st.sidebar.markdown("---")
st.sidebar.info(f"""
This app allows you to explore how neural network features respond to different dynamical systems.

**Current System**: {st.session_state.get('system_type', 'None')}
**Current Activation**: {st.session_state.get('activation_site', 'None')}.{st.session_state.get('activation_component', 'None')}
**Current SAE Layer**: {list(st.session_state.sae_paths.keys())[list(st.session_state.sae_paths.values()).index(st.session_state.current_sae_path)] if st.session_state.current_sae_path in st.session_state.sae_paths.values() else 'Custom'}

1. Use the sidebar to select a system and adjust parameters
2. Choose SAE model for different layers
3. Choose activation sites and components to analyze
4. Explore feature activations at specific time points
5. Analyze patterns in the neural network representations
""")