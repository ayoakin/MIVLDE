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

# Function to get activations for the currently selected site and component
def get_activations(model, solution):
    """Get activations for the currently selected site and component"""
    site_name = st.session_state.activation_site
    component = st.session_state.activation_component
    
    all_activations, activations = get_model_activations(
        model, solution, site_name=site_name, component=component
    )
    
    # Store all collected activations for component selection
    if all_activations:
        st.session_state.all_collected_activations = all_activations
        
        # Print info about collected activations to debug
        print(f"Collection complete. Found sites: {list(all_activations.keys())}")
        for site in all_activations:
            print(f"Site '{site}' has components: {list(all_activations[site].keys())}")
    
    # If collection failed with the selected site/component, try the defaults
    if activations is None and (site_name != 'RESIDUAL' or component != 'encoder.transformer.residual1'):
        st.warning(f"Failed to collect activations for {site_name}.{component}. Trying defaults.")
        _, activations = get_model_activations(model, solution)
        
        if activations is not None:
            # Update session state to reflect the actual values used
            st.session_state.activation_site = 'RESIDUAL'
            st.session_state.activation_component = 'encoder.transformer.residual1'
    
    return activations

# Function to process activation through SAE
def apply_sae(sae_model, activations):
    """Apply SAE to get latent features"""
    inputs = torch.tensor(activations, dtype=torch.float32)
    _, latent = sae_model(inputs)
    return latent.squeeze(0).detach().numpy()

# Main Streamlit app
st.title("Dynamic Systems Feature Explorer")

if st.session_state.models_loaded:
    # System selection
    st.sidebar.header("System Selection")
    system_type = st.sidebar.selectbox(
        "Select System Type",
        ["Harmonic Oscillator", "Sinusoidal Function", "Linear Function", "Lotka-Volterra", "FitzHugh-Nagumo", "Coupled Linear System"],
        index=["Harmonic Oscillator", "Sinusoidal Function", "Linear Function", "Lotka-Volterra", "FitzHugh-Nagumo", "Coupled Linear System"].index(st.session_state.system_type) if st.session_state.system_type in ["Harmonic Oscillator", "Sinusoidal Function", "Linear Function", "Lotka-Volterra", "FitzHugh-Nagumo", "Coupled Linear System"] else 0
    )
    st.session_state.system_type = system_type

    # Initial conditions and time span
    st.sidebar.header("Time Settings")
    t_start = st.sidebar.number_input("Start Time", value=0.0, step=1.0)
    
    # Adjust end time based on system type
    if system_type == "FitzHugh-Nagumo":
        t_end_default = 100.0
    elif system_type == "Lotka-Volterra":
        t_end_default = 20.0
    else:
        t_end_default = 10.0
        
    t_end = st.sidebar.number_input("End Time", value=t_end_default, step=1.0, min_value=t_start + 0.1)
    t_points = st.sidebar.slider("Number of Time Points", 10, 1000, 100, 10)
    times = np.linspace(t_start, t_end, t_points)

    # Parameter inputs based on system selection
    st.sidebar.header("System Parameters")
    
    if system_type == "Harmonic Oscillator":
        st.sidebar.subheader("Equation Parameters")
        omega = st.sidebar.slider("Natural Frequency (ω)", 0.1, 5.0, 1.0, 0.1)
        gamma = st.sidebar.slider("Damping Coefficient (γ)", 0.0, 5.0, 0.5, 0.1)
        
        st.sidebar.subheader("Initial Conditions")
        x0 = st.sidebar.slider("Initial Position (x₀)", -5.0, 5.0, 1.0, 0.1)
        v0 = st.sidebar.slider("Initial Velocity (v₀)", -5.0, 5.0, 0.0, 0.1)
        
        # Create parameters object for current system
        system_params = (omega, gamma, x0, v0)
        
    elif system_type == "Sinusoidal Function":
        st.sidebar.subheader("Function Parameters")
        amplitude = st.sidebar.slider("Amplitude (A)", 0.1, 5.0, 1.0, 0.1)
        frequency = st.sidebar.slider("Frequency (ω)", 0.1, 5.0, 1.0, 0.1)
        phase = st.sidebar.slider("Phase (φ, radians)", 0.0, 2*np.pi, 0.0, 0.1)
        use_cos = st.sidebar.checkbox("Use Cosine instead of Sine", False)
        
        # No initial conditions for pure functions
        
        # Create parameters object for current system
        system_params = (amplitude, frequency, phase, use_cos)
        
    elif system_type == "Linear Function":
        st.sidebar.subheader("Function Parameters")
        slope = st.sidebar.slider("Slope (m)", -5.0, 5.0, 1.0, 0.1)
        intercept = st.sidebar.slider("Intercept (b)", -5.0, 5.0, 0.0, 0.1)
        
        # No initial conditions for pure functions
        
        # Create parameters object for current system
        system_params = (slope, intercept)
        
    elif system_type == "Lotka-Volterra":
        st.sidebar.subheader("Equation Parameters")
        alpha = st.sidebar.slider("Prey Growth Rate (α)", 0.1, 2.0, 0.5, 0.1)
        beta = st.sidebar.slider("Predation Rate (β)", 0.01, 1.0, 0.2, 0.01)
        delta = st.sidebar.slider("Predator Death Rate (δ)", 0.1, 2.0, 0.5, 0.1)
        gamma = st.sidebar.slider("Predator Growth from Prey (γ)", 0.01, 1.0, 0.1, 0.01)
        
        st.sidebar.subheader("Initial Conditions")
        prey0 = st.sidebar.slider("Initial Prey Population (x₀)", 0.1, 5.0, 1.0, 0.1)
        predator0 = st.sidebar.slider("Initial Predator Population (y₀)", 0.1, 5.0, 0.5, 0.1)
        
        # Create parameters object for current system
        system_params = (alpha, beta, delta, gamma, prey0, predator0)
        
    elif system_type == "FitzHugh-Nagumo":
        st.sidebar.subheader("Equation Parameters")
        a = st.sidebar.slider("Parameter a", -1.0, 1.0, 0.7, 0.1)
        b = st.sidebar.slider("Parameter b", 0.1, 1.0, 0.8, 0.1)
        tau = st.sidebar.slider("Time Scale (τ)", 1.0, 20.0, 12.5, 0.5)
        I = st.sidebar.slider("Input Current (I)", 0.0, 2.0, 0.5, 0.1)
        
        st.sidebar.subheader("Initial Conditions")
        v0 = st.sidebar.slider("Initial Membrane Potential (v₀)", -2.0, 2.0, 0.0, 0.1)
        w0 = st.sidebar.slider("Initial Recovery Variable (w₀)", -2.0, 2.0, 0.0, 0.1)
        
        # Create parameters object for current system
        system_params = (a, b, tau, I, v0, w0)
        
    elif system_type == "Coupled Linear System":
        st.sidebar.subheader("Equation Parameters")
        alpha = st.sidebar.slider("Alpha (α) coefficient", -5.0, 5.0, 1.0, 0.1)
        beta = st.sidebar.slider("Beta (β) coefficient", -5.0, 5.0, -1.0, 0.1)
        
        st.sidebar.subheader("Initial Conditions")
        x0 = st.sidebar.slider("Initial x value (x₀)", -5.0, 5.0, 1.0, 0.1)
        y0 = st.sidebar.slider("Initial y value (y₀)", -5.0, 5.0, 0.0, 0.1)
        
        # Create parameters object for current system
        system_params = (alpha, beta, x0, y0)
    
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
    else:
        solution = st.session_state.current_solution
    
    if solution:
        # Plot the solution
        st.subheader(f"{system_type} Solution")
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
        
        # Collect activations only if needed
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
        
        # Create solution plot
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 5))
        ax1.plot(times, var1, label=var1_name, linewidth=2)
        ax1.plot(times, var2, label=var2_name, linewidth=2)
        
        if var3 is not None:
            ax1.plot(times, var3, label=var3_name, linewidth=1.5, linestyle='--')
            
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
        
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'{system_type} Solution with Feature Activations')
        st.pyplot(fig1)
        
        # Special plot for Lotka-Volterra - phase space
        if system_type == "Lotka-Volterra":
            st.subheader("Phase Space Plot")
            fig_phase, ax_phase = plt.subplots(1, 1, figsize=(8, 8))
            ax_phase.plot(var1, var2)
            ax_phase.set_xlabel(var1_name)
            ax_phase.set_ylabel(var2_name)
            ax_phase.grid(True)
            ax_phase.set_title("Lotka-Volterra Phase Space")
            # Mark starting point
            ax_phase.plot([var1[0]], [var2[0]], 'go', markersize=8, label='Start')
            # Mark current time point
            if time_point < len(var1):
                ax_phase.plot([var1[time_point]], [var2[time_point]], 'ro', markersize=8, label='Current')
            ax_phase.legend()
            st.pyplot(fig_phase)
        
        # Special plot for FitzHugh-Nagumo - phase space
        if system_type == "FitzHugh-Nagumo":
            st.subheader("Phase Space Plot")
            fig_phase, ax_phase = plt.subplots(1, 1, figsize=(8, 8))
            ax_phase.plot(var1, var2)
            ax_phase.set_xlabel(var1_name)
            ax_phase.set_ylabel(var2_name)
            ax_phase.grid(True)
            ax_phase.set_title("FitzHugh-Nagumo Phase Space")
            # Mark starting point
            ax_phase.plot([var1[0]], [var2[0]], 'go', markersize=8, label='Start')
            # Mark current time point
            if time_point < len(var1):
                ax_phase.plot([var1[time_point]], [var2[time_point]], 'ro', markersize=8, label='Current')
            ax_phase.legend()
            st.pyplot(fig_phase)
            
        # Special plot for Coupled Linear System - phase space
        if system_type == "Coupled Linear System":
            st.subheader("Phase Space Plot")
            fig_phase, ax_phase = plt.subplots(1, 1, figsize=(8, 8))
            ax_phase.plot(var1, var2)
            ax_phase.set_xlabel(var1_name)
            ax_phase.set_ylabel(var2_name)
            ax_phase.grid(True)
            ax_phase.set_title("Coupled Linear System Phase Space")
            # Mark starting point
            ax_phase.plot([var1[0]], [var2[0]], 'go', markersize=8, label='Start')
            # Mark current time point
            if time_point < len(var1):
                ax_phase.plot([var1[time_point]], [var2[time_point]], 'ro', markersize=8, label='Current')
            ax_phase.legend()
            st.pyplot(fig_phase)
        
        # COMBINED FEATURE EXPLORATION SECTION
        st.subheader("Feature Exploration")
        
        # Add spacing for better visual separation
        st.write("")
        
        # FEATURE SELECTION SECTION - FULL WIDTH
        st.markdown("### Feature Selection")
        
        # Feature selection interface - full width
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
        
        # Plot feature activation over time - FULL WIDTH
        feature_values = latent_features[:, feature_idx]
        fig2, ax2 = plt.subplots(1, 1, figsize=(12, 5))  # Increased size for better visibility
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
        
        # TIME POINT ANALYSIS SECTION - BELOW FEATURE CHART
        st.markdown("### Time Point Analysis")
        
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
        
        # Show selected feature value in the rightmost column for all system types
        with m_cols[3]:
            st.metric(f"Feature {feature_idx} value", f"{feature_values[time_point]:.4f}")
        
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
        
        # Create bar chart with gradient coloring - FULL WIDTH
        fig3, ax3 = plt.subplots(1, 1, figsize=(14, 4))  # Increased width
        
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