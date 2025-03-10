import streamlit as st
import numpy as np
np.infty=np.inf
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

# Initialize session state for model loading status
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'using_real_activations' not in st.session_state:
    st.session_state.using_real_activations = False
    
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
            install()
            model = load_odeformer_model()
            st.sidebar.success("ODEformer model loaded successfully")
            
            # SAE model loading
            sae_path = st.sidebar.text_input(
                "SAE Model Path", 
                "../sae/sae.encoder.outer.residual1_20250308_230626/checkpoints/sae_best_encoder.outer.residual1.pt"
            )
            
            if Path(sae_path).exists():
                sae_model = load_sae_model(sae_path)
                st.sidebar.success("SAE model loaded successfully")
                st.session_state.models_loaded = True
            else:
                st.sidebar.error(f"SAE model not found at {sae_path}")
        except Exception as e:
            st.sidebar.error(f"Error loading ODEformer model: {str(e)}")
            st.sidebar.code(traceback.format_exc())
    
    except Exception as e:
        st.sidebar.error(f"Error loading models: {str(e)}")

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

def solve_ho(omega, gamma, y0=np.array([1.0, 1.0]), t=np.linspace(0, 10, 100)):
    """Solve harmonic oscillator system with specific parameters"""
    template = "dx/dt = y, dy/dt = -{}*x - {}*y"
    system = template.format(omega**2, gamma)
    solution = integrate_ode(y0, t, system)
    
    if solution is not None:
        return {
            'params': (omega, gamma),
            'equations': system,
            'solution': solution,
            'time_points': t
        }
    return None


# Function to collect activations
def collect_activations(model, solution):
    """Get activations from the model for a given solution"""
    return get_residual_activations(model, solution)

# Function to process activation through SAE
def apply_sae(sae_model, activations):
    """Apply SAE to get latent features"""
    inputs = torch.tensor(activations, dtype=torch.float32)
    _, latent = sae_model(inputs)
    return latent.squeeze(0).detach().numpy()

# Main Streamlit app
st.title("Harmonic Oscillator Feature Explorer")

if st.session_state.models_loaded:
    # Parameter controls
    st.sidebar.header("Harmonic Oscillator Parameters")
    omega = st.sidebar.slider("Natural Frequency (ω)", 0.1, 5.0, 1.0, 0.1)
    gamma = st.sidebar.slider("Damping Coefficient (γ)", 0.1, 5.0, 0.5, 0.1)
    
    # Solve the system
    solution = solve_ho(omega, gamma)
    
    if solution:
        # Plot the solution
        st.subheader("Harmonic Oscillator Solution")
        st.write(f"Equation: {solution['equations']}")
        
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 4))
        ax1.plot(solution['time_points'], solution['solution'][:, 0], label='Position (x)')
        ax1.plot(solution['time_points'], solution['solution'][:, 1], label='Velocity (y)')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True)
        st.pyplot(fig1)
        
        # Collect activations with progress indicator
        with st.spinner("Collecting neural network activations..."):
            activations = collect_activations(model, solution)
        
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
        
        # Apply SAE
        latent_features = apply_sae(sae_model, activations)
        
        # Feature exploration
        st.subheader("Feature Exploration")
        # Create tabs for different visualizations
        tab1, tab2 = st.tabs(["Feature Activations", "Time Point Analysis"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Feature selection
                feature_idx = st.slider("Feature Index", 0, latent_features.shape[1]-1, 727)
                
            with col2:
                # Time point selection
                time_point = st.slider("Time Point", 0, latent_features.shape[0]-1, 41)
            
            # Plot feature activations
            fig2, ax2 = plt.subplots(1, 1, figsize=(10, 4))
            feature_values = latent_features[:, feature_idx]
            ax2.plot(solution['time_points'], feature_values)
            ax2.set_xlabel('Time')
            ax2.set_ylabel(f'Feature {feature_idx} Value')
            ax2.set_title(f'Feature {feature_idx} Activation Over Time')
            ax2.grid(True)
            st.pyplot(fig2)
            
            # Display feature value at selected time point
            st.metric(f"Feature {feature_idx} at Time Point {time_point}", 
                      f"{feature_values[time_point]:.4f}")
        
        with tab2:
            # Add raw activation visualization as an option
            show_raw = st.checkbox("Show raw activations (pre-SAE)", False)
            
            if show_raw:
                # Plot the raw activations for the selected time point
                fig_raw, ax_raw = plt.subplots(1, 1, figsize=(10, 3))
                ax_raw.plot(activations[time_point])
                ax_raw.set_xlabel('Neuron Index')
                ax_raw.set_ylabel('Activation Value')
                ax_raw.set_title(f'Raw Activations at Time Point {time_point}')
                ax_raw.grid(True)
                st.pyplot(fig_raw)
            
            # Plot all features at a specific time point
            st.write(f"All Feature Values at Time Point {time_point}")
            
            fig3, ax3 = plt.subplots(1, 1, figsize=(12, 4))
            
            # Get top activated features
            features_at_time = latent_features[time_point, :]
            sorted_indices = np.argsort(features_at_time)[::-1]  # Sort by activation, descending
            top_n = st.slider("Number of top features to show", 10, 100, 50)
            top_indices = sorted_indices[:top_n]
            
            ax3.bar(top_indices, features_at_time[top_indices])
            ax3.set_xlabel('Feature Index')
            ax3.set_ylabel('Activation Value')
            ax3.set_title(f'Top {top_n} Feature Activations at Time Point {time_point}')
            st.pyplot(fig3)
            
            # Show all features as heatmap
            st.write("All Features Heatmap")
            fig4, ax4 = plt.subplots(1, 1, figsize=(12, 4))
            heatmap = ax4.imshow(latent_features.T, aspect='auto', cmap='viridis')
            ax4.set_xlabel('Time Point')
            ax4.set_ylabel('Feature Index')
            ax4.set_title('All Features Over Time')
            plt.colorbar(heatmap, ax=ax4, label='Activation Value')
            st.pyplot(fig4)
            
    else:
        st.error("Failed to solve the ODE system with the given parameters.")
else:
    st.warning("Please add paths and load models using the sidebar controls.")

# Add some information about the app
st.sidebar.markdown("---")
st.sidebar.info("""
This app allows you to explore how neural network features respond to different harmonic oscillator parameters.
1. Use the sliders to adjust ω and γ
2. Explore how different features respond
3. Analyze activation patterns at specific time points
""")