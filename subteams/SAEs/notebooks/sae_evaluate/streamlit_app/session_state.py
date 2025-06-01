import streamlit as st
import numpy as np

def initialize_session_state():
    """Initialize all session state variables."""
    # Core app state
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
    
    # Activation collection variables
    if 'activation_site' not in st.session_state:
        st.session_state.activation_site = 'RESIDUAL'
    if 'activation_component' not in st.session_state:
        st.session_state.activation_component = 'encoder.transformer.residual1'
    if 'all_collected_activations' not in st.session_state:
        st.session_state.all_collected_activations = None
    if 'sae_paths' not in st.session_state:
        st.session_state.sae_paths = {
            "Residual Layer 1": "../../sae/sae.encoder.outer.residual1_20250308_230626/checkpoints/sae_best_encoder.outer.residual1.pt",
            "Residual Layer 2": "../../sae/sae.encoder.outer.residual2_20250309_122318/checkpoints/sae_best_encoder.outer.residual2.pt",
            "Residual Layer 3": "../../sae/sae.encoder.outer.residual3_20250309_213613/checkpoints/sae_best_encoder.outer.residual3.pt",
            "Custom Path": ""
        }
    
    # System parameters
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

def get_all_system_types():
    """Return list of all available system types."""
    return [
        "Harmonic Oscillator", 
        "Sinusoidal Function", 
        "Linear Function", 
        "Lotka-Volterra", 
        "FitzHugh-Nagumo", 
        "Coupled Linear System",
        "Van der Pol Oscillator", 
        "Duffing Oscillator", 
        "Double Pendulum", 
        "Lorenz System"
    ]