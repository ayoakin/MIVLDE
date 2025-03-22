import streamlit as st
import torch
import traceback
import importlib
from pathlib import Path
from activation_collector import install, collect_activations_during_fit

@st.cache_resource
def load_odeformer_model():
    """Load the ODEformer model with caching."""
    try:
        from odeformer.model import SymbolicTransformerRegressor
        model = SymbolicTransformerRegressor(from_pretrained=True)
        model_args = {'beam_size': 20, 'beam_temperature': 0.1}
        model.set_model_args(model_args)
        return model
    except Exception as e:
        st.error(f"Error loading ODEformer model: {str(e)}")
        st.code(traceback.format_exc())
        return None
            
@st.cache_resource
def load_sae_model(model_path):
    """Load a sparse autoencoder model with caching."""
    try:
        SparseAutoencoder = st.session_state.sae_module.SparseAutoencoder
        sae_model = SparseAutoencoder(input_dim=256, latent_dim=1280)
        sae_model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        sae_model.eval()
        return sae_model
    except Exception as e:
        st.error(f"Error loading SAE model: {str(e)}")
        st.code(traceback.format_exc())
        return None

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

def apply_sae(sae_model, activations):
    """Apply SAE to get latent features"""
    inputs = torch.tensor(activations, dtype=torch.float32)
    _, latent = sae_model(inputs)
    return latent.squeeze(0).detach().numpy()

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

def load_models():
    """Load all required models."""
    try:
        # Import required modules
        from odeformer.model import SymbolicTransformerRegressor
        st.session_state.sae_module = importlib.import_module("sae")
        
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
                st.session_state.all_collected_activations = None
                st.sidebar.info("SAE model changed - cache cleared")
        else:
            st.sidebar.error(f"SAE model not found at {sae_path}")
            
        # Store models in session state for later access
        st.session_state.model = model
        st.session_state.sae_model = sae_model
        
        return True
        
    except Exception as e:
        st.sidebar.error(f"Error loading models: {str(e)}")
        st.sidebar.code(traceback.format_exc())
        return False