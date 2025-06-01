import streamlit as st
import numpy as np
np.infty=np.inf
np.random.seed(42)
import matplotlib.pyplot as plt
import pandas as pd
import traceback
import sys
from pathlib import Path
import importlib

# Import our modules
from session_state import initialize_session_state, get_all_system_types
from ui_components import (
    setup_sidebar, setup_activation_ui, setup_system_selection_ui,
    display_time_point_controls, display_feature_controls, display_state_metrics
)
from model_utils import load_models, get_activations, apply_sae, get_learned_equations
from ode_solvers import (
    solve_ho, solve_sinusoidal, solve_linear, solve_lotka_volterra, 
    solve_fitzhugh_nagumo, solve_coupled_linear, solve_van_der_pol, 
    solve_duffing, solve_double_pendulum, solve_lorenz, 
    compute_poincare_section
)
from visualization import (
    plot_solution, plot_feature_activation, plot_phase_portrait_2d,
    plot_phase_portrait_3d, plot_double_pendulum_phase, plot_poincare_section,
    plot_top_features, plot_features_heatmap, compute_dimensionality_reduction
)

# Initialize the app
st.title("Dynamic Systems Feature Explorer")

# Initialize session state variables
initialize_session_state()

# Setup the sidebar and paths
paths_added = setup_sidebar()

# Only proceed if paths are added
if not paths_added:
    st.warning("Please add paths using the sidebar controls before proceeding.")
    st.stop()

# Load models button
if st.sidebar.button("Load Models") or st.session_state.models_loaded:
    models_loaded = load_models()
    if not models_loaded:
        st.error("Failed to load models. Check console for details.")
        st.stop()

    # Setup activation UI in the sidebar
    setup_activation_ui()

    # System selection UI
    system_type, times, system_params = setup_system_selection_ui()

    # Define a function to solve the selected system
    def solve_selected_system():
        if system_type == "Harmonic Oscillator":
            omega, gamma, x0, v0 = system_params
            return solve_ho(omega, gamma, y0=np.array([x0, v0]), t=times)
        
        elif system_type == "Sinusoidal Function":
            amplitude, frequency, phase, use_cos = system_params
            return solve_sinusoidal(amplitude, frequency, phase, use_cos, t=times)
        
        elif system_type == "Linear Function":
            slope, intercept = system_params
            return solve_linear(slope, intercept, t=times)
        
        elif system_type == "Lotka-Volterra":
            alpha, beta, delta, gamma, prey0, predator0 = system_params
            return solve_lotka_volterra(alpha, beta, delta, gamma, 
                                       y0=np.array([prey0, predator0]), t=times)
        
        elif system_type == "FitzHugh-Nagumo":
            a, b, tau, I, v0, w0 = system_params
            return solve_fitzhugh_nagumo(a, b, tau, I, 
                                        y0=np.array([v0, w0]), t=times)
        
        elif system_type == "Coupled Linear System":
            alpha, beta, x0, y0 = system_params
            return solve_coupled_linear(alpha, beta, 
                                       y0=np.array([x0, y0]), t=times)
        
        elif system_type == "Van der Pol Oscillator":
            mu, x0, y0 = system_params
            return solve_van_der_pol(mu, y0=np.array([x0, y0]), t=times)
        
        elif system_type == "Duffing Oscillator":
            alpha, beta, delta, gamma, omega, x0, y0 = system_params
            return solve_duffing(alpha, beta, delta, gamma, omega,
                                y0=np.array([x0, y0]), t=times)
        
        elif system_type == "Double Pendulum":
            g, m1, m2, l1, l2, theta1, omega1, theta2, omega2 = system_params
            return solve_double_pendulum(g, m1, m2, l1, l2,
                                        y0=np.array([theta1, omega1, theta2, omega2]), t=times)
        
        elif system_type == "Lorenz System":
            sigma, rho, beta, x0, y0, z0 = system_params
            return solve_lorenz(sigma, rho, beta,
                               y0=np.array([x0, y0, z0]), t=times)
        
        return None

    # Check if parameters have changed
    # Use the length of time points as part of the params hash to detect time changes
    current_params = (system_type, len(times), tuple(times[:5]), system_params)  # Use first 5 points as a sample
    params_changed = (st.session_state.current_params != current_params)

    # Solve the system if parameters changed or no solution exists
    if params_changed or st.session_state.current_solution is None:
        solution = solve_selected_system()
        st.session_state.current_params = current_params
        st.session_state.current_solution = solution
        
        # Clear existing activations and features since parameters changed
        st.session_state.current_activations = None
        st.session_state.current_latent_features = None
        st.session_state.all_collected_activations = None  # Also clear all cached activations
        st.session_state.learned_equation = None
        
        # Log information about what changed
        if st.session_state.current_solution is not None:
            st.info(f"Parameters changed. Recalculating solution and activations. Time points: {len(times)}.")
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
        
        # Feature activation plot - Check for length mismatch
        feature_values = latent_features[:, feature_idx]

        # Make sure feature_values has the same length as time_points
        if len(solution['time_points']) != len(feature_values):
            st.error(f"""
            Length mismatch detected! Time points ({len(solution['time_points'])}) and feature values ({len(feature_values)}) have different lengths.
            This likely happened because the time points were changed but activations weren't recollected.
            Clearing cached activations and features...
            """)
            
            # Force clearing the cached activations and features
            st.session_state.current_activations = None
            st.session_state.current_latent_features = None
            st.session_state.all_collected_activations = None
            
            # Stop execution and ask user to re-run
            st.stop()
        
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
        
        # Extract variables based on system type
        if system_type == "Harmonic Oscillator":
            var1_name = "Position (x)"
            var2_name = "Velocity (dx/dt)"
            var1 = solution['solution'][:, 0]
            var2 = solution['solution'][:, 1]
            # Calculate acceleration for harmonic oscillator
            omega, gamma = solution['params'][:2]
            var3_name = "Acceleration (d²x/dt²)"
            var3 = -omega**2 * var1 - gamma * var2
            
        elif system_type == "Sinusoidal Function":
            var1_name = "Value (y)"
            var2_name = "Derivative (dy/dt)"
            var1 = solution['solution'][:, 0]
            var2 = solution['solution'][:, 1]
            var3_name = "Second Derivative (d²y/dt²)"
            # Calculate second derivative
            amplitude, frequency, phase, use_cos = solution['params']
            t = solution['time_points']
            var3 = -amplitude * frequency**2 * (np.sin(frequency * t + phase) if not use_cos else np.cos(frequency * t + phase))
            
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
            var4_name = None
            var4 = None
        
        # Plot the solution
        st.subheader(f"{system_type} Solution and Feature Activations")
        st.write(f"Equation: {solution['equations']}")
        
        # Create solution plot - handle systems with and without var4
        if system_type == "Double Pendulum":
            # Only Double Pendulum uses var4
            fig1 = plot_solution(
                solution, time_point, var1, var2, var3, var4, 
                var1_name, var2_name, var3_name, var4_name,
                top_activation_times, top_activation_values
            )
        else:
            # All other systems only use up to var3
            fig1 = plot_solution(
                solution, time_point, var1, var2, var3, None, 
                var1_name, var2_name, var3_name, None,
                top_activation_times, top_activation_values
            )
        st.pyplot(fig1)
        
        # Create feature activation plot
        fig2 = plot_feature_activation(
            solution['time_points'], feature_values, time_point, top_activation_times
        )
        st.pyplot(fig2)
        
        # Phase portrait based on system type
        st.subheader("Phase Portrait")
        
        if system_type in ["Harmonic Oscillator", "Van der Pol Oscillator", "Duffing Oscillator", 
                          "Coupled Linear System", "Lotka-Volterra", "FitzHugh-Nagumo"]:
            # Simple 2D phase portrait
            fig_phase = plot_phase_portrait_2d(
                var1, var2, var1_name, var2_name, feature_values, time_point, system_type
            )
            st.pyplot(fig_phase)
            
        elif system_type == "Double Pendulum":
            # Double pendulum phase portrait
            cart_sol = solution.get('cartesian_solution', None)
            fig_phase = plot_double_pendulum_phase(
                var1, var2, var3, var4, cart_sol, time_point, feature_values
            )
            st.pyplot(fig_phase)
            
        elif system_type == "Lorenz System":
            # 3D phase portrait
            fig_phase = plot_phase_portrait_3d(
                var1, var2, var3, var1_name, var2_name, var3_name, 
                feature_values, time_point, system_type
            )
            st.pyplot(fig_phase)
            
            # Poincaré section
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
            
            # Plot Poincaré section if points found
            fig_poincare, num_points = plot_poincare_section(
                poincare_points, section_axis, section_value, section_direction
            )
            
            if fig_poincare:
                st.pyplot(fig_poincare)
                st.write(f"Found {num_points} intersection points. Direction: {'Positive' if section_direction > 0 else 'Negative' if section_direction < 0 else 'Both'}")
            else:
                st.warning("No Poincaré section points found. Try adjusting the section parameters.")
        
        # Display learned equations alongside original equations
        st.subheader("Model Prediction")

        # Create columns for side-by-side comparison
        eq_col1, eq_col2 = st.columns(2)

        with eq_col1:
            st.markdown("#### Original Equation")
            st.markdown(f"```\n{solution['equations']}\n```")
            
            # Display system parameters
            st.markdown("**Parameters:**")
            if system_type == "Harmonic Oscillator":
                omega, gamma, x0, v0 = system_params
                st.markdown(f"- Natural Frequency (ω): {omega}")
                st.markdown(f"- Damping Coefficient (γ): {gamma}")
                st.markdown(f"- Initial Position (x₀): {x0}")
                st.markdown(f"- Initial Velocity (v₀): {v0}")
            elif system_type == "Sinusoidal Function":
                amplitude, frequency, phase, use_cos = system_params
                st.markdown(f"- Amplitude (A): {amplitude}")
                st.markdown(f"- Frequency (ω): {frequency}")
                st.markdown(f"- Phase (φ): {phase}")
                st.markdown(f"- Function: {'Cosine' if use_cos else 'Sine'}")
            elif system_type == "Linear Function":
                slope, intercept = system_params
                st.markdown(f"- Slope (m): {slope}")
                st.markdown(f"- Intercept (b): {intercept}")
            elif system_type == "Lotka-Volterra":
                alpha, beta, delta, gamma, prey0, predator0 = system_params
                st.markdown(f"- Prey Growth Rate (α): {alpha}")
                st.markdown(f"- Predation Rate (β): {beta}")
                st.markdown(f"- Predator Death Rate (δ): {delta}")
                st.markdown(f"- Predator Growth from Prey (γ): {gamma}")
                st.markdown(f"- Initial Prey: {prey0}")
                st.markdown(f"- Initial Predator: {predator0}")
            elif system_type == "FitzHugh-Nagumo":
                a, b, tau, I, v0, w0 = system_params
                st.markdown(f"- Parameter a: {a}")
                st.markdown(f"- Parameter b: {b}")
                st.markdown(f"- Time Scale (τ): {tau}")
                st.markdown(f"- Input Current (I): {I}")
                st.markdown(f"- Initial v: {v0}")
                st.markdown(f"- Initial w: {w0}")
            elif system_type == "Coupled Linear System":
                alpha, beta, x0, y0 = system_params
                st.markdown(f"- Alpha (α): {alpha}")
                st.markdown(f"- Beta (β): {beta}")
                st.markdown(f"- Initial x: {x0}")
                st.markdown(f"- Initial y: {y0}")
            elif system_type == "Van der Pol Oscillator":
                mu, x0, y0 = system_params
                st.markdown(f"- Nonlinearity Parameter (μ): {mu}")
                st.markdown(f"- Initial Position (x₀): {x0}")
                st.markdown(f"- Initial Velocity (y₀): {y0}")
            elif system_type == "Duffing Oscillator":
                alpha, beta, delta, gamma, omega, x0, y0 = system_params
                st.markdown(f"- Linear Stiffness (α): {alpha}")
                st.markdown(f"- Nonlinear Stiffness (β): {beta}")
                st.markdown(f"- Damping (δ): {delta}")
                st.markdown(f"- Forcing Amplitude (γ): {gamma}")
                st.markdown(f"- Forcing Frequency (ω): {omega}")
                st.markdown(f"- Initial Position (x₀): {x0}")
                st.markdown(f"- Initial Velocity (y₀): {y0}")
            elif system_type == "Double Pendulum":
                g, m1, m2, l1, l2, theta1, omega1, theta2, omega2 = system_params
                st.markdown(f"- Gravity (g): {g}")
                st.markdown(f"- Mass 1 (m₁): {m1}")
                st.markdown(f"- Mass 2 (m₂): {m2}")
                st.markdown(f"- Length 1 (l₁): {l1}")
                st.markdown(f"- Length 2 (l₂): {l2}")
                st.markdown(f"- Initial Angle 1 (θ₁): {theta1:.2f} rad")
                st.markdown(f"- Initial Angular Velocity 1 (ω₁): {omega1}")
                st.markdown(f"- Initial Angle 2 (θ₂): {theta2:.2f} rad")
                st.markdown(f"- Initial Angular Velocity 2 (ω₂): {omega2}")
            elif system_type == "Lorenz System":
                sigma, rho, beta, x0, y0, z0 = system_params
                st.markdown(f"- Sigma (σ): {sigma}")
                st.markdown(f"- Rho (ρ): {rho}")
                st.markdown(f"- Beta (β): {beta}")
                st.markdown(f"- Initial x₀: {x0}")
                st.markdown(f"- Initial y₀: {y0}")
                st.markdown(f"- Initial z₀: {z0}")

        with eq_col2:
            st.markdown("#### Learned Equation")
            
            # Add a button to run the prediction
            if st.button("Run ODEformer Prediction"):
                with st.spinner("Running model to learn equation..."):
                    # Get learned equations from model
                    learned_eq_result = get_learned_equations(st.session_state.model, solution)
                    
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
        time_point = display_time_point_controls(time_point, latent_features.shape[0]-1)
        
        # Save time_point to session state
        st.session_state.time_point = time_point
        
        # Display system state metrics
        display_state_metrics(solution, time_point, system_type, feature_values, feature_idx)
        
        # FEATURE EXPLORATION SECTION
        feature_idx = display_feature_controls(feature_idx, latent_features.shape[1]-1)
        
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
        
        # Plot top features
        top_n = st.slider("Number of top features to show", 10, 100, 50, step=5)
        fig3, highlight_info = plot_top_features(latent_features, time_point, feature_idx, top_n)
        st.pyplot(fig3)
        st.caption(highlight_info)
        
        # Add all features heatmap and table
        st.markdown("### All Features Heatmap")
        st.write("This heatmap shows all feature activations across all time points.")
        
        # Use same highlighting logic as before
        highlight_top_n = st.checkbox("Highlight top activations on heatmap", True)
        if highlight_top_n:
            fig4, top_activations_data, threshold = plot_features_heatmap(
                latent_features, time_point, feature_idx, highlight_top_n, top_n_global
            )
            
            st.pyplot(fig4)
            st.caption(f"Highlighting all activations >= {threshold:.4f}")
            
            # Display the table of top activations
            st.markdown("#### Top Activations Values")
            st.write(f"Showing all {len(top_activations_data)} activations above threshold {threshold:.4f}")
            
            # Use a dataframe for better formatting and sortability
            df = pd.DataFrame(top_activations_data)
            
            # Allow sorting by clicking on column headers
            st.dataframe(df, use_container_width=True, height=min(400, len(df) * 35 + 38))

            df['Activation'] = df['Activation'].apply(lambda x: float(x))
            grouped_df = (
                df.groupby('Feature')
                .agg(Count=('Feature', 'size'), Mean_Activation=('Activation', 'mean'))
                .reset_index()
                .sort_values(by='Mean_Activation', ascending=False)
            )
            
            st.markdown("#### Summary by Feature")
            st.dataframe(grouped_df, use_container_width=True, height=min(400, len(grouped_df) * 35 + 38))
            
        else:
            fig4, _, _ = plot_features_heatmap(
                latent_features, time_point, feature_idx, highlight_top_n=False
            )
            st.pyplot(fig4)
        
        # Add t-SNE and UMAP visualization section
        st.markdown("### Dimensionality Reduction Visualization")
        st.write("Visualize the latent feature vectors using dimensionality reduction techniques.")
        
        # Check for libraries
        sklearn_available = importlib.util.find_spec("sklearn") is not None
        umap_available = importlib.util.find_spec("umap") is not None
        
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
**Current SAE Layer**: {list(st.session_state.sae_paths.keys())[list(st.session_state.sae_paths.values()).index(st.session_state.current_sae_path)] if st.session_state.current_sae_path in st.session_state.sae_paths.values() else 'Custom' if st.session_state.current_sae_path else 'None'}

1. Use the sidebar to select a system and adjust parameters
2. Choose SAE model for different layers
3. Choose activation sites and components to analyze
4. Explore feature activations at specific time points
5. Analyze patterns in the neural network representations
""")