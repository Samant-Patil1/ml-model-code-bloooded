import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import io

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="NASA Exoplanet AI Detector",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 20px;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model_package():
    """Load the complete model package"""
    try:
        # ⚠️ UPDATE THIS FILENAME WITH YOUR ACTUAL MODEL FILE
        package = joblib.load("exoplanet_final_model.joblib")
        return package
    except Exception as e:
        st.error(f" Error loading model: {e}")
        st.error("Please update the filename in the code (line 47)")
        st.stop()

# Load package
with st.spinner(" Loading AI model..."):
    package = load_model_package()

model = package['ensemble_model']
scaler = package['scaler']
feature_names = package['feature_names']
metadata = package['metadata']

# ==================== HEADER ====================
st.markdown('<div class="main-header">🪐 NASA Space Apps Challenge 2025</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Exoplanet Detection System</div>', unsafe_allow_html=True)
st.markdown(f"<div class='sub-header'>Trained on {', '.join(metadata['missions'])} mission data</div>", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://www.nasa.gov/wp-content/uploads/2018/07/nasa-logo.svg", width=200)
    
    st.markdown("---")
    st.subheader(" Ensemble Components")
    for model_name in metadata['ensemble_model_names']:
        st.text(f"• {model_name}")
    
    st.markdown("---")
    st.subheader(" Missions")
    for mission in metadata['missions']:
        st.text(f"• {mission}")
    
    st.markdown("---")
    st.info(f"**Model Version:** {metadata['version']}")
    st.info(f"**Created:** {metadata['created_date']}")

# ==================== MAIN TABS ====================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Single Prediction", 
    " Batch Analysis", 
    " Model Analytics", 
    " Hyperparameter Tuning", 
    "ℹ About"
])

# ==================== TAB 1: SINGLE PREDICTION ====================
with tab1:
    st.header(" Analyze Single Exoplanet Candidate")
    st.markdown("Enter the parameters of an exoplanet candidate to predict if it's a **planet** or **false positive**")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader(" Orbital Properties")
            period = st.number_input("Orbital Period (days)", 0.0, 10000.0, 10.0, 
                                     help="Time for one complete orbit around the star")
            duration = st.number_input("Transit Duration (hours)", 0.0, 48.0, 3.0,
                                       help="Time the planet takes to cross the star")
            depth = st.number_input("Transit Depth (ppm)", 0.0, 100000.0, 1000.0,
                                    help="Brightness dip when planet transits")
        
        with col2:
            st.subheader(" Planet Properties")
            planet_radius = st.number_input("Planet Radius (Earth radii)", 0.1, 100.0, 1.0,
                                           help="Size relative to Earth")
            equilibrium_temp = st.number_input("Equilibrium Temperature (K)", 0, 5000, 288,
                                               help="Expected temperature of the planet")
            insolation = st.number_input("Insolation Flux (Earth units)", 0.0, 10000.0, 1.0,
                                         help="Energy received from star (Earth=1.0)")
        
        with col3:
            st.subheader(" Stellar Properties")
            star_radius = st.number_input("Star Radius (Solar radii)", 0.1, 50.0, 1.0,
                                         help="Size relative to the Sun")
            star_temp = st.number_input("Star Temperature (K)", 2000, 50000, 5778,
                                       help="Surface temperature (Sun=5778K)")
            star_logg = st.number_input("Star log(g)", 0.0, 5.0, 4.4,
                                       help="Surface gravity indicator")
        
        mission = st.selectbox("Mission", metadata['missions'], help="Which telescope detected this candidate")
        
        submit_button = st.form_submit_button(" Analyze Candidate", type="primary")
    
    if submit_button:
        with st.spinner(" Analyzing candidate..."):
            # Create feature dictionary
            features_dict = {}
            
            # Basic features
            feature_map = {
                'period': period,
                'duration': duration,
                'depth': depth,
                'planet_radius': planet_radius,
                'star_radius': star_radius,
                'star_temp': star_temp,
                'star_logg': star_logg,
                'equilibrium_temp': equilibrium_temp,
                'insolation_flux': insolation
            }
            
            for fname, fval in feature_map.items():
                if fname in feature_names:
                    features_dict[fname] = fval
            
            # Engineered features
            if 'transit_period_ratio' in feature_names and period > 0:
                features_dict['transit_period_ratio'] = duration / (period * 24)
            
            if 'radius_ratio' in feature_names and star_radius > 0:
                features_dict['radius_ratio'] = planet_radius / star_radius
            
            if 'period_log' in feature_names and period > 0:
                features_dict['period_log'] = np.log10(period)
            
            if 'insolation_flux_log' in feature_names and insolation > 0:
                features_dict['insolation_flux_log'] = np.log10(insolation)
            
            if 'habitable_zone_dist' in feature_names:
                features_dict['habitable_zone_dist'] = abs(equilibrium_temp - 288) / 288
            
            # Stellar classification
            if 'star_class' in feature_names:
                if star_temp >= 7500: star_class = 5
                elif star_temp >= 6000: star_class = 4
                elif star_temp >= 5200: star_class = 3
                elif star_temp >= 3700: star_class = 2
                else: star_class = 1
                features_dict['star_class'] = star_class
            
            if 'luminosity_class' in feature_names:
                if star_logg < 3.5: lum_class = 3
                elif star_logg < 4.0: lum_class = 2
                else: lum_class = 1
                features_dict['luminosity_class'] = lum_class
            
            # Mission encoding
            for m in metadata['missions']:
                col_name = f'mission_{m}'
                if col_name in feature_names:
                    features_dict[col_name] = 1 if m == mission else 0
            
            # Create feature vector
            feature_vector = [features_dict.get(f, 0) for f in feature_names]
            X_input = np.array(feature_vector).reshape(1, -1)
            
            # Scale and predict
            X_scaled = scaler.transform(X_input)
            prediction = model.predict(X_scaled)[0]
            probabilities = model.predict_proba(X_scaled)[0]
            
            # Display results
            st.markdown("---")
            st.markdown("###  Prediction Results")
            
            result_col1, result_col2, result_col3 = st.columns([2, 2, 3])
            
            with result_col1:
                if prediction == 1:
                    st.success("###  PLANET DETECTED!")
                    confidence = probabilities[1]
                else:
                    st.error("###  FALSE POSITIVE")
                    confidence = probabilities[0]
            
            with result_col2:
                st.metric("Confidence Score", f"{confidence*100:.1f}%",
                         delta=f"{(confidence-0.5)*100:.1f}% from neutral")
                
                if confidence > 0.9:
                    st.info(" Very High Confidence")
                elif confidence > 0.75:
                    st.info(" High Confidence")
                elif confidence > 0.6:
                    st.info(" Moderate Confidence")
                else:
                    st.info(" Low Confidence")
            
            with result_col3:
                # Probability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=probabilities[1] * 100,
                    title={'text': "Planet Probability (%)"},
                    delta={'reference': 50, 'increasing': {'color': "green"}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 25], 'color': "lightgray"},
                            {'range': [25, 50], 'color': "gray"},
                            {'range': [50, 75], 'color': "lightblue"},
                            {'range': [75, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=280, margin=dict(l=20, r=20, t=80, b=20))
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed probabilities
            st.markdown("---")
            st.subheader(" Detailed Probabilities")
            
            prob_col1, prob_col2 = st.columns(2)
            
            with prob_col1:
                st.metric("False Positive Probability", f"{probabilities[0]*100:.2f}%")
            with prob_col2:
                st.metric("Planet Probability", f"{probabilities[1]*100:.2f}%")

# ==================== TAB 2: BATCH ANALYSIS ====================
# ==================== TAB 2: BATCH ANALYSIS ====================
with tab2:
    st.header("📊 Batch Analysis")
    st.markdown("Upload a CSV file with multiple exoplanet candidates for batch predictions")
    
    st.info("💡 **Tip:** Your CSV should contain the same columns as used in single prediction (orbital period, transit duration, depth, etc.)")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file:
        df_upload = pd.read_csv(uploaded_file)
        
        st.subheader("📋 Uploaded Data Preview")
        st.dataframe(df_upload.head(10), use_container_width=True)
        
        st.metric("Total Candidates", len(df_upload))
        
        if st.button("⚡ Analyze All Candidates", type="primary"):
            with st.spinner("Analyzing all candidates... This may take a moment"):
                
                # Initialize results lists
                predictions = []
                planet_probabilities = []
                false_positive_probabilities = []
                confidence_levels = []
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process each row
                for idx, row in df_upload.iterrows():
                    # Update progress
                    progress = (idx + 1) / len(df_upload)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing candidate {idx + 1} of {len(df_upload)}...")
                    
                    try:
                        # Extract features from row
                        features_dict = {}
                        
                        # Map CSV columns to features (handle different column name formats)
                        column_mapping = {
                            'koi_period': 'period',
                            'koi_duration': 'duration',
                            'koi_depth': 'depth',
                            'koi_prad': 'planet_radius',
                            'koi_teq': 'equilibrium_temp',
                            'koi_insol': 'insolation',
                            'koi_steff': 'star_temp',
                            'koi_slogg': 'star_logg',
                            'koi_srad': 'star_radius',
                            'ra': 'ra',
                            'dec': 'dec',
                            'koi_impact': 'impact_param',
                        }
                        
                        # Also handle direct column names
                        direct_columns = ['period', 'duration', 'depth', 'planet_radius', 
                                        'equilibrium_temp', 'insolation', 'star_temp', 
                                        'star_logg', 'star_radius', 'ra', 'dec', 'impact_param']
                        
                        # Extract values with fallbacks
                        period = 0
                        duration = 0
                        star_radius = 0
                        planet_radius = 0
                        insolation = 0
                        equilibrium_temp = 0
                        star_temp = 0
                        star_logg = 0
                        
                        # Try mapped column names first, then direct names
                        for csv_col, feature_name in column_mapping.items():
                            if csv_col in row and pd.notna(row[csv_col]):
                                value = float(row[csv_col])
                                features_dict[feature_name] = value
                                
                                # Store commonly used values
                                if feature_name == 'period': period = value
                                elif feature_name == 'duration': duration = value
                                elif feature_name == 'star_radius': star_radius = value
                                elif feature_name == 'planet_radius': planet_radius = value
                                elif feature_name == 'insolation': insolation = value
                                elif feature_name == 'equilibrium_temp': equilibrium_temp = value
                                elif feature_name == 'star_temp': star_temp = value
                                elif feature_name == 'star_logg': star_logg = value
                        
                        # Try direct column names
                        for col in direct_columns:
                            if col in row and pd.notna(row[col]) and col not in features_dict:
                                value = float(row[col])
                                features_dict[col] = value
                                
                                if col == 'period': period = value
                                elif col == 'duration': duration = value
                                elif col == 'star_radius': star_radius = value
                                elif col == 'planet_radius': planet_radius = value
                                elif col == 'insolation': insolation = value
                                elif col == 'equilibrium_temp': equilibrium_temp = value
                                elif col == 'star_temp': star_temp = value
                                elif col == 'star_logg': star_logg = value
                        
                        # Engineer features
                        if period > 0:
                            features_dict['transit_period_ratio'] = duration / (period * 24)
                            features_dict['period_log'] = np.log10(period)
                        
                        if star_radius > 0 and planet_radius > 0:
                            features_dict['radius_ratio'] = planet_radius / star_radius
                        
                        if insolation > 0:
                            features_dict['insolation_flux_log'] = np.log10(insolation)
                        
                        if equilibrium_temp > 0:
                            features_dict['habitable_zone_dist'] = abs(equilibrium_temp - 288) / 288
                        
                        # Stellar classification
                        if star_temp > 0:
                            if star_temp >= 7500: star_class = 5
                            elif star_temp >= 6000: star_class = 4
                            elif star_temp >= 5200: star_class = 3
                            elif star_temp >= 3700: star_class = 2
                            else: star_class = 1
                            features_dict['star_class'] = star_class
                        
                        if star_logg > 0:
                            if star_logg < 3.5: lum_class = 3
                            elif star_logg < 4.0: lum_class = 2
                            else: lum_class = 1
                            features_dict['luminosity_class'] = lum_class
                        
                        # Mission encoding (assume 'kepler' if mission column not present)
                        mission = row.get('mission', 'kepler')
                        for m in metadata['missions']:
                            features_dict[f'mission_{m}'] = 1 if m == mission else 0
                        
                        # Create feature vector in correct order
                        feature_vector = [features_dict.get(f, 0) for f in feature_names]
                        X_input = np.array(feature_vector).reshape(1, -1)
                        
                        # Scale and predict
                        X_scaled = scaler.transform(X_input)
                        prediction = model.predict(X_scaled)[0]
                        probabilities = model.predict_proba(X_scaled)[0]
                        
                        # Store results
                        predictions.append("PLANET" if prediction == 1 else "FALSE POSITIVE")
                        planet_probabilities.append(probabilities[1])
                        false_positive_probabilities.append(probabilities[0])
                        
                        # Confidence level
                        confidence = max(probabilities)
                        if confidence > 0.9:
                            confidence_levels.append("Very High")
                        elif confidence > 0.75:
                            confidence_levels.append("High")
                        elif confidence > 0.6:
                            confidence_levels.append("Moderate")
                        else:
                            confidence_levels.append("Low")
                    
                    except Exception as e:
                        # Handle errors gracefully
                        predictions.append("ERROR")
                        planet_probabilities.append(0.0)
                        false_positive_probabilities.append(0.0)
                        confidence_levels.append(f"Error: {str(e)[:50]}")
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Add results to dataframe
                df_upload['Prediction'] = predictions
                df_upload['Planet_Probability_%'] = [f"{p*100:.2f}" for p in planet_probabilities]
                df_upload['FalsePositive_Probability_%'] = [f"{fp*100:.2f}" for fp in false_positive_probabilities]
                df_upload['Confidence_Level'] = confidence_levels
                
                # Display summary statistics
                st.success("✅ Analysis Complete!")
                st.balloons()
                
                st.markdown("---")
                st.subheader("📊 Summary Statistics")
                
                summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                
                with summary_col1:
                    planet_count = predictions.count("PLANET")
                    st.metric("Planets Detected", planet_count, 
                             delta=f"{planet_count/len(predictions)*100:.1f}%")
                
                with summary_col2:
                    fp_count = predictions.count("FALSE POSITIVE")
                    st.metric("False Positives", fp_count,
                             delta=f"{fp_count/len(predictions)*100:.1f}%")
                
                with summary_col3:
                    high_conf = confidence_levels.count("Very High") + confidence_levels.count("High")
                    st.metric("High Confidence", high_conf,
                             delta=f"{high_conf/len(predictions)*100:.1f}%")
                
                with summary_col4:
                    avg_planet_prob = np.mean([p for p in planet_probabilities if p > 0])
                    st.metric("Avg Planet Prob", f"{avg_planet_prob*100:.1f}%")
                
                # Display results table
                st.markdown("---")
                st.subheader("📋 Detailed Results")
                
                # Create display dataframe with key columns
                display_cols = ['Prediction', 'Planet_Probability_%', 
                               'FalsePositive_Probability_%', 'Confidence_Level']
                
                # Add original identifying columns if they exist
                id_cols = []
                for col in ['kepid', 'kepoi_name', 'kepler_name', 'koi_id', 'name', 'id']:
                    if col in df_upload.columns:
                        id_cols.append(col)
                
                display_df = df_upload[id_cols + display_cols] if id_cols else df_upload[display_cols]
                
                # Color code predictions
                def highlight_predictions(row):
                    if row['Prediction'] == 'PLANET':
                        return ['background-color: #90EE90'] * len(row)
                    elif row['Prediction'] == 'FALSE POSITIVE':
                        return ['background-color: #FFB6C6'] * len(row)
                    else:
                        return ['background-color: #FFD700'] * len(row)
                
                st.dataframe(
                    display_df.style.apply(highlight_predictions, axis=1),
                    use_container_width=True,
                    height=400
                )
                
                # Visualization
                st.markdown("---")
                st.subheader("📈 Visualization")
                
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    # Pie chart of predictions
                    pred_counts = pd.Series(predictions).value_counts()
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=pred_counts.index,
                        values=pred_counts.values,
                        hole=.3,
                        marker_colors=['#90EE90', '#FFB6C6', '#FFD700']
                    )])
                    fig_pie.update_layout(
                        title="Prediction Distribution",
                        height=400
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with viz_col2:
                    # Histogram of planet probabilities
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(
                        x=[p*100 for p in planet_probabilities],
                        nbinsx=20,
                        marker_color='lightblue',
                        name='Planet Probability'
                    ))
                    fig_hist.update_layout(
                        title="Distribution of Planet Probabilities",
                        xaxis_title="Planet Probability (%)",
                        yaxis_title="Count",
                        height=400
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                # Download results
                st.markdown("---")
                st.subheader("💾 Download Results")
                
                # Convert to CSV
                csv = df_upload.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Full Results as CSV",
                    data=csv,
                    file_name=f"exoplanet_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    type="primary"
                )
                
                # Summary report
                summary_text = f"""
EXOPLANET DETECTION - BATCH ANALYSIS REPORT
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY STATISTICS:
- Total Candidates Analyzed: {len(predictions)}
- Planets Detected: {planet_count} ({planet_count/len(predictions)*100:.1f}%)
- False Positives: {fp_count} ({fp_count/len(predictions)*100:.1f}%)
- High Confidence Predictions: {high_conf} ({high_conf/len(predictions)*100:.1f}%)
- Average Planet Probability: {avg_planet_prob*100:.2f}%

CONFIDENCE BREAKDOWN:
- Very High Confidence: {confidence_levels.count('Very High')}
- High Confidence: {confidence_levels.count('High')}
- Moderate Confidence: {confidence_levels.count('Moderate')}
- Low Confidence: {confidence_levels.count('Low')}
                """
                
                st.download_button(
                    label="📄 Download Summary Report (TXT)",
                    data=summary_text,
                    file_name=f"exoplanet_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

# ==================== TAB 3: MODEL ANALYTICS ====================
with tab3:
    st.header(" Model Performance Analytics")
    
    # Metrics Overview
    st.subheader(" Test Set Performance")
    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
    
    with metric_col1:
        st.metric("Accuracy", f"{metadata['test_accuracy']*100:.2f}%")
    with metric_col2:
        st.metric("Precision", f"{metadata['test_precision']:.3f}")
    with metric_col3:
        st.metric("Recall", f"{metadata['test_recall']:.3f}")
    with metric_col4:
        st.metric("F1 Score", f"{metadata['test_f1_score']:.3f}")
    with metric_col5:
        st.metric("ROC-AUC", f"{metadata['test_roc_auc']:.3f}")
    
    st.markdown("---")
    
    # Dataset Information
    st.subheader(" Dataset Information")
    data_col1, data_col2, data_col3, data_col4 = st.columns(4)
    
    with data_col1:
        st.metric("Total Samples", f"{metadata['total_samples']:,}")
    with data_col2:
        st.metric("Planets", f"{metadata['planets_total']:,}")
    with data_col3:
        st.metric("False Positives", f"{metadata['false_positives_total']:,}")
    with data_col4:
        st.metric("Planet %", f"{metadata['planet_percentage']:.1f}%")
    
    st.markdown("---")
    
    # Model Comparison
    st.subheader(" Individual Model Performance (Validation Set)")
    
    if 'validation_scores' in metadata:
        val_scores_df = pd.DataFrame([
            {"Model": k, "ROC-AUC": v} 
            for k, v in metadata['validation_scores'].items()
        ]).sort_values('ROC-AUC', ascending=False)
        
        fig = px.bar(val_scores_df, x='ROC-AUC', y='Model', orientation='h',
                     title='Model Comparison (Validation ROC-AUC)',
                     color='ROC-AUC', color_continuous_scale='viridis')
        fig.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Cross-Validation
    st.subheader(" Cross-Validation Results")
    cv_col1, cv_col2, cv_col3 = st.columns(3)
    
    with cv_col1:
        st.metric("CV Mean ROC-AUC", f"{metadata['cv_mean_roc_auc']:.4f}")
    with cv_col2:
        st.metric("CV Std Dev", f"±{metadata['cv_std_roc_auc']:.4f}")
    with cv_col3:
        overfitting_status = metadata.get('overfitting_check', 'Unknown')
        st.metric("Overfitting Check", overfitting_status)

# ==================== TAB 4: HYPERPARAMETER TUNING ====================
with tab4:
    st.header(" Hyperparameter Tuning")
    st.markdown("Customize model hyperparameters and train new models")
    
    # ==================== PRESET CONFIGURATIONS ====================
    st.subheader(" Quick Presets")
    
    preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
    
    with preset_col1:
        if st.button(" Best Performance", help="Optimized for maximum accuracy"):
            st.session_state.preset = "best"
    
    with preset_col2:
        if st.button(" Fast Training", help="Quick training, good accuracy"):
            st.session_state.preset = "fast"
    
    with preset_col3:
        if st.button(" Anti-Overfit", help="Maximum generalization"):
            st.session_state.preset = "safe"
    
    with preset_col4:
        if st.button(" Research Grade", help="Publication-quality"):
            st.session_state.preset = "research"
    
    # Initialize session state
    if 'preset' not in st.session_state:
        st.session_state.preset = "best"
    
    # Define presets
    presets = {
        "best": {
            "rf_n_estimators": 300, "rf_max_depth": 15, "rf_min_samples_split": 8,
            "rf_min_samples_leaf": 4, "rf_max_features": "sqrt",
            "gb_n_estimators": 150, "gb_learning_rate": 0.05, "gb_max_depth": 5,
            "gb_min_samples_split": 10, "gb_subsample": 0.8,
            "xgb_n_estimators": 200, "xgb_learning_rate": 0.05, "xgb_max_depth": 6,
            "xgb_min_child_weight": 5, "xgb_subsample": 0.8, "xgb_colsample": 0.8,
            "lgb_n_estimators": 200, "lgb_learning_rate": 0.05, "lgb_max_depth": 7,
            "lgb_num_leaves": 25, "lgb_min_child_samples": 20, "lgb_subsample": 0.8
        },
        "fast": {
            "rf_n_estimators": 100, "rf_max_depth": 10, "rf_min_samples_split": 10,
            "rf_min_samples_leaf": 5, "rf_max_features": "sqrt",
            "gb_n_estimators": 75, "gb_learning_rate": 0.1, "gb_max_depth": 4,
            "gb_min_samples_split": 10, "gb_subsample": 0.8,
            "xgb_n_estimators": 100, "xgb_learning_rate": 0.1, "xgb_max_depth": 5,
            "xgb_min_child_weight": 3, "xgb_subsample": 0.8, "xgb_colsample": 0.8,
            "lgb_n_estimators": 100, "lgb_learning_rate": 0.1, "lgb_max_depth": 6,
            "lgb_num_leaves": 20, "lgb_min_child_samples": 15, "lgb_subsample": 0.8
        },
        "safe": {
            "rf_n_estimators": 200, "rf_max_depth": 10, "rf_min_samples_split": 15,
            "rf_min_samples_leaf": 8, "rf_max_features": "sqrt",
            "gb_n_estimators": 100, "gb_learning_rate": 0.03, "gb_max_depth": 3,
            "gb_min_samples_split": 20, "gb_subsample": 0.7,
            "xgb_n_estimators": 150, "xgb_learning_rate": 0.03, "xgb_max_depth": 4,
            "xgb_min_child_weight": 8, "xgb_subsample": 0.7, "xgb_colsample": 0.7,
            "lgb_n_estimators": 150, "lgb_learning_rate": 0.03, "lgb_max_depth": 5,
            "lgb_num_leaves": 15, "lgb_min_child_samples": 30, "lgb_subsample": 0.7
        },
        "research": {
            "rf_n_estimators": 400, "rf_max_depth": 18, "rf_min_samples_split": 6,
            "rf_min_samples_leaf": 3, "rf_max_features": "sqrt",
            "gb_n_estimators": 200, "gb_learning_rate": 0.03, "gb_max_depth": 6,
            "gb_min_samples_split": 8, "gb_subsample": 0.85,
            "xgb_n_estimators": 250, "xgb_learning_rate": 0.03, "xgb_max_depth": 7,
            "xgb_min_child_weight": 4, "xgb_subsample": 0.85, "xgb_colsample": 0.85,
            "lgb_n_estimators": 250, "lgb_learning_rate": 0.03, "lgb_max_depth": 8,
            "lgb_num_leaves": 30, "lgb_min_child_samples": 15, "lgb_subsample": 0.85
        }
    }
    
    selected_preset = presets[st.session_state.preset]
    st.success(f" Using '{st.session_state.preset.upper()}' preset configuration!")
    
    st.markdown("---")
    
    # Create two columns for different models
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader(" Random Forest")
        rf_n_estimators = st.slider("RF: n_estimators", 50, 500, selected_preset["rf_n_estimators"], 10)
        rf_max_depth = st.slider("RF: max_depth", 5, 30, selected_preset["rf_max_depth"], 1)
        rf_min_samples_split = st.slider("RF: min_samples_split", 2, 20, selected_preset["rf_min_samples_split"], 1)
        rf_min_samples_leaf = st.slider("RF: min_samples_leaf", 1, 10, selected_preset["rf_min_samples_leaf"], 1)
        rf_max_features = st.selectbox("RF: max_features", ['sqrt', 'log2', None], index=0)
    
    with col_right:
        st.subheader(" Gradient Boosting")
        gb_n_estimators = st.slider("GB: n_estimators", 50, 300, selected_preset["gb_n_estimators"], 10)
        gb_learning_rate = st.slider("GB: learning_rate", 0.01, 0.3, selected_preset["gb_learning_rate"], 0.01)
        gb_max_depth = st.slider("GB: max_depth", 3, 10, selected_preset["gb_max_depth"], 1)
        gb_min_samples_split = st.slider("GB: min_samples_split", 2, 20, selected_preset["gb_min_samples_split"], 1)
        gb_subsample = st.slider("GB: subsample", 0.5, 1.0, selected_preset["gb_subsample"], 0.05)
    
    with st.expander(" XGBoost Parameters"):
        col1, col2 = st.columns(2)
        with col1:
            xgb_n_estimators = st.slider("XGB: n_estimators", 50, 300, selected_preset["xgb_n_estimators"], 10, key="xgb_n")
            xgb_learning_rate = st.slider("XGB: learning_rate", 0.01, 0.3, selected_preset["xgb_learning_rate"], 0.01, key="xgb_lr")
            xgb_max_depth = st.slider("XGB: max_depth", 3, 10, selected_preset["xgb_max_depth"], 1, key="xgb_depth")
        with col2:
            xgb_min_child_weight = st.slider("XGB: min_child_weight", 1, 10, selected_preset["xgb_min_child_weight"], 1)
            xgb_subsample = st.slider("XGB: subsample", 0.5, 1.0, selected_preset["xgb_subsample"], 0.05, key="xgb_sub")
            xgb_colsample = st.slider("XGB: colsample_bytree", 0.5, 1.0, selected_preset["xgb_colsample"], 0.05)
    
    with st.expander(" LightGBM Parameters"):
        col1, col2 = st.columns(2)
        with col1:
            lgb_n_estimators = st.slider("LGB: n_estimators", 50, 300, selected_preset["lgb_n_estimators"], 10, key="lgb_n")
            lgb_learning_rate = st.slider("LGB: learning_rate", 0.01, 0.3, selected_preset["lgb_learning_rate"], 0.01, key="lgb_lr")
            lgb_max_depth = st.slider("LGB: max_depth", 3, 15, selected_preset["lgb_max_depth"], 1, key="lgb_depth")
        with col2:
            lgb_num_leaves = st.slider("LGB: num_leaves", 10, 100, selected_preset["lgb_num_leaves"], 5)
            lgb_min_child_samples = st.slider("LGB: min_child_samples", 5, 50, selected_preset["lgb_min_child_samples"], 5)
            lgb_subsample = st.slider("LGB: subsample", 0.5, 1.0, selected_preset["lgb_subsample"], 0.05, key="lgb_sub")
    
    st.markdown("---")
    
    # Generate code button
    st.subheader(" Generated Training Code")
    
    if st.button(" Generate Retraining Code"):
        generated_code = f"""# Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

# Random Forest Parameters
rf_params = {{
    'n_estimators': {rf_n_estimators},
    'max_depth': {rf_max_depth},
    'min_samples_split': {rf_min_samples_split},
    'min_samples_leaf': {rf_min_samples_leaf},
    'max_features': {repr(rf_max_features)},
    'random_state': 42, 'n_jobs': -1, 'class_weight': 'balanced'
}}

# Gradient Boosting Parameters
gb_params = {{
    'n_estimators': {gb_n_estimators},
    'learning_rate': {gb_learning_rate},
    'max_depth': {gb_max_depth},
    'min_samples_split': {gb_min_samples_split},
    'subsample': {gb_subsample},
    'random_state': 42
}}

# XGBoost Parameters
xgb_params = {{
    'n_estimators': {xgb_n_estimators},
    'learning_rate': {xgb_learning_rate},
    'max_depth': {xgb_max_depth},
    'min_child_weight': {xgb_min_child_weight},
    'subsample': {xgb_subsample},
    'colsample_bytree': {xgb_colsample},
    'random_state': 42, 'n_jobs': -1
}}

# LightGBM Parameters
lgb_params = {{
    'n_estimators': {lgb_n_estimators},
    'learning_rate': {lgb_learning_rate},
    'max_depth': {lgb_max_depth},
    'num_leaves': {lgb_num_leaves},
    'min_child_samples': {lgb_min_child_samples},
    'subsample': {lgb_subsample},
    'random_state': 42, 'n_jobs': -1, 'verbose': -1
}}

# Train models
trained_models, final_model = train_all_models_anti_overfit(
    X_train_scaled, y_train, X_val_scaled, y_val
)
"""
        st.code(generated_code, language="python")
        st.success(" Code generated! Copy and paste into Jupyter notebook.")
    
    st.markdown("---")
    
    # ==================== TRAIN MODEL IN STREAMLIT ====================
    st.subheader(" Train Model with Custom Parameters")
    
    train_col1, train_col2 = st.columns([3, 1])
    
    with train_col1:
        st.info("""
        **How it works:**
        1. Adjust hyperparameters above
        2. Click "Train New Model"
        3. Wait 5-15 minutes for training
        4. Download trained model
        5. Replace old model and restart app
        """)
    
    with train_col2:
        train_button = st.button(" Train New Model", type="primary", use_container_width=True)
    
    if train_button:
        st.markdown("---")
        st.header(" Training in Progress...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Load Data
            status_text.text("Step 1/5: Loading datasets...")
            progress_bar.progress(10)
            
            @st.cache_data
            def load_training_data():
                import requests
                from io import StringIO
                datasets = {}
                try:
                    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+koi&format=csv"
                    response = requests.get(url, timeout=30)
                    if response.status_code == 200:
                        datasets['kepler'] = pd.read_csv(StringIO(response.text))
                except: pass
                try:
                    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+toi&format=csv"
                    response = requests.get(url, timeout=30)
                    if response.status_code == 200:
                        datasets['tess'] = pd.read_csv(StringIO(response.text))
                except: pass
                try:
                    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+k2pandc&format=csv"
                    response = requests.get(url, timeout=30)
                    if response.status_code == 200:
                        datasets['k2'] = pd.read_csv(StringIO(response.text))
                except: pass
                return datasets
            
            datasets = load_training_data()
            if len(datasets) == 0:
                st.error(" Unable to load datasets")
                st.stop()
            
            st.success(f" Loaded {len(datasets)} dataset(s)")
            progress_bar.progress(20)
            
            # Step 2: Preprocess
            status_text.text("Step 2/5: Preprocessing...")
            
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import RobustScaler
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
            
            def quick_preprocess(datasets):
                dfs = []
                for mission, df in datasets.items():
                    df_copy = df.copy()
                    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
                    target_cols = ['koi_disposition', 'tfopwg_disp', 'disposition']
                    target_col = None
                    for tc in target_cols:
                        if tc in df_copy.columns:
                            target_col = tc
                            break
                    if target_col is None:
                        continue
                    
                    # Create binary target
                    if mission == 'kepler':
                        df_copy['target'] = df_copy[target_col].apply(
                            lambda x: 1 if str(x).upper() in ['CONFIRMED', 'CANDIDATE'] else 0
                        )
                    elif mission == 'tess':
                        df_copy['target'] = df_copy[target_col].apply(
                            lambda x: 1 if str(x).upper() in ['PC', 'CP', 'KP'] else 0
                        )
                    else:
                        df_copy['target'] = df_copy[target_col].apply(
                            lambda x: 1 if str(x).upper() in ['CONFIRMED', 'CANDIDATE'] else 0
                        )
                    
                    keep_cols = [col for col in numeric_cols if col != target_col] + ['target']
                    df_subset = df_copy[keep_cols].copy()
                    dfs.append(df_subset)
                
                # Combine all datasets
                combined = pd.concat(dfs, ignore_index=True)
                
                # CRITICAL: Remove columns with too many missing values FIRST
                missing_pct = combined.isnull().sum() / len(combined)
                cols_to_keep = missing_pct[missing_pct < 0.7].index.tolist()  # Keep columns with <70% missing
                combined = combined[cols_to_keep]
                
                # Fill remaining NaN values with median
                for col in combined.columns:
                    if col != 'target':
                        if combined[col].isnull().any():
                            median_val = combined[col].median()
                            # If median is also NaN (all values are NaN), use 0
                            if pd.isna(median_val):
                                combined[col].fillna(0, inplace=True)
                            else:
                                combined[col].fillna(median_val, inplace=True)
                
                # Replace infinite values
                combined = combined.replace([np.inf, -np.inf], 0)
                
                # Remove rows with ANY remaining missing values in features
                combined = combined.dropna(subset=[col for col in combined.columns if col != 'target'])
                
                # Final safety check: ensure NO NaN values remain
                assert combined.isnull().sum().sum() == 0, "NaN values still present after preprocessing!"
                
                return combined
            
            processed_data = quick_preprocess(datasets)
            X = processed_data.drop('target', axis=1)
            y = processed_data['target']
            
            st.success(f" Preprocessed {len(X)} samples")
            progress_bar.progress(35)
            
            # Step 3: Split and Scale
            status_text.text("Step 3/5: Splitting and scaling...")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            scaler_new = RobustScaler()
            X_train_scaled = scaler_new.fit_transform(X_train)
            X_test_scaled = scaler_new.transform(X_test)
            
            progress_bar.progress(45)
            
            # Step 4: Train Models
            status_text.text("Step 4/5: Training models...")
            
            models_trained = {}
            
            st.write(" Training Random Forest...")
            rf_new = RandomForestClassifier(
                n_estimators=rf_n_estimators, max_depth=rf_max_depth,
                min_samples_split=rf_min_samples_split, min_samples_leaf=rf_min_samples_leaf,
                max_features=rf_max_features, class_weight='balanced',
                random_state=42, n_jobs=-1
            )
            rf_new.fit(X_train_scaled, y_train)
            models_trained['RandomForest'] = rf_new
            progress_bar.progress(55)
            
            st.write(" Training Gradient Boosting...")
            gb_new = GradientBoostingClassifier(
                n_estimators=gb_n_estimators, learning_rate=gb_learning_rate,
                max_depth=gb_max_depth, min_samples_split=gb_min_samples_split,
                subsample=gb_subsample, random_state=42
            )
            gb_new.fit(X_train_scaled, y_train)
            models_trained['GradientBoosting'] = gb_new
            progress_bar.progress(65)
            
            try:
                import xgboost as xgb
                st.write(" Training XGBoost...")
                xgb_new = xgb.XGBClassifier(
                    n_estimators=xgb_n_estimators, learning_rate=xgb_learning_rate,
                    max_depth=xgb_max_depth, min_child_weight=xgb_min_child_weight,
                    subsample=xgb_subsample, colsample_bytree=xgb_colsample,
                    random_state=42, n_jobs=-1
                )
                xgb_new.fit(X_train_scaled, y_train)
                models_trained['XGBoost'] = xgb_new
            except:
                st.warning(" XGBoost not available")
            progress_bar.progress(75)
            
            try:
                import lightgbm as lgb
                st.write(" Training LightGBM...")
                lgb_new = lgb.LGBMClassifier(
                    n_estimators=lgb_n_estimators, learning_rate=lgb_learning_rate,
                    max_depth=lgb_max_depth, num_leaves=lgb_num_leaves,
                    min_child_samples=lgb_min_child_samples, subsample=lgb_subsample,
                    random_state=42, n_jobs=-1, verbose=-1
                )
                lgb_new.fit(X_train_scaled, y_train)
                models_trained['LightGBM'] = lgb_new
            except:
                st.warning(" LightGBM not available")
            progress_bar.progress(85)
            
            st.write(" Creating Ensemble...")
            estimators = [(name, model) for name, model in models_trained.items()]
            ensemble_new = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
            ensemble_new.fit(X_train_scaled, y_train)
            progress_bar.progress(90)
            
            # Step 5: Evaluate
            status_text.text("Step 5/5: Evaluating...")
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            y_pred = ensemble_new.predict(X_test_scaled)
            y_pred_proba = ensemble_new.predict_proba(X_test_scaled)[:, 1]
            
            new_metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            progress_bar.progress(100)
            status_text.text(" Training complete!")
            
            st.success(" Model training complete!")
            
            st.markdown("---")
            st.subheader(" New Model Performance")
            
            metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
            with metric_col1:
                st.metric("Accuracy", f"{new_metrics['accuracy']:.3f}")
            with metric_col2:
                st.metric("Precision", f"{new_metrics['precision']:.3f}")
            with metric_col3:
                st.metric("Recall", f"{new_metrics['recall']:.3f}")
            with metric_col4:
                st.metric("F1 Score", f"{new_metrics['f1_score']:.3f}")
            with metric_col5:
                st.metric("ROC-AUC", f"{new_metrics['roc_auc']:.3f}")
            
            # Save model
            st.markdown("---")
            st.subheader(" Download New Model")
            
            new_model_package = {
                'ensemble_model': ensemble_new,
                'individual_models': models_trained,
                'scaler': scaler_new,
                'feature_names': X.columns.tolist(),
                'metadata': {
                    'version': '2.0',
                    'created_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                    'created_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'missions': list(datasets.keys()),
                    'total_samples': len(X),
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'n_features': len(X.columns),
                    'test_accuracy': float(new_metrics['accuracy']),
                    'test_precision': float(new_metrics['precision']),
                    'test_recall': float(new_metrics['recall']),
                    'test_f1_score': float(new_metrics['f1_score']),
                    'test_roc_auc': float(new_metrics['roc_auc']),
                    'n_models_in_ensemble': len(models_trained),
                    'ensemble_model_names': list(models_trained.keys()),
                    'planets_total': int(y.sum()),
                    'false_positives_total': int((y==0).sum()),
                    'planet_percentage': float(y.mean() * 100),
                    'cv_mean_roc_auc': 0.0,
                    'cv_std_roc_auc': 0.0,
                    'overfitting_check': 'Not tested'
                }
            }
            
            buffer = io.BytesIO()
            joblib.dump(new_model_package, buffer)
            buffer.seek(0)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"exoplanet_final_model.joblib"
            
            st.download_button(
                label="⬇ Download New Model",
                data=buffer,
                file_name=filename,
                mime="application/octet-stream",
                type="primary"
            )
            
            st.success(f" Model ready! Update line 47 with: `{filename}`")
            
        except Exception as e:
            st.error(f" Error: {str(e)}")

# ==================== TAB 5: ABOUT ====================
with tab5:
    st.header("ℹ About This System")
    
    st.markdown("""
    ###  Project Overview
    AI-powered exoplanet detection using NASA telescope data.
    
    ###  Data Sources
    - **Kepler Mission**: Stellar transit observations
    - **TESS Mission**: Transiting Exoplanet Survey Satellite
    - **K2 Mission**: Extended Kepler observations
    
    ###  ML Approach
    Multi-model ensemble with advanced feature engineering
    
    ###  NASA Space Apps Challenge 2025
    Built for "A World Away: Hunting for Exoplanets with AI"
    
    ###  Resources
    - [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
    - [Space Apps Challenge](https://www.spaceappschallenge.org/)
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>NASA Space Apps Challenge 2025</strong></p>
    <p>Built with ❤️ using Streamlit & Machine Learning</p>
    <p>🌟 Detecting exoplanets • One transit at a time 🪐</p>
</div>
""", unsafe_allow_html=True)
