import streamlit as st
import joblib
import pandas as pd
import numpy as np
import io

# Page config
st.set_page_config(
    page_title="Exoplanet Detection AI",
    page_icon="🌌",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .planet-result {
        font-size: 2rem;
        font-weight: bold;
        color: #2ecc71;
        text-align: center;
        padding: 1rem;
    }
    .not-planet-result {
        font-size: 2rem;
        font-weight: bold;
        color: #e74c3c;
        text-align: center;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model artifacts
@st.cache_resource
def load_models():
    """Load all model components"""
    model = joblib.load("model_export/exoplanet_ensemble_model.joblib")
    scaler = joblib.load("model_export/feature_scaler.joblib")
    imputer = joblib.load("model_export/feature_imputer.joblib")
    selector = joblib.load("model_export/variance_selector.joblib")
    feature_info = joblib.load("model_export/feature_info.joblib")
    metrics = joblib.load("model_export/model_metrics.joblib")
    return model, scaler, imputer, selector, feature_info, metrics

try:
    model, scaler, imputer, selector, feature_info, metrics = load_models()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Title
st.markdown('<h1 class="main-header">🌌 Exoplanet Detection AI</h1>', unsafe_allow_html=True)
st.markdown("### NASA Space Apps Challenge 2025 - Hunting Exoplanets with Machine Learning")
st.markdown("---")

# Sidebar - Model Info
with st.sidebar:
    st.header("📊 Model Performance")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Test AUC", f"{metrics['test_auc']:.3f}")
        st.metric("Test F1", f"{metrics['test_f1']:.3f}")
    with col2:
        st.metric("Holdout AUC", f"{metrics['holdout_auc']:.3f}")
        st.metric("CV AUC", f"{metrics['cv_mean_auc']:.3f}")
    
    st.markdown("---")
    st.markdown(f"**Dataset:** {metrics['trained_on']}")
    st.markdown(f"**Features:** {feature_info['n_features']}")
    st.markdown(f"**Trained:** {metrics['timestamp'][:8]}")
    
    st.markdown("---")
    st.markdown("**Ensemble Models:**")
    for i, model_name in enumerate(metrics['top_models'], 1):
        st.write(f"{i}. {model_name}")
    
    st.markdown("---")
    st.markdown("**Anti-Overfitting:**")
    st.write("✓ 3-way data split")
    st.write("✓ Stratified CV")
    st.write("✓ Regularization")
    st.write("✓ Feature scaling")
    st.write("✓ Class balancing")

# Main content
tab1, tab2, tab3 = st.tabs(["🔮 Single Prediction", "📊 Batch CSV Upload", "ℹ️ About"])

# ==================== TAB 1: SINGLE PREDICTION ====================
with tab1:
    st.header("Enter Exoplanet Candidate Parameters")
    st.markdown("Fill in the values for a single exoplanet candidate")
    
    # Get feature names
    features = feature_info['original_features']
    
    # Create input fields in columns
    num_cols = 3
    cols = st.columns(num_cols)
    
    input_data = {}
    for i, feature in enumerate(features):
        col_idx = i % num_cols
        with cols[col_idx]:
            input_data[feature] = st.number_input(
                f"{feature}",
                value=0.0,
                format="%.6f",
                help=f"Enter value for {feature}",
                key=f"single_{feature}"
            )
    
    st.markdown("---")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_button = st.button("🚀 Predict Exoplanet", type="primary", use_container_width=True)
    
    if predict_button:
        try:
            # Prepare data
            input_df = pd.DataFrame([input_data])
            
            # Apply preprocessing pipeline
            input_imputed = imputer.transform(input_df)
            input_selected = selector.transform(input_imputed)
            input_scaled = scaler.transform(input_selected)
            
            # Predict
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            # Display results
            st.markdown("### 🎯 Prediction Results")
            st.markdown("---")
            
            # Main prediction
            if prediction == 1:
                st.markdown(f'<div class="planet-result">🪐 EXOPLANET DETECTED!</div>', 
                          unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="not-planet-result">❌ NOT AN EXOPLANET</div>', 
                          unsafe_allow_html=True)
            
            # Metrics
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric(
                    "Classification",
                    "Planet" if prediction == 1 else "Not Planet"
                )
            
            with col_b:
                st.metric(
                    "Planet Probability",
                    f"{probability[1]:.2%}"
                )
            
            with col_c:
                confidence_val = abs(probability[1] - 0.5)
                if confidence_val > 0.35:
                    confidence = "Very High"
                    conf_color = "🟢"
                elif confidence_val > 0.25:
                    confidence = "High"
                    conf_color = "🟡"
                elif confidence_val > 0.15:
                    confidence = "Medium"
                    conf_color = "🟠"
                else:
                    confidence = "Low"
                    conf_color = "🔴"
                
                st.metric(
                    "Confidence",
                    f"{conf_color} {confidence}"
                )
            
            # Probability bar
            st.markdown("#### Probability Distribution")
            prob_df = pd.DataFrame({
                'Class': ['Not Planet', 'Planet'],
                'Probability': [probability[0], probability[1]]
            })
            st.bar_chart(prob_df.set_index('Class'))
            
            # Interpretation
            st.markdown("---")
            st.markdown("#### 🔬 Scientific Interpretation")
            
            if prediction == 1 and probability[1] > 0.75:
                st.success("✅ **Strong planetary signal detected!** This candidate shows characteristics highly consistent with an exoplanet. Recommend for immediate follow-up observation and verification.")
            elif prediction == 1 and probability[1] > 0.6:
                st.info("🔵 **Likely exoplanet candidate.** The signal suggests a high probability of a planet, but additional observations would strengthen the confirmation.")
            elif prediction == 1:
                st.warning("⚠️ **Possible planet candidate.** The signal has planetary characteristics but with moderate confidence. Further analysis and observations recommended.")
            elif probability[1] > 0.4:
                st.warning("⚠️ **Ambiguous signal.** The data shows mixed characteristics. Review data quality and consider re-observation.")
            else:
                st.error("❌ **Likely false positive.** The signal characteristics are inconsistent with an exoplanet. Not recommended for follow-up.")
            
            # Show input summary
            with st.expander("📋 View Input Parameters"):
                st.dataframe(input_df.T, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.info("Please check that all input values are valid numbers.")

# ==================== TAB 2: BATCH CSV UPLOAD ====================
with tab2:
    st.header("📊 Batch Predictions from CSV")
    st.markdown("Upload a CSV file with multiple exoplanet candidates for batch processing")
    
    # Download template
    st.markdown("### Step 1: Download Template")
    st.markdown("Download the template CSV file with the correct feature columns:")
    
    try:
        template_df = pd.DataFrame(columns=feature_info['original_features'])
        csv_template = template_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV Template",
            data=csv_template,
            file_name="exoplanet_input_template.csv",
            mime="text/csv",
            help="Download this template and fill it with your data"
        )
    except Exception as e:
        st.error(f"Error creating template: {e}")
    
    st.markdown("---")
    
    # Upload file
    st.markdown("### Step 2: Upload Your CSV File")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with exoplanet candidate data"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            batch_df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(batch_df)} candidates.")
            
            # Show preview
            with st.expander("👁️ Preview Uploaded Data (first 5 rows)"):
                st.dataframe(batch_df.head(), use_container_width=True)
            
            # Validate columns
            missing_cols = set(feature_info['original_features']) - set(batch_df.columns)
            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
                st.info("Please ensure your CSV has all required columns. Download the template for reference.")
            else:
                st.success("✅ All required columns present!")
                
                # Predict button
                if st.button("🚀 Run Batch Predictions", type="primary", use_container_width=True):
                    with st.spinner("Processing predictions..."):
                        try:
                            # Ensure correct column order
                            batch_input = batch_df[feature_info['original_features']].copy()
                            
                            # Apply preprocessing pipeline
                            batch_imputed = imputer.transform(batch_input)
                            batch_selected = selector.transform(batch_imputed)
                            batch_scaled = scaler.transform(batch_selected)
                            
                            # Predict
                            batch_predictions = model.predict(batch_scaled)
                            batch_probabilities = model.predict_proba(batch_scaled)
                            
                            # Create results dataframe
                            results_df = batch_df.copy()
                            results_df['Prediction'] = ['Planet' if p == 1 else 'Not Planet' 
                                                       for p in batch_predictions]
                            results_df['Planet_Probability'] = batch_probabilities[:, 1]
                            results_df['Confidence'] = batch_probabilities.max(axis=1)
                            
                            # Add confidence levels
                            def get_confidence_level(conf):
                                if conf > 0.85:
                                    return "Very High"
                                elif conf > 0.75:
                                    return "High"
                                elif conf > 0.65:
                                    return "Medium"
                                else:
                                    return "Low"
                            
                            results_df['Confidence_Level'] = results_df['Confidence'].apply(get_confidence_level)
                            
                            # Summary statistics
                            st.markdown("### 📈 Batch Prediction Summary")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Total Candidates", len(results_df))
                            
                            with col2:
                                planet_count = (batch_predictions == 1).sum()
                                st.metric("Predicted Planets", planet_count)
                            
                            with col3:
                                not_planet_count = (batch_predictions == 0).sum()
                                st.metric("Not Planets", not_planet_count)
                            
                            with col4:
                                avg_prob = batch_probabilities[:, 1].mean()
                                st.metric("Avg Planet Prob", f"{avg_prob:.2%}")
                            
                            # Visualizations
                            st.markdown("---")
                            st.markdown("### 📊 Visualizations")
                            
                            viz_col1, viz_col2 = st.columns(2)
                            
                            with viz_col1:
                                # Prediction distribution
                                pred_counts = results_df['Prediction'].value_counts()
                                st.bar_chart(pred_counts)
                                st.caption("Distribution of Predictions")
                            
                            with viz_col2:
                                # Confidence distribution
                                conf_counts = results_df['Confidence_Level'].value_counts()
                                st.bar_chart(conf_counts)
                                st.caption("Confidence Level Distribution")
                            
                            # Show results table
                            st.markdown("---")
                            st.markdown("### 📋 Detailed Results")
                            
                            # Reorder columns for better readability
                            result_cols = ['Prediction', 'Planet_Probability', 'Confidence_Level'] + list(batch_df.columns)
                            st.dataframe(
                                results_df[result_cols].style.format({
                                    'Planet_Probability': '{:.4f}',
                                    'Confidence': '{:.4f}'
                                }),
                                use_container_width=True,
                                height=400
                            )
                            
                            # Download results
                            st.markdown("---")
                            st.markdown("### 💾 Download Results")
                            
                            csv_results = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv_results,
                                file_name=f"exoplanet_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                help="Download the full results with predictions"
                            )
                            
                            # Export high-confidence planets only
                            high_conf_planets = results_df[
                                (results_df['Prediction'] == 'Planet') & 
                                (results_df['Planet_Probability'] > 0.7)
                            ]
                            
                            if len(high_conf_planets) > 0:
                                csv_high_conf = high_conf_planets.to_csv(index=False)
                                st.download_button(
                                    label="⭐ Download High-Confidence Planets Only",
                                    data=csv_high_conf,
                                    file_name=f"high_confidence_planets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    help="Download only candidates with >70% planet probability"
                                )
                                st.success(f"Found {len(high_conf_planets)} high-confidence planet candidates!")
                        
                        except Exception as e:
                            st.error(f"Error during batch prediction: {e}")
                            st.info("Please check that your CSV file format matches the template.")
        
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            st.info("Please ensure you've uploaded a valid CSV file.")

# ==================== TAB 3: ABOUT ====================
with tab3:
    st.header("ℹ️ About This Model")
    
    st.markdown(f"""
    ### 🎯 Model Performance
    
    This ensemble machine learning model was trained on **{metrics['trained_on']}** data from NASA's Exoplanet Archive.
    
    #### 📊 Validation Results:
    
    | Metric | Test Set | Holdout Set | Cross-Validation |
    |--------|----------|-------------|------------------|
    | **AUC Score** | {metrics['test_auc']:.4f} | {metrics['holdout_auc']:.4f} | {metrics['cv_mean_auc']:.4f} ± {metrics['cv_std_auc']:.4f} |
    | **Accuracy** | {metrics['test_accuracy']:.4f} | {metrics['holdout_accuracy']:.4f} | - |
    | **F1 Score** | {metrics['test_f1']:.4f} | {metrics['holdout_f1']:.4f} | - |
    
    > **Note:** The holdout set represents completely unseen data, never used during model training or tuning.
    
    #### 🛡️ Anti-Overfitting Measures:
    
    - ✅ **Three-way data split** (64% train, 16% test, 20% holdout)
    - ✅ **Stratified cross-validation** (5-fold)
    - ✅ **Regularization** on all models (L1/L2, depth limits, sample requirements)
    - ✅ **Class balancing** for imbalanced datasets
    - ✅ **Feature scaling** (StandardScaler)
    - ✅ **Variance filtering** (removes uninformative features)
    - ✅ **Ensemble averaging** (reduces prediction variance)
    
    #### 🤖 Ensemble Composition:
    
    This model combines the predictions of multiple algorithms:
    
    {chr(10).join([f"**{i}.** {name}" for i, name in enumerate(metrics['top_models'], 1)])}
    
    #### 📏 Features Used ({feature_info['n_features']}):
    
    The model analyzes {feature_info['n_features']} physical and orbital parameters:
    
    {chr(10).join([f"- `{feat}`" for feat in feature_info['selected_features']])}
    
    ### 🌌 About Exoplanet Detection
    
    Exoplanets are planets that orbit stars outside our solar system. This AI model uses the **transit method** 
    to identify potential exoplanets by analyzing:
    
    - **Orbital characteristics** (period, eccentricity, semi-major axis)
    - **Planetary properties** (radius, mass, density, temperature)
    - **Stellar properties** (temperature, radius, mass, surface gravity)
    - **Observational data** (transit depth, duration, signal-to-noise ratio)
    
    ### 🏆 NASA Space Apps Challenge 2025
    
    **Challenge:** A World Away: Hunting for Exoplanets with AI
    
    This application demonstrates how machine learning can accelerate exoplanet discovery by:
    - Automating the classification of planetary candidates
    - Providing confidence estimates for follow-up prioritization
    - Processing large datasets efficiently
    - Generalizing across different space missions (TESS, Kepler, K2)
    
    ### 🔬 Scientific Impact
    
    Automated exoplanet detection helps astronomers:
    - **Save time:** Reduce manual vetting from hours to seconds
    - **Increase accuracy:** Ensemble methods outperform individual algorithms
    - **Enable discoveries:** Process millions of candidates from space missions
    - **Prioritize resources:** Focus follow-up observations on high-probability candidates
    
    ### 📚 References
    
    - NASA Exoplanet Archive: https://exoplanetarchive.ipac.caltech.edu/
    - TESS Mission: https://tess.mit.edu/
    - Kepler Mission: https://www.nasa.gov/kepler
    - K2 Mission: https://www.nasa.gov/k2
    
    ### 👥 Team Information
    
    Built for NASA Space Apps Challenge 2025
    
    **Goal:** Win Local Awards + Global Nominee Selection
    
    ---
    
    *This model is for research and educational purposes. All predictions should be validated 
    through professional astronomical observation and analysis.*
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🌌 Exoplanet Detection AI | NASA Space Apps Challenge 2025</p>
    <p>Built with Streamlit • Powered by Ensemble Machine Learning</p>
</div>
""", unsafe_allow_html=True)
