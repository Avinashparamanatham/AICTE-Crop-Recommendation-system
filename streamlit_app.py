import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="🌾 Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #228B22;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #F0F8FF;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 20px 0;
    }
    .info-box {
        background-color: #E6F3FF;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and scaler"""
    try:
        if os.path.exists('models/crop_recommendation_model.pkl'):
            with open('models/crop_recommendation_model.pkl', 'rb') as file:
                model_data = pickle.load(file)
                return model_data['model'], model_data['scaler']
        else:
            # If no saved model, create a dummy one for demo
            st.warning("No pre-trained model found. Using demo model.")
            return create_demo_model()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return create_demo_model()

def create_demo_model():
    """Create a demo model if the trained model is not available"""
    # This is a placeholder - in production, you'd want to train with real data
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    scaler = StandardScaler()
    
    # Create dummy training data for demo
    np.random.seed(42)
    X_dummy = np.random.rand(100, 7)
    y_dummy = np.random.choice(['rice', 'wheat', 'corn', 'tomato'], 100)
    
    X_scaled = scaler.fit_transform(X_dummy)
    model.fit(X_scaled, y_dummy)
    
    return model, scaler

@st.cache_data
def load_crop_data():
    """Load sample crop data for visualization"""
    try:
        if os.path.exists('Crop_recommendation.csv'):
            return pd.read_csv('Crop_recommendation.csv')
        else:
            # Create sample data if CSV not available
            return create_sample_data()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return create_sample_data()

def create_sample_data():
    """Create sample crop data for demo"""
    np.random.seed(42)
    crops = ['rice', 'wheat', 'corn', 'tomato', 'potato', 'cotton', 'sugarcane']
    
    data = []
    for crop in crops:
        for _ in range(50):
            data.append({
                'N': np.random.normal(80, 20),
                'P': np.random.normal(60, 15),
                'K': np.random.normal(40, 10),
                'temperature': np.random.normal(25, 5),
                'humidity': np.random.normal(70, 15),
                'ph': np.random.normal(6.5, 1),
                'rainfall': np.random.normal(150, 50),
                'label': crop
            })
    
    return pd.DataFrame(data)

def predict_crop(model, scaler, features):
    """Make crop prediction"""
    try:
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_scaled)[0]
            classes = model.classes_
            prob_dict = dict(zip(classes, probabilities))
            return prediction, prob_dict
        else:
            return prediction, None
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

def validate_inputs(n, p, k, temp, humidity, ph, rainfall):
    """Validate user inputs"""
    errors = []
    
    if not (0 <= n <= 200):
        errors.append("Nitrogen should be between 0-200 kg/ha")
    if not (0 <= p <= 200):
        errors.append("Phosphorus should be between 0-200 kg/ha")
    if not (0 <= k <= 300):
        errors.append("Potassium should be between 0-300 kg/ha")
    if not (0 <= temp <= 50):
        errors.append("Temperature should be between 0-50°C")
    if not (0 <= humidity <= 100):
        errors.append("Humidity should be between 0-100%")
    if not (0 <= ph <= 14):
        errors.append("pH should be between 0-14")
    if not (0 <= rainfall <= 500):
        errors.append("Rainfall should be between 0-500mm")
    
    return errors

def create_radar_chart(features, feature_names):
    """Create a radar chart for input features"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=features,
        theta=feature_names,
        fill='toself',
        name='Input Values',
        line_color='rgb(46, 139, 87)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(features) * 1.2]
            )),
        showlegend=True,
        title="Input Parameters Visualization"
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🌾 Crop Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown("### Get personalized crop recommendations based on soil and climate conditions")
    
    # Load model and data
    model, scaler = load_model()
    data = load_crop_data()
    
    # Sidebar
    st.sidebar.markdown("## 📊 Navigation")
    page = st.sidebar.selectbox("Choose a page:", ["🏠 Home", "📈 Data Visualization", "ℹ️ About"])
    
    if page == "🏠 Home":
        home_page(model, scaler)
    elif page == "📈 Data Visualization":
        data_visualization_page(data)
    elif page == "ℹ️ About":
        about_page()

def home_page(model, scaler):
    """Main prediction page"""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">🌱 Input Parameters</h2>', unsafe_allow_html=True)
        
        # Soil parameters
        st.markdown("**Soil Parameters:**")
        n = st.slider("Nitrogen (N) - kg/ha", 0, 200, 90, help="Amount of nitrogen in the soil")
        p = st.slider("Phosphorus (P) - kg/ha", 0, 200, 50, help="Amount of phosphorus in the soil")
        k = st.slider("Potassium (K) - kg/ha", 0, 300, 40, help="Amount of potassium in the soil")
        ph = st.slider("pH Level", 0.0, 14.0, 6.5, 0.1, help="Soil acidity/alkalinity level")
        
        # Climate parameters
        st.markdown("**Climate Parameters:**")
        temp = st.slider("Temperature - °C", 0, 50, 25, help="Average temperature")
        humidity = st.slider("Humidity - %", 0, 100, 70, help="Relative humidity")
        rainfall = st.slider("Rainfall - mm", 0, 500, 150, help="Annual rainfall")
        
        # Predict button
        if st.button("🔮 Get Crop Recommendation", type="primary"):
            features = [n, p, k, temp, humidity, ph, rainfall]
            
            # Validate inputs
            errors = validate_inputs(n, p, k, temp, humidity, ph, rainfall)
            
            if errors:
                for error in errors:
                    st.error(error)
            else:
                prediction, probabilities = predict_crop(model, scaler, features)
                
                if prediction:
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.success(f"🎯 **Recommended Crop: {prediction.upper()}**")
                    
                    if probabilities:
                        st.markdown("**Prediction Confidence:**")
                        for crop, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]:
                            st.write(f"• {crop}: {prob:.2%}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h2 class="sub-header">📊 Parameter Visualization</h2>', unsafe_allow_html=True)
        
        # Create radar chart
        features = [n, p, k, temp, humidity, ph, rainfall]
        feature_names = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall']
        
        fig = create_radar_chart(features, feature_names)
        st.plotly_chart(fig, use_container_width=True)
        
        # Parameter summary
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("**Parameter Summary:**")
        st.write(f"• Nitrogen: {n} kg/ha")
        st.write(f"• Phosphorus: {p} kg/ha")
        st.write(f"• Potassium: {k} kg/ha")
        st.write(f"• Temperature: {temp}°C")
        st.write(f"• Humidity: {humidity}%")
        st.write(f"• pH: {ph}")
        st.write(f"• Rainfall: {rainfall}mm")
        st.markdown('</div>', unsafe_allow_html=True)

def data_visualization_page(data):
    """Data visualization page"""
    st.markdown('<h2 class="sub-header">📈 Dataset Analysis</h2>', unsafe_allow_html=True)
    
    if data is not None and not data.empty:
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", len(data))
        with col2:
            st.metric("Number of Crops", data['label'].nunique())
        with col3:
            st.metric("Features", len(data.columns) - 1)
        with col4:
            st.metric("Missing Values", data.isnull().sum().sum())
        
        # Crop distribution
        st.markdown("### Crop Distribution")
        crop_counts = data['label'].value_counts()
        fig = px.bar(x=crop_counts.index, y=crop_counts.values, 
                     title="Distribution of Crops in Dataset")
        fig.update_layout(xaxis_title="Crop", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature distributions
        st.markdown("### Feature Distributions")
        feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        selected_feature = st.selectbox("Select feature to visualize:", feature_cols)
        
        if selected_feature in data.columns:
            fig = px.box(data, x='label', y=selected_feature, 
                        title=f'Distribution of {selected_feature} by Crop')
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.markdown("### Feature Correlation")
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            fig = px.imshow(numeric_data.corr(), text_auto=True, aspect="auto",
                           title="Feature Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)
        
        # Raw data
        if st.checkbox("Show raw data"):
            st.dataframe(data)

def about_page():
    """About page"""
    st.markdown('<h2 class="sub-header">ℹ️ About the Crop Recommendation System</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Purpose
    This application helps farmers and agricultural professionals make informed decisions about crop selection based on soil and climate conditions.
    
    ### 🔬 How It Works
    The system uses machine learning algorithms to analyze:
    - **Soil Parameters**: Nitrogen, Phosphorus, Potassium levels, and pH
    - **Climate Conditions**: Temperature, Humidity, and Rainfall
    
    ### 🤖 Technology Stack
    - **Machine Learning**: Scikit-learn (Random Forest, SVM, KNN)
    - **Frontend**: Streamlit
    - **Visualization**: Plotly, Matplotlib
    - **Data Processing**: Pandas, NumPy
    
    ### 📊 Features
    - Interactive parameter input with sliders
    - Real-time crop predictions
    - Data visualization and analysis
    - Parameter validation and recommendations
    
    ### 🌾 Supported Crops
    The system can recommend various crops including:
    Rice, Wheat, Corn, Cotton, Sugarcane, Tomato, Potato, and more.
    
    ### 📈 Model Performance
    The machine learning model is trained on agricultural datasets and optimized for accuracy using cross-validation and hyperparameter tuning.
    
    ### 🚀 Usage Tips
    1. Adjust the sliders to match your soil and climate conditions
    2. Ensure values are within realistic ranges
    3. Consider the prediction confidence scores
    4. Use the visualization page to understand data patterns
    
    ### 👨‍💻 Development
    This system demonstrates the application of machine learning in agriculture for sustainable farming practices.
    """)
    
    st.markdown("---")
    st.info("🌱 **Remember**: This tool provides recommendations based on data patterns. Always consult with local agricultural experts for the best results!")

if __name__ == "__main__":
    main()