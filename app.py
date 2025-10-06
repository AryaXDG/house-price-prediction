import streamlit as st
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("🏠 House Price Prediction App")
st.markdown("---")
st.markdown("""
**Predict house prices using multiple ML models!**  
Enter the house details below and get predictions from 6 different machine learning models.
""")

# Load models and preprocessing components
@st.cache_data
def load_models_and_preprocessors():
    """Load all trained models and preprocessing components"""
    models_dir = Path('saved_models')
    
    if not models_dir.exists():
        st.error("❌ Models directory not found! Please ensure 'saved_models' folder exists.")
        return None, None, None, None, None
    
    try:
        # Load models
        models = {}
        model_files = {
            'Linear Regression': 'linear_regression_model.pkl',
            'Ridge': 'ridge_model.pkl', 
            'Lasso': 'lasso_model.pkl',
            'Decision Tree': 'decision_tree_model.pkl',
            'KNN': 'knn_model.pkl',
            'Random Forest': 'random_forest_model.pkl'
        }
        
        for name, filename in model_files.items():
            model_path = models_dir / filename
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    models[name] = pickle.load(f)
        
        # Load preprocessing components
        with open(models_dir / 'imputer.pkl', 'rb') as f:
            imputer = pickle.load(f)
        
        with open(models_dir / 'scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open(models_dir / 'feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        with open(models_dir / 'label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        
        return models, imputer, scaler, feature_names, label_encoders
        
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None, None, None, None

# Load all components
models, imputer, scaler, feature_names, label_encoders = load_models_and_preprocessors()

if models is None:
    st.stop()

# Create input form
st.markdown("## 🏡 Enter House Details")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    # Numeric inputs
    area = st.number_input(
        "🏠 **Area (sq ft)**", 
        min_value=300, 
        max_value=15000, 
        value=2000,
        step=50,
        help="Total area of the house in square feet"
    )
    
    bedrooms = st.slider(
        "🛏️ **Bedrooms**", 
        min_value=1, 
        max_value=12, 
        value=3,
        help="Number of bedrooms"
    )
    
    bathrooms = st.slider(
        "🚿 **Bathrooms**", 
        min_value=1, 
        max_value=12, 
        value=2,
        help="Number of bathrooms"
    )
    
    floors = st.slider(
        "🏢 **Floors**", 
        min_value=1, 
        max_value=7, 
        value=2,
        help="Number of floors"
    )

with col2:
    # Year and categorical inputs
    year_built = st.slider(
        "📅 **Year Built**", 
        min_value=1900, 
        max_value=2025, 
        value=2000,
        help="Year the house was built"
    )
    
    # Get available categories from label encoders
    location_options = list(label_encoders['Location'].classes_) if 'Location' in label_encoders else ['Downtown', 'Urban', 'Suburban', 'Rural']
    location = st.selectbox(
        "📍 **Location**", 
        options=location_options,
        index=2 if len(location_options) > 2 else 0,
        help="Location type of the house"
    )
    
    condition_options = list(label_encoders['Condition'].classes_) if 'Condition' in label_encoders else ['Poor', 'Fair', 'Good', 'Excellent']
    condition = st.selectbox(
        "⭐ **Condition**", 
        options=condition_options,
        index=2 if len(condition_options) > 2 else 0,
        help="Overall condition of the house"
    )
    
    garage_options = list(label_encoders['Garage'].classes_) if 'Garage' in label_encoders else ['No', 'Yes', 'Attached', 'Detached']
    garage = st.selectbox(
        "🚗 **Garage**", 
        options=garage_options,
        index=1 if len(garage_options) > 1 else 0,
        help="Garage availability"
    )

# Prediction function
def predict_house_price(models, imputer, scaler, feature_names, label_encoders, 
                       area, bedrooms, bathrooms, floors, year_built, 
                       location, condition, garage):
    """Make predictions using all available models"""
    try:
        # Calculate house age
        house_age = 2023 - year_built
        
        # Encode categorical features
        location_encoded = label_encoders['Location'].transform([location])[0]
        condition_encoded = label_encoders['Condition'].transform([condition])[0]
        garage_encoded = label_encoders['Garage'].transform([garage])[0]
        
        # Create feature dictionary
        feature_dict = {
            'Area': area,
            'Bedrooms': bedrooms,
            'Bathrooms': bathrooms,
            'Floors': floors,
            'House_Age': house_age,
            'Location_encoded': location_encoded,
            'Condition_encoded': condition_encoded,
            'Garage_encoded': garage_encoded
        }
        
        # Create feature vector in the same order as training
        feature_vector = []
        for feature in feature_names:
            if feature in feature_dict:
                feature_vector.append(feature_dict[feature])
            else:
                feature_vector.append(0)  # Default value for missing features
        
        # Convert to numpy array and reshape
        X_new = np.array(feature_vector).reshape(1, -1)
        
        # Apply preprocessing
        X_new_imputed = imputer.transform(X_new)
        X_new_scaled = scaler.transform(X_new_imputed)
        
        # Make predictions with all models
        predictions = {}
        for name, model in models.items():
            pred = model.predict(X_new_scaled)[0]
            predictions[name] = max(0, pred)  # Ensure non-negative price
        
        return predictions
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None

# Prediction button
st.markdown("---")
if st.button("🔮 **Predict House Price**", type="primary", use_container_width=True):
    
    # Show input summary
    with st.expander("📋 **Input Summary**", expanded=False):
        st.write(f"**🏠 Area:** {area:,} sq ft")
        st.write(f"**🛏️ Bedrooms:** {bedrooms}")
        st.write(f"**🚿 Bathrooms:** {bathrooms}")
        st.write(f"**🏢 Floors:** {floors}")
        st.write(f"**📅 Year Built:** {year_built} (Age: {2023 - year_built} years)")
        st.write(f"**📍 Location:** {location}")
        st.write(f"**⭐ Condition:** {condition}")
        st.write(f"**🚗 Garage:** {garage}")
    
    # Make predictions
    with st.spinner("🤖 Making predictions..."):
        predictions = predict_house_price(
            models, imputer, scaler, feature_names, label_encoders,
            area, bedrooms, bathrooms, floors, year_built, 
            location, condition, garage
        )
    
    if predictions:
        st.markdown("## 💰 **Prediction Results**")
        
        # Sort predictions by value
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        # Display predictions in cards
        for i, (model_name, price) in enumerate(sorted_predictions):
            
            # Different colors for different ranks
            if i == 0:  # Highest prediction
                st.success(f"🥇 **{model_name}**: ${price:,.0f}")
            elif i == 1:  # Second highest
                st.info(f"🥈 **{model_name}**: ${price:,.0f}")
            elif i == 2:  # Third highest  
                st.warning(f"🥉 **{model_name}**: ${price:,.0f}")
            else:
                st.write(f"🔸 **{model_name}**: ${price:,.0f}")
        
        # Calculate statistics
        prices = list(predictions.values())
        avg_price = np.mean(prices)
        std_price = np.std(prices)
        min_price = min(prices)
        max_price = max(prices)
        
        # Display statistics
        st.markdown("---")
        st.markdown("## 📊 **Ensemble Statistics**")
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            st.metric(
                label="💵 **Average**",
                value=f"${avg_price:,.0f}"
            )
        
        with stat_col2:
            st.metric(
                label="📈 **Highest**", 
                value=f"${max_price:,.0f}"
            )
        
        with stat_col3:
            st.metric(
                label="📉 **Lowest**",
                value=f"${min_price:,.0f}"
            )
        
        with stat_col4:
            st.metric(
                label="📊 **Std Dev**",
                value=f"${std_price:,.0f}"
            )
        
        # Price range
        st.info(f"**💡 Price Range:** ${avg_price - std_price:,.0f} - ${avg_price + std_price:,.0f}")
        
        # Show confidence level based on standard deviation
        cv = (std_price / avg_price) * 100  # Coefficient of variation
        
        if cv < 5:
            confidence = "🟢 **High Confidence** - Models are in strong agreement"
        elif cv < 15:
            confidence = "🟡 **Medium Confidence** - Models show moderate agreement"  
        else:
            confidence = "🔴 **Low Confidence** - Models show significant disagreement"
        
        st.markdown(f"**Model Agreement:** {confidence}")

# Add sidebar with information
st.sidebar.markdown("## ℹ️ **About This App**")
st.sidebar.info("""
This app uses 6 different machine learning models to predict house prices:

🔸 **Linear Regression**  
🔸 **Ridge Regression**  
🔸 **Lasso Regression**  
🔸 **Decision Tree**  
🔸 **K-Nearest Neighbors**  
🔸 **Random Forest**  

Each model provides a different perspective on the prediction.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("## 💡 **Tips**")
st.sidebar.markdown("""
- **Area** has the strongest impact on price
- **Location** significantly affects pricing
- **Condition** and **Age** influence value
- **Multiple predictions** provide better insights
- **Price range** indicates prediction uncertainty
""")

st.sidebar.markdown("---")
st.sidebar.markdown("## 🏠 **Quick Examples**")

# Quick example buttons
if st.sidebar.button("🏡 Suburban Family Home"):
    st.session_state.update({
        'area': 2500, 'bedrooms': 4, 'bathrooms': 3, 'floors': 2,
        'year_built': 2005, 'location': 'Suburban', 'condition': 'Good', 'garage': 'Attached'
    })
    st.experimental_rerun()

if st.sidebar.button("🏢 Downtown Condo"):
    st.session_state.update({
        'area': 1200, 'bedrooms': 2, 'bathrooms': 2, 'floors': 1,
        'year_built': 2015, 'location': 'Downtown', 'condition': 'Excellent', 'garage': 'No'
    })
    st.experimental_rerun()

if st.sidebar.button("🌾 Rural Farmhouse"):
    st.session_state.update({
        'area': 3500, 'bedrooms': 5, 'bathrooms': 3, 'floors': 2,
        'year_built': 1985, 'location': 'Rural', 'condition': 'Fair', 'garage': 'Detached'
    })
    st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        🏠 House Price Prediction App | Built with Streamlit & Scikit-learn
    </div>
    """, 
    unsafe_allow_html=True
)