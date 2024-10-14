import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('winequality-red.csv')
    return df

# Preprocess data
def preprocess_data(df):
    X = df.drop('quality', axis=1)
    y = df['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Train models
@st.cache_resource
def train_models(X_train_scaled, y_train):
    linear_reg = LinearRegression()
    lasso_reg = Lasso(alpha=0.01)
    mlp_reg = MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42)
    
    base_models = [
        ('linear', LinearRegression()),
        ('lasso', Lasso(alpha=0.01)),
        ('mlp', MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42))
    ]
    meta_model = LinearRegression()
    stacking_reg = StackingRegressor(estimators=base_models, final_estimator=meta_model)
    
    linear_reg.fit(X_train_scaled, y_train)
    lasso_reg.fit(X_train_scaled, y_train)
    mlp_reg.fit(X_train_scaled, y_train)
    stacking_reg.fit(X_train_scaled, y_train)
    
    return {
        'Linear Regression': linear_reg,
        'Lasso Regression': lasso_reg,
        'MLP Regressor': mlp_reg,
        'Stacking Regressor': stacking_reg
    }

# Make predictions
def make_predictions(model, X_test_scaled):
    return model.predict(X_test_scaled)

# Evaluate model
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

# Streamlit app
def main():
    st.title('Wine Quality Prediction')
    
    # Load and preprocess data
    df = load_data()
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(df)
    
    # Train models
    models = train_models(X_train_scaled, y_train)
    
    # User input form
    st.header('Input Wine Parameters')
    
    input_features = {}
    col1, col2 = st.columns(2)
    
    with col1:
        input_features['fixed acidity'] = st.number_input('Fixed Acidity', min_value=0.0, max_value=20.0, value=7.0, step=0.1)
        input_features['volatile acidity'] = st.number_input('Volatile Acidity', min_value=0.0, max_value=2.0, value=0.5, step=0.01)
        input_features['citric acid'] = st.number_input('Citric Acid', min_value=0.0, max_value=1.0, value=0.3, step=0.01)
        input_features['residual sugar'] = st.number_input('Residual Sugar', min_value=0.0, max_value=20.0, value=2.0, step=0.1)
        input_features['chlorides'] = st.number_input('Chlorides', min_value=0.0, max_value=1.0, value=0.08, step=0.001)
        input_features['free sulfur dioxide'] = st.number_input('Free Sulfur Dioxide', min_value=0, max_value=100, value=30, step=1)
    
    with col2:
        input_features['total sulfur dioxide'] = st.number_input('Total Sulfur Dioxide', min_value=0, max_value=300, value=100, step=1)
        input_features['density'] = st.number_input('Density', min_value=0.9, max_value=1.1, value=0.996, step=0.0001)
        input_features['pH'] = st.number_input('pH', min_value=2.0, max_value=5.0, value=3.2, step=0.01)
        input_features['sulphates'] = st.number_input('Sulphates', min_value=0.0, max_value=2.0, value=0.5, step=0.01)
        input_features['alcohol'] = st.number_input('Alcohol', min_value=8.0, max_value=15.0, value=10.0, step=0.1)
    
    # Model selection
    selected_model = st.selectbox('Select Model for Prediction', list(models.keys()))
    
    # Create a DataFrame from user input
    input_df = pd.DataFrame([input_features])
    
    # Scale the input
    input_scaled = scaler.transform(input_df)
    
    if st.button('Predict Wine Quality'):
        # Make prediction
        prediction = models[selected_model].predict(input_scaled)[0]
        
        # Display prediction
        st.header('Wine Quality Prediction')
        st.write(f'{selected_model} prediction: {prediction:.2f}')
    
    # Model evaluation
    if st.button('Evaluate All Models'):
        st.header('Model Evaluation')
        
        for name, model in models.items():
            y_pred = make_predictions(model, X_test_scaled)
            mae, rmse, r2 = evaluate_model(y_test, y_pred)
            
            st.subheader(name)
            st.write(f'MAE: {mae:.4f}')
            st.write(f'RMSE: {rmse:.4f}')
            st.write(f'R-squared: {r2:.4f}')

if __name__ == '__main__':
    main()