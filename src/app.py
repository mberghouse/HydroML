import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.sklearn_models import ModelSelector
from models.pytorch_models import ModelFactory
from utils.data_processing import clean_feature_name, drop_non_numeric_data
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_diabetes
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, mean_absolute_error
import torch
import torch.nn as nn
import plotly.express as px
from utils.nwis_data import (
    get_site_data, 
    get_parameter_name,
    collect_single_parameter_data,
    collect_multiple_parameter_data,
    collect_high_frequency_data
)

def train_sklearn_model(data, model_name, target_column, parameters, model_selector):
    try:
        X = data.drop(columns=[target_column])
        y = data[target_column]

        X.columns = [clean_feature_name(col) for col in X.columns]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model_class = model_selector.get_model(model_name)
        model = model_class(**parameters)
        pipeline = make_pipeline(StandardScaler(), model)
        
        with st.spinner('Training model...'):
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Predictions vs Actual
            ax1.scatter(y_test, y_pred)
            ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax1.set_xlabel("True Values")
            ax1.set_ylabel("Predictions")
            ax1.set_title("Predictions vs True Values")
            
            # Residuals
            residuals = y_test - y_pred
            ax2.hist(residuals, bins=30)
            ax2.set_xlabel("Residual Value")
            ax2.set_ylabel("Count")
            ax2.set_title("Residuals Distribution")
            
            st.pyplot(fig)
            
            # Display metrics
            st.write("### Model Performance Metrics")
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            metrics_df = pd.DataFrame({
                'Metric': ['MSE', 'RMSE', 'R²', 'MAE'],
                'Value': [mse, rmse, r2, mae]
            })
            st.table(metrics_df)
            
            # Feature importance if available
            if hasattr(model, "feature_importances_"):
                st.write("### Feature Importances")
                importances = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(importances['feature'], importances['importance'])
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                st.dataframe(importances)
                
    except Exception as e:
        st.error(f"Error training model: {str(e)}")

def train_pytorch_model(model, data, target_column, epochs=10, batch_size=32, learning_rate=0.001):
    try:
        X = data.drop(columns=[target_column]).values
        y = data[target_column].values
        
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X, y.reshape(-1, 1))
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        train_losses = []
        test_losses = []
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            with st.spinner(f'Training epoch {epoch+1}/{epochs}...'):
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                train_losses.append(epoch_loss / len(train_loader))
                
                # Validation
                model.eval()
                test_loss = 0
                predictions = []
                actuals = []
                
                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        outputs = model(batch_X)
                        test_loss += criterion(outputs, batch_y).item()
                        predictions.extend(outputs.numpy().flatten())
                        actuals.extend(batch_y.numpy().flatten())
                
                test_losses.append(test_loss / len(test_loader))
                
                # Update progress
                st.write(f"Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}")
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(train_losses, label='Train Loss')
        ax1.plot(test_losses, label='Test Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.set_title('Training and Test Loss')
        
        # Predictions vs Actuals
        ax2.scatter(actuals, predictions)
        ax2.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--')
        ax2.set_xlabel('Actual Values')
        ax2.set_ylabel('Predictions')
        ax2.set_title('Predictions vs Actual Values')
        
        st.pyplot(fig)
        
        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        st.write(f"Final MSE: {mse:.4f}")
        st.write(f"R² Score: {r2:.4f}")
        
        return model
        
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None

def pytorch_model_designer(data=None, target_column=None):
    st.subheader("PyTorch Model Designer")
    
    if data is None:
        st.error("Please load data first")
        return
        
    data_type = st.radio("Select input data type", ["Tabular", "Image"])
    
    if data_type == "Tabular":
        input_dim = data.drop(columns=[target_column]).shape[1]
        st.write(f"Input dimension: {input_dim} (based on data)")
        output_dim = 1
        
        num_layers = st.slider("Number of hidden layers", 1, 5, 2)
        hidden_dims = []
        
        for i in range(num_layers):
            dim = st.number_input(f"Hidden layer {i+1} dimension", 
                                min_value=1, 
                                value=max(input_dim, output_dim))
            hidden_dims.append(dim)
            
        if st.button("Create and Train Model"):
            model = ModelFactory.get_custom_mlp(input_dim, hidden_dims, output_dim)
            st.write("Model architecture:")
            st.code(str(model))
            
            epochs = st.slider("Number of epochs", 5, 100, 10)
            batch_size = st.slider("Batch size", 8, 128, 32)
            learning_rate = st.number_input("Learning rate", value=0.001, format="%.4f")
            
            trained_model = train_pytorch_model(model, data, target_column, 
                                             epochs=epochs, 
                                             batch_size=batch_size, 
                                             learning_rate=learning_rate)

    else:  # Image
        available_models = ModelFactory.get_available_image_models()
        model_name = st.selectbox("Select image model", available_models)
        num_classes = st.number_input("Number of classes", min_value=1, value=1000)
        pretrained = st.checkbox("Use pretrained weights", value=True)
        
        if st.button("Create Image Model"):
            model = ModelFactory.get_timm_model(model_name, num_classes, pretrained)
            st.write("Model architecture:")
            st.code(str(model))

def nwis_data_selector():
    st.subheader("USGS NWIS Data Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # State selection
        state_code = st.selectbox("Select State", 
                                ["NV", "CA", "OR", "WA", "ID", "AZ", "UT"],
                                index=0)
        # Optional county selection
        county = st.text_input("County Code (optional, e.g., '001')", 
                             help="Enter 3-digit county code")
    
    with col2:
        # Year range selection
        start_year = st.number_input("Start Year (optional)", 
                                   min_value=1900, 
                                   max_value=2023, 
                                   value=None)
        end_year = st.number_input("End Year (optional)", 
                                 min_value=1900, 
                                 max_value=2023, 
                                 value=None)
    
    # Parameter selection
    parameter_codes = {
        'Discharge (00060)': '00060',
        'Temperature (00010)': '00010',
        'Conductivity (00095)': '00095',
        'pH (00400)': '00400',
        'Dissolved Oxygen (00300)': '00300',
        'Phosphorus (00665)': '00665',
        'Nitrate + Nitrite (00631)': '00631',
        'Ammonia (00608)': '00608',
        'Sulfate (00945)': '00945',
        'Chloride (00940)': '00940'
    }
    
    data_collection_method = st.radio(
        "Data Collection Method",
        ["Single Parameter", "Multiple Parameters (Sites with all parameters)", "High-Frequency Sites"]
    )
    
    if data_collection_method == "Single Parameter":
        selected_params = st.multiselect("Select Parameters", 
                                       list(parameter_codes.keys()),
                                       default=['Discharge (00060)'])
        
        if st.button("Fetch NWIS Data"):
            return collect_single_parameter_data(state_code, selected_params, parameter_codes, 
                                              county, start_year, end_year)
            
    elif data_collection_method == "Multiple Parameters (Sites with all parameters)":
        selected_params = st.multiselect("Select Parameters", 
                                       list(parameter_codes.keys()),
                                       default=['Discharge (00060)', 'Temperature (00010)'])
        min_values = st.number_input("Minimum number of values per site", value=10000, step=1000)
        
        if st.button("Find Sites and Fetch Data"):
            return collect_multiple_parameter_data(state_code, selected_params, parameter_codes, 
                                                min_values, county, start_year, end_year)
            
    else:  # High-Frequency Sites
        param = st.selectbox("Select Parameter", list(parameter_codes.keys()))
        min_values = st.number_input("Minimum number of values", value=10000, step=1000)
        
        if st.button("Find High-Frequency Sites"):
            return collect_high_frequency_data(state_code, param, parameter_codes, min_values,
                                            county, start_year, end_year)

def main():
    st.title("HydroML - Machine Learning Model Designer")
    
    # Data source selection
    data_source = st.sidebar.radio("Select Data Source", 
                                  ["Upload CSV", "NWIS Data", "Toy Dataset"])
    
    data = None
    if data_source == "Upload CSV":
        # File upload
        file = st.file_uploader("Upload CSV file", type=["csv"])
        if file is not None:
            try:
                data = pd.read_csv(file, encoding='latin1', low_memory=False)
            except UnicodeDecodeError:
                data = pd.read_csv(file, encoding='utf-16', low_memory=False)
            data = drop_non_numeric_data(data)
            
    elif data_source == "NWIS Data":
        data = nwis_data_selector()
        
    else:  # Toy Dataset
        toy_dataset = st.sidebar.selectbox("Select a toy dataset", 
                                         ("Diabetes (Regression)", "Iris (Classification)"))
        
        if toy_dataset == "Diabetes (Regression)":
            diabetes = load_diabetes()
            data = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
            data['target'] = diabetes.target
        elif toy_dataset == "Iris (Classification)":
            iris = load_iris()
            data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
            data['target'] = iris.target

    # Only show model options if data is loaded
    if data is not None:
        model_type = st.sidebar.radio("Select Model Type", ["Sklearn", "PyTorch"])
        
        # Target column selection
        target_column = st.sidebar.selectbox("Select the target column", data.columns)
        
        if model_type == "Sklearn":
            # Model selection
            model_selector = ModelSelector()
            model_name = st.sidebar.selectbox("Select a model", 
                list(model_selector.regression_models.keys()) + 
                list(model_selector.classification_models.keys()))

            # Model parameters
            parameters = model_selector.get_model_parameters(model_name)
            param_container = st.sidebar.expander("Model Parameters")
            with param_container:
                for param, value in parameters.items():
                    if isinstance(value, float):
                        parameters[param] = st.number_input(param, value=value, format="%.5f")
                    elif isinstance(value, int):
                        parameters[param] = st.number_input(param, value=value, step=1)
                    else:
                        parameters[param] = st.text_input(param, value=str(value))

            if st.button("Train Model"):
                train_sklearn_model(data, model_name, target_column, parameters, model_selector)
        else:
            pytorch_model_designer(data, target_column)
    else:
        st.info("Please select a data source to begin")

if __name__ == "__main__":
    main()
