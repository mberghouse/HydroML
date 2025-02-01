import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.sklearn_models import ModelSelector
from models.pytorch_models import ModelFactory
from utils.data_processing import clean_feature_name, drop_non_numeric_data
from sklearn.model_selection import train_test_split, cross_val_score
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
import optuna
from sklearn.preprocessing import LabelEncoder

def train_sklearn_model(data, model_name, target_column, parameters, model_selector):
    try:
        X = data.drop(columns=[target_column])
        y = data[target_column]
        X.columns = [clean_feature_name(col) for col in X.columns]
        
        use_optuna = st.checkbox("Use Optuna for hyperparameter optimization")
        
        if use_optuna:
            n_trials = st.number_input("Number of Optuna trials", min_value=1, value=20)
            study_duration = st.number_input("Study duration (minutes)", min_value=1, value=10)
            
            def objective(trial):
                # Define parameter search space based on model type
                optimized_params = model_selector.get_optuna_parameters(model_name, trial)
                model = model_selector.get_model(model_name)(**optimized_params)
                pipeline = make_pipeline(StandardScaler(), model)
                
                score = cross_val_score(pipeline, X, y, cv=5).mean()
                return score
            
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials, timeout=60*study_duration)
            
            parameters = study.best_params
            st.write("### Best Parameters Found:")
            st.write(parameters)
        
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

def train_pytorch_model(model, data, target_column, epochs=10, batch_size=32, 
                       learning_rate=0.001, optimizer_name='Adam', 
                       criterion=None, return_best_loss=False):
    try:
        if criterion is None:
            criterion = nn.MSELoss()
            
        # Determine if classification
        unique_values = len(data[target_column].unique())
        is_classification = unique_values < 10
        
        # Split data into train and validation sets
        X = data.drop(columns=[target_column]).values
        y = data[target_column].values
        
        if is_classification:
            # Convert labels to integers if they aren't already
            if not np.issubdtype(y.dtype, np.integer):
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, 
                                                         random_state=42)
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        X_val = torch.FloatTensor(X_val)
        
        if is_classification:
            y_train = torch.LongTensor(y_train)
            y_val = torch.LongTensor(y_val)
        else:
            y_train = torch.FloatTensor(y_train)
            y_val = torch.FloatTensor(y_val)
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train, 
                                                      y_train.reshape(-1, 1))
        val_dataset = torch.utils.data.TensorDataset(X_val, 
                                                    y_val.reshape(-1, 1))
        
        train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                  batch_size=batch_size, 
                                                  shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, 
                                               batch_size=batch_size)
        
        # Select optimizer based on name
        optimizer_dict = {
            'Adam': torch.optim.Adam,
            'SGD': torch.optim.SGD,
            'RMSprop': torch.optim.RMSprop,
            'AdamW': torch.optim.AdamW
        }
        optimizer_class = optimizer_dict.get(optimizer_name, torch.optim.Adam)
        optimizer = optimizer_class(model.parameters(), lr=learning_rate)
        
        best_test_loss = float('inf')
        train_losses = []
        test_losses = []
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                
                if is_classification:
                    if unique_values == 2:  # Binary classification
                        outputs = outputs.squeeze()
                        batch_y = batch_y.float().squeeze()
                    else:  # Multi-class
                        batch_y = batch_y.squeeze()
                
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            train_losses.append(epoch_loss / len(train_loader))
            
            # Validation
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    if is_classification:
                        if unique_values == 2:  # Binary classification
                            outputs = outputs.squeeze()
                            batch_y = batch_y.float().squeeze()
                        else:  # Multi-class
                            batch_y = batch_y.squeeze()
                    test_loss += criterion(outputs, batch_y).item()
            
            test_loss = test_loss / len(val_loader)
            test_losses.append(test_loss)
            best_test_loss = min(best_test_loss, test_loss)
            
            if not return_best_loss:
                st.write(f"Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_loss:.4f}")
        
        if return_best_loss:
            return best_test_loss
            
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(train_losses, label='Train Loss')
        ax1.plot(test_losses, label='Test Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.set_title('Training and Test Loss')
        
        # Convert data to tensors for predictions
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            predictions = model(X_tensor).cpu().numpy().flatten()
        
        # Predictions vs Actuals
        ax2.scatter(y, predictions)
        ax2.plot([min(y), max(y)], [min(y), max(y)], 'r--')
        ax2.set_xlabel('Actual Values')
        ax2.set_ylabel('Predictions')
        ax2.set_title('Predictions vs Actual Values')
        
        st.pyplot(fig)
        
        # Calculate metrics
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        st.write(f"Final MSE: {mse:.4f}")
        st.write(f"R² Score: {r2:.4f}")
        
        return model
        
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        if return_best_loss:
            return float('inf')
        return None

def pytorch_model_designer(data=None, target_column=None):
    if data is None:
        st.error("Please load data first")
        return
        
    data_type = st.radio("Select input data type", ["Tabular", "Image"])
    
    if data_type == "Tabular":
        input_dim = data.drop(columns=[target_column]).shape[1]
        st.write(f"Input dimension: {input_dim} (based on data)")
        
        # Determine if classification or regression based on unique values
        unique_values = len(data[target_column].unique())
        is_classification = unique_values < 10  # Heuristic for classification
        output_dim = unique_values if is_classification else 1
        
        use_optuna = st.checkbox("Use Optuna for hyperparameter optimization")
        epochs = st.number_input("Number of epochs", min_value=1, value=10)
        
        if use_optuna:
            st.subheader("Hyperparameter Optimization Settings")
            
            # Loss function selection based on task type
            if is_classification:
                if output_dim == 2:  # Binary classification
                    loss_functions = {
                        "BCEWithLogits": nn.BCEWithLogitsLoss(),
                        "BCE": nn.BCELoss()
                    }
                else:  # Multi-class classification
                    loss_functions = {
                        "CrossEntropy": nn.CrossEntropyLoss(),
                        "NLLLoss": nn.NLLLoss(),
                        "KLDiv": nn.KLDivLoss()
                    }
                st.write(f"Classification task detected ({output_dim} classes)")
            else:
                loss_functions = {
                    "MSE": nn.MSELoss(),
                    "MAE": nn.L1Loss(),
                    "Huber": nn.HuberLoss(),
                    "SmoothL1": nn.SmoothL1Loss(),
                    "PoissonNLL": nn.PoissonNLLLoss()
                }
                st.write("Regression task detected")
                
            loss_function = st.selectbox("Loss Function", list(loss_functions.keys()))
            
            # Hyperparameter ranges
            st.write("### Set Hyperparameter Ranges")
            col1, col2 = st.columns(2)
            
            with col1:
                lr_min = st.number_input("Learning Rate Min", value=1e-5, format="%.1e")
                lr_max = st.number_input("Learning Rate Max", value=1e-1, format="%.1e")
                
                batch_sizes = st.text_input("Batch Sizes (comma-separated)", 
                                          value="16,32,64,128")
                batch_size_list = [int(x.strip()) for x in batch_sizes.split(",")]
                
                min_layers = st.number_input("Min Layers", min_value=1, value=1)
                max_layers = st.number_input("Max Layers", min_value=1, value=5)
            
            with col2:
                dropout_min = st.number_input("Dropout Min", min_value=0.0, 
                                            max_value=1.0, value=0.1)
                dropout_max = st.number_input("Dropout Max", min_value=0.0, 
                                            max_value=1.0, value=0.5)
                
                hidden_dim_min = st.number_input("Hidden Dim Min", 
                                               min_value=4, value=min(input_dim, 4))
                hidden_dim_max = st.number_input("Hidden Dim Max", 
                                               min_value=4, value=max(input_dim * 2, 512))
            
            optimizers = st.multiselect("Optimizers to Try", 
                                      ['Adam', 'SGD', 'RMSprop', 'AdamW'],
                                      default=['Adam'])
            
            n_trials = st.number_input("Number of Optuna trials", min_value=1, value=20)
            study_duration = st.number_input("Study duration (minutes)", 
                                           min_value=1, value=10)

            def objective(trial):
                # Hyperparameter search space using user-defined ranges
                lr = trial.suggest_float('lr', lr_min, lr_max, log=True)
                batch_size = trial.suggest_categorical('batch_size', batch_size_list)
                n_layers = trial.suggest_int('n_layers', min_layers, max_layers)
                dropout = trial.suggest_float('dropout', dropout_min, dropout_max)
                optimizer_name = trial.suggest_categorical('optimizer', optimizers)
                
                hidden_dims = []
                for i in range(n_layers):
                    hidden_dims.append(
                        trial.suggest_int(f'hidden_dim_{i}', 
                                        hidden_dim_min, 
                                        hidden_dim_max)
                    )
                
                model = ModelFactory.get_custom_mlp(input_dim, hidden_dims, 
                                                  output_dim, dropout)
                
                try:
                    final_loss = train_pytorch_model(
                        model, data, target_column,
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=lr,
                        optimizer_name=optimizer_name,
                        criterion=loss_functions[loss_function],
                        return_best_loss=True
                    )
                    return final_loss  # Negative because Optuna minimizes
                except Exception as e:
                    return float('inf')

            if st.button("Start Optimization"):
                study = optuna.create_study(direction="minimize")
                with st.spinner('Optimizing hyperparameters...'):
                    study.optimize(objective, n_trials=n_trials, 
                                 timeout=60*study_duration)
                
                st.write("### Best Parameters Found:")
                st.write(study.best_params)
                
                # Use best parameters
                best_params = study.best_params
                hidden_dims = [best_params[f'hidden_dim_{i}'] 
                             for i in range(best_params['n_layers'])]
                model = ModelFactory.get_custom_mlp(input_dim, hidden_dims, 
                                                  output_dim, best_params['dropout'])
                
                trained_model = train_pytorch_model(
                    model, data, target_column,
                    epochs=epochs,
                    batch_size=best_params['batch_size'],
                    learning_rate=best_params['lr'],
                    optimizer_name=best_params['optimizer'],
                    criterion=loss_functions[loss_function]
                )

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
