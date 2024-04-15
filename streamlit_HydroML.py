import matplotlib.pyplot as plt
import os
import numpy as np
import streamlit as st
import pandas as pd
from ml_models import ModelSelector
import re

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.cluster import affinity_propagation, cluster_optics_dbscan, cluster_optics_xi, dbscan, estimate_bandwidth, k_means, kmeans_plusplus, mean_shift, spectral_clustering, ward_tree

from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline, make_union
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer, load_digits, load_files, load_linnerud, load_sample_images, load_wine, fetch_california_housing, fetch_covtype, fetch_kddcup99, fetch_rcv1, fetch_olivetti_faces
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, jaccard_score, roc_auc_score,average_precision_score
from sklearn.metrics import max_error, explained_variance_score, mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, ConfusionMatrixDisplay
from sklearn.preprocessing import Binarizer, FunctionTransformer, KBinsDiscretizer, KernelCenterer, LabelBinarizer, LabelEncoder, MaxAbsScaler, MinMaxScaler, Normalizer, OneHotEncoder, OrdinalEncoder, PolynomialFeatures, PowerTransformer, QuantileTransformer, SplineTransformer, StandardScaler, binarize, label_binarize, maxabs_scale, minmax_scale, normalize

def clean_feature_name(name):
    # Remove invalid characters and replace spaces with underscores
    name = re.sub(r'[^a-zA-Z0-9_]', '', name)
    name = re.sub(r'\s+', '_', name)
    return name

def drop_non_numeric_data(data):
    numeric_columns = data.select_dtypes(include=['number']).columns
    non_numeric_columns = data.columns.difference(numeric_columns)

    if len(non_numeric_columns) > 0:
        print("Non-numeric columns found:")
        print(non_numeric_columns)

        drop_rows = tk.messagebox.askyesno("Non-numeric Data", "The dataset contains non-numeric data. Do you want to drop rows with non-numeric values?")
        if drop_rows:
            data = data[numeric_columns]
        else:
            drop_columns = tk.messagebox.askyesno("Non-numeric Data", "Do you want to drop non-numeric columns instead?")
            if drop_columns:
                data = data.select_dtypes(include=['number'])
            else:
                # Convert non-numeric columns to numeric if possible
                for column in non_numeric_columns:
                    try:
                        data[column] = pd.to_numeric(data[column], errors='coerce')
                    except ValueError:
                        pass

                # Drop any remaining non-numeric columns
                data = data.select_dtypes(include=['number'])

    return data

def train_model(data, model_name, target_column, parameters, model_selector):
    X = data.drop(columns=[target_column])
    y = data[target_column]

    X.columns = [clean_feature_name(col) for col in X.columns]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_class = model_selector.get_model(model_name)
    model = model_class(**parameters)
    pipeline = make_pipeline(StandardScaler(), model)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = pipeline.score(X_test, y_test)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred)
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predictions")
    ax.set_title(f"{model_name} - Predictions vs True Values - Accuracy: {accuracy}")
    st.pyplot(fig)

    st.write(f"Model: {model_name}")
    st.write(f"Accuracy: {accuracy:.2f}")

# Streamlit app
def main():
    st.title("Machine Learning Model Trainer")

    # File upload
    file = st.file_uploader("Upload CSV file", type=["csv"])

    if file is not None:
        try:
            data = pd.read_csv(file, encoding='latin1', low_memory=False)
        except UnicodeDecodeError:
            data = pd.read_csv(file, encoding='utf-16', low_memory=False)
        data = drop_non_numeric_data(data)

    # Toy dataset selection
    toy_dataset = st.sidebar.selectbox("Select a toy dataset", ("None", "Diabetes (Regression)", "Iris (Classification)"))

    if toy_dataset == "Diabetes (Regression)":
        diabetes = load_diabetes()
        data = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
        data['target'] = diabetes.target
    elif toy_dataset == "Iris (Classification)":
        iris = load_iris()
        data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        data['target'] = iris.target

    if 'data' in locals():
        # Target column selection
        target_column = st.sidebar.selectbox("Select the target column", data.columns)

        # Model selection
        model_selector = ModelSelector()
        model_name = st.sidebar.selectbox("Select a model", list(model_selector.regression_models.keys()) + list(model_selector.classification_models.keys()))

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

        # Train model button
        if st.button("Train Model"):
            train_model(data, model_name, target_column, parameters, model_selector)

if __name__ == '__main__':
    main()