import matplotlib.pyplot as plt
import os
import numpy as np
import streamlit as st

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, RidgeCV, Lasso, LassoLars, BayesianRidge, TweedieRegressor, SGDRegressor, SGDClassifier, Perceptron, TheilSenRegressor, HuberRegressor, ElasticNet, OrthogonalMatchingPursuit
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, DBSCAN, BisectingKMeans, MiniBatchKMeans, SpectralClustering
from sklearn.cluster import affinity_propagation, cluster_optics_dbscan, cluster_optics_xi, dbscan, estimate_bandwidth, k_means, kmeans_plusplus, mean_shift, spectral_clustering, ward_tree
from sklearn.compose import ColumnTransformer
from sklearn.covariance import EmpiricalCovariance, GraphicalLasso, GraphicalLassoCV, MinCovDet, OAS, empirical_covariance, graphical_lasso, ledoit_wolf, oas
from sklearn.cross_decomposition import CCA, PLSCanonical, PLSRegression, PLSSVD
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.decomposition import FactorAnalysis, FastICA, IncrementalPCA, MiniBatchSparsePCA, PCA
from xgboost import XGBRegressor as xgbr
from xgboost import DMatrix
import re
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.neural_network import BernoulliRBM, MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline, make_union
from sklearn.ensemble import BaggingClassifier, BaggingRegressor, ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier, StackingClassifier, StackingRegressor, VotingClassifier, VotingRegressor, HistGradientBoostingRegressor,HistGradientBoostingClassifier
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer, load_digits, load_files, load_linnerud, load_sample_images, load_wine, fetch_california_housing, fetch_covtype, fetch_kddcup99, fetch_rcv1, fetch_olivetti_faces
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, jaccard_score, roc_auc_score,average_precision_score
from sklearn.metrics import max_error, explained_variance_score, mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, ConfusionMatrixDisplay
from sklearn.preprocessing import Binarizer, FunctionTransformer, KBinsDiscretizer, KernelCenterer, LabelBinarizer, LabelEncoder, MaxAbsScaler, MinMaxScaler, Normalizer, OneHotEncoder, OrdinalEncoder, PolynomialFeatures, PowerTransformer, QuantileTransformer, SplineTransformer, StandardScaler, binarize, label_binarize, maxabs_scale, minmax_scale, normalize
class ModelSelector:
    def __init__(self):
        self.regression_models = {
        "Ridge": Ridge, 
        "RidgeCV": RidgeCV, 
        "Lasso": Lasso, 
        "LassoLars": LassoLars,
        "BayesianRidge": BayesianRidge, 
        "TweedieRegressor": TweedieRegressor, 
        "SGDRegressor": SGDRegressor, 
        "SGDClassifier": SGDClassifier, 
        "Perceptron": Perceptron, 
        "TheilSenRegressor": TheilSenRegressor,
        "HuberRegressor": HuberRegressor,
        "ElasticNet": ElasticNet, 
        "OrthogonalMatchingPursuit": OrthogonalMatchingPursuit,
            "RandomForestRegressor": RandomForestRegressor,
            "LinearRegression": LinearRegression,
            "SVR": SVR,
            "KNeighborsRegressor": KNeighborsRegressor,
            "XGBRegressor": xgbr
        }
        self.classification_models = {
            "RandomForestClassifier": RandomForestClassifier,
            "LogisticRegression": LogisticRegression,
            "SVC": SVC,
            "KNeighborsClassifier": KNeighborsClassifier
        }
        self.model_parameters = {
        "RandomForestRegressor": {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1
        },
        "Ridge": {
            "alpha": 1.0,
            "fit_intercept": True,
            "copy_X": True,
            "max_iter": 1000,
            "tol": 0.0001,
            "solver": "auto",
            "random_state": None
        },
        "RidgeCV": {
            "alphas": (0.1, 1.0, 10.0),
            "fit_intercept": True,
            "scoring": None,
            "cv": None,
            "gcv_mode": None,
            "store_cv_values": False
        },
        "Lasso": {
            "alpha": 1.0,
            "fit_intercept": True,
            "precompute": False,
            "copy_X": True,
            "max_iter": 1000,
            "tol": 0.0001,
            "warm_start": False,
            "positive": False,
            "random_state": None,
            "selection": "cyclic"
        },
        "LassoLars": {
            "alpha": 1.0,
            "fit_intercept": True,
            "verbose": False,
            "precompute": "auto",
            "max_iter": 500,
            "eps": 2.220446049250313e-16,
            "copy_X": True,
            "fit_path": True,
            "positive": False,
            "jitter": None,
            "random_state": None
        },
        "BayesianRidge": {
            "n_iter": 300,
            "tol": 0.001,
            "alpha_1": 1e-06,
            "alpha_2": 1e-06,
            "lambda_1": 1e-06,
            "lambda_2": 1e-06,
            "compute_score": False,
            "fit_intercept": True,
            "copy_X": True,
            "verbose": False
        },
        "TweedieRegressor": {
            "power": 0.0,
            "alpha": 1.0,
            "fit_intercept": True,
            "link": "auto",
            "max_iter": 100,
            "tol": 0.0001,
            "warm_start": False,
            "verbose": False
        },
        "SGDRegressor": {
            "loss": "squared_error",
            "penalty": "l2",
            "alpha": 0.0001,
            "l1_ratio": 0.15,
            "fit_intercept": True,
            "max_iter": 1000,
            "tol": 0.001,
            "shuffle": True,
            "verbose": 0,
            "epsilon": 0.1,
            "random_state": None,
            "learning_rate": "invscaling",
            "eta0": 0.01,
            "power_t": 0.25,
            "early_stopping": False,
            "validation_fraction": 0.1,
            "n_iter_no_change": 5,
            "warm_start": False,
            "average": False
        },
        "SGDClassifier": {
            "loss": "hinge",
            "penalty": "l2",
            "alpha": 0.0001,
            "l1_ratio": 0.15,
            "fit_intercept": True,
            "max_iter": 1000,
            "tol": 0.001,
            "shuffle": True,
            "verbose": 0,
            "epsilon": 0.1,
            "n_jobs": None,
            "random_state": None,
            "learning_rate": "optimal",
            "eta0": 0.0,
            "power_t": 0.5,
            "early_stopping": False,
            "validation_fraction": 0.1,
            "n_iter_no_change": 5,
            "class_weight": None,
            "warm_start": False,
            "average": False
        },
        "Perceptron": {
            "penalty": None,
            "alpha": 0.0001,
            "fit_intercept": True,
            "max_iter": 1000,
            "tol": 0.001,
            "shuffle": True,
            "verbose": 0,
            "eta0": 1.0,
            "n_jobs": None,
            "random_state": 0,
            "early_stopping": False,
            "validation_fraction": 0.1,
            "n_iter_no_change": 5,
            "class_weight": None,
            "warm_start": False
        },
        "TheilSenRegressor": {
            "fit_intercept": True,
            "copy_X": True,
            "max_subpopulation": 10000,
            "n_subsamples": None,
            "max_iter": 300,
            "tol": 0.001,
            "random_state": None,
            "n_jobs": 2,
            "verbose": False
        },
        "HuberRegressor": {
            "epsilon": 1.35,
            "max_iter": 100,
            "alpha": 0.0001,
            "warm_start": False,
            "fit_intercept": True,
            "tol": 1e-05
        },
        "ElasticNet": {
            "alpha": 1.0,
            "l1_ratio": 0.5,
            "fit_intercept": True,
            "precompute": False,
            "max_iter": 1000,
            "copy_X": True,
            "tol": 0.0001,
            "warm_start": False,
            "positive": False,
            "random_state": None,
            "selection": "cyclic"
        },
        "OrthogonalMatchingPursuit": {
            "n_nonzero_coefs": None,
            "tol": None,
            "fit_intercept": True,
            "precompute": "auto"
        },
            "SVR": {
                "kernel": "rbf",
                "C": 1.0,
                "epsilon": 0.1
            },
            "KNeighborsRegressor": {
                "n_neighbors": 5,
                "weights": "uniform",
                "algorithm": "auto"
            },
            "XGBRegressor": {
                "n_estimators": 100,
                "eta": 0.3,
                "gamma": 0.0,
                "lambda": 1.0,
                "alpha": 0.0,
                "min_child_weight" : 1.0,
                "learning_rate": 0.1,
                "max_depth": 3,
                "subsample": 1.0,
                "colsample_bytree": 1.0
            },
            "RandomForestClassifier": {
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1
            },
            "LogisticRegression": {
                "penalty": "l2",
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 100
            },
            "SVC": {
                "C": 1.0,
                "kernel": "rbf",
                "degree": 3,
                "gamma": "scale"
            },
            "KNeighborsClassifier": {
                "n_neighbors": 5,
                "weights": "uniform",
                "algorithm": "auto"
            }
        }

    def get_model(self, model_name):
        if model_name in self.regression_models:
            return self.regression_models[model_name]
        elif model_name in self.classification_models:
            return self.classification_models[model_name]
        else:
            raise ValueError(f"Model '{model_name}' not found.")

    def get_model_parameters(self, model_name):
        if model_name in self.model_parameters:
            return self.model_parameters[model_name]
        else:
            raise ValueError(f"Parameters for model '{model_name}' not found.")

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
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)

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