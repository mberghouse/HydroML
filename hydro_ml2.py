import matplotlib.pyplot as plt
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, RidgeCV, Lasso, LassoLars, BayesianRidge, TweedieRegressor, SGDRegressor, SGDClassifier, Perceptron, TheilSenRegressor, HuberRegressor, ElasticNet, OrthogonalMatchingPursuit
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, DBSCAN, BisectingKMeans, MiniBatchKMeans, SpectralClustering, SpectralBiClustering
from sklearn.cluster import affinity_propagation, cluster_optics_dbscan, cluster_optics_xi, computer_optics_graph, dbscan, estimate_bandwidth, k_means, kmeans_plusplus, mean_shift, spectral_clustering, ward_tree
from sklearn.compose import ColumnTransformer
from sklearn.covariance import EmpiricalCovariance, GraphicalLasso, GraphicalLassoCV, MinCovDet, OAS, empirical_covariance, graphical_lasso, ledoit_wolf, oas
from sklearn.cross_decomposition import CCA, PLSCanonical, PLSRegression, PLSSVD
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBRegressor as xgbr
from xgboost import DMatrix
import re
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer, load_digits, load_files, load_linnerud, load_sample_images, load_wine, fetch_california_housing, fetch_covtype, fetch_kddcup99, fetch_rcv1, fetch_olivetti_faces
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, jaccard_score, roc_auc_score,average_precision_score
from sklearn.metrics import max_error, explained_variance_score, mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, ConfusionMatrixDisplay,
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
            "max_iter": None,
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
            "loss": "squared_loss",
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
            "n_jobs": None,
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
                "learning_rate": 0.1,
                "max_depth": 3,
                "subsample": 1,
                "colsample_bytree": 1
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

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    file_entry.delete(0, tk.END)
    file_entry.insert(tk.END, file_path)

def select_toy_dataset():
    global use_toy_dataset
    use_toy_dataset = True
    toy_dataset_type_var.set(False)
    toy_dataset_type_frame.pack(side=tk.LEFT)
    browse_button.config(text="Back", command=go_back)

def go_back():
    global use_toy_dataset
    use_toy_dataset = False
    toy_dataset_type_frame.pack_forget()
    browse_button.config(text="Browse", command=browse_file)
    file_label.config(text="Select CSV file:")
    file_entry.delete(0, tk.END)

def update_file_entry():
    toy_dataset_type = toy_dataset_type_var.get()
    if toy_dataset_type == "regression":
        file_entry.delete(0, tk.END)
        file_entry.insert(tk.END, "Diabetes Dataset")
    elif toy_dataset_type == "classification":
        file_entry.delete(0, tk.END)
        file_entry.insert(tk.END, "Iris Dataset")

def select_target_column():
    global data

    if use_toy_dataset:
        toy_dataset_type = toy_dataset_type_var.get()
        if toy_dataset_type == "regression":
            # Load the diabetes dataset for regression
            diabetes = load_diabetes()
            data = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
            data['target'] = diabetes.target
        elif toy_dataset_type == "classification":
            # Load the iris dataset for classification
            iris = load_iris()
            data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
            data['target'] = iris.target
    else:
        file_path = file_entry.get()
        try:
            data = pd.read_csv(file_path, encoding='latin1', low_memory=False)
        except UnicodeDecodeError:
            data = pd.read_csv(file_path, encoding='utf-16', low_memory=False)

    try:
        data = drop_non_numeric_data(data)
    except ValueError as e:
        tk.messagebox.showerror("Error", str(e))
        return

    column_window = tk.Toplevel(window)
    column_window.title("Select Target Column")

    column_label = tk.Label(column_window, text="Select the target column:")
    column_label.pack()

    column_var = tk.StringVar(column_window)
    column_var.set(data.columns[0])

    column_dropdown = tk.OptionMenu(column_window, column_var, *data.columns)
    column_dropdown.pack()

    def set_target_column():
        target_column = column_var.get()
        target_entry.delete(0, tk.END)
        target_entry.insert(tk.END, target_column)
        column_window.destroy()

    ok_button = tk.Button(column_window, text="OK", command=set_target_column)
    ok_button.pack()

def select_model():
    model_name = model_var.get()
    model_entry.delete(0, tk.END)
    model_entry.insert(tk.END, model_name)
    load_model_parameters()

def load_model_parameters():
    model_name = model_entry.get()
    parameters = model_selector.get_model_parameters(model_name)

    param_window = tk.Toplevel(window)
    param_window.title("Model Parameters")

    param_frame = ttk.Frame(param_window)
    param_frame.pack(padx=10, pady=10)

    param_entries = {}
    for param, value in parameters.items():
        param_label = ttk.Label(param_frame, text=param)
        param_label.pack()
        param_entry = ttk.Entry(param_frame)
        param_entry.insert(tk.END, str(value))
        param_entry.pack()
        param_entries[param] = param_entry

    def update_parameters():
        for param, entry in param_entries.items():
            value = entry.get()
            try:
                parameters[param] = eval(value)
            except (NameError, SyntaxError):
                parameters[param] = value
        param_window.destroy()

    update_button = ttk.Button(param_window, text="Update Parameters", command=update_parameters)
    update_button.pack()

def train_model(data):
    model_name = model_entry.get()
    target_column = target_entry.get()

    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Clean feature names
    X.columns = [clean_feature_name(col) for col in X.columns]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_class = model_selector.get_model(model_name)
    parameters = model_selector.get_model_parameters(model_name)

    # if model_name == "XGBRegressor":
        # dtrain = DMatrix(X_train, label=y_train, feature_names=list(X_train.columns))
        # dtest = DMatrix(X_test, label=y_test, feature_names=list(X_test.columns))
        # model = model_class(**parameters)
        # model.fit(dtrain)
        # y_pred = model.predict(dtest)
        # accuracy = model.score(dtest)
    # else:
    model = model_class(**parameters)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)

    print('Data shapes: ', X_train.shape, y_train.shape)
    print('Target data type: ', y_train.dtype)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title(f"{model_name} - Predictions vs True Values - Accuracy: {accuracy}")
    plt.show()

    result_label.config(text=f"Model: {model_name}\nAccuracy: {accuracy:.2f}")

# Create the main window
window = tk.Tk()
window.title("Machine Learning Model Trainer")
window.geometry("800x600")  # Set the window size

# Set the font size
font_size = 14
window.option_add("*Font", f"Arial {font_size}")

# Create and pack the widgets
file_label = tk.Label(window, text="Select CSV file:")
file_label.pack(pady=10)

file_entry = tk.Entry(window, width=50)
file_entry.pack()

browse_button = tk.Button(window, text="Browse", command=browse_file)
browse_button.pack(pady=10)

toy_dataset_frame = tk.Frame(window)
toy_dataset_frame.pack(pady=10)

toy_dataset_button = tk.Button(toy_dataset_frame, text="Toy Dataset", command=select_toy_dataset)
toy_dataset_button.pack(side=tk.LEFT, padx=5)

toy_dataset_type_frame = tk.Frame(toy_dataset_frame)
toy_dataset_type_frame.pack(side=tk.LEFT)

toy_dataset_type_var = tk.StringVar()
regression_button = tk.Radiobutton(toy_dataset_type_frame, text="Regression", variable=toy_dataset_type_var, value="regression", command=update_file_entry)
regression_button.pack(side=tk.LEFT, padx=5)
classification_button = tk.Radiobutton(toy_dataset_type_frame, text="Classification", variable=toy_dataset_type_var, value="classification", command=update_file_entry)
classification_button.pack(side=tk.LEFT, padx=5)

target_label = tk.Label(window, text="Target column:")
target_label.pack(pady=10)

target_entry = tk.Entry(window, width=30)
target_entry.pack()

select_target_button = tk.Button(window, text="Select Target Column", command=select_target_column)
select_target_button.pack(pady=10)

model_label = tk.Label(window, text="Select a model:")
model_label.pack(pady=10)

model_var = tk.StringVar()
model_selector = ModelSelector()
model_dropdown = tk.OptionMenu(window, model_var, *list(model_selector.regression_models.keys()) + list(model_selector.classification_models.keys()))
model_dropdown.pack()

select_model_button = tk.Button(window, text="Select Model Parameters", command=select_model)
select_model_button.pack(pady=10)

model_entry = tk.Entry(window, width=30)
model_entry.pack()

train_button = tk.Button(window, text="Train Model", command=lambda: train_model(data))
train_button.pack(pady=10)

result_label = tk.Label(window, text="")
result_label.pack()

# Initialize variables
use_toy_dataset = False
data = None

# Start the main event loop
window.mainloop()