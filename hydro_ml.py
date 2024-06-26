import matplotlib.pyplot as plt
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error as MSE
from xgboost import XGBRegressor as xgbr
import re
xgb=0

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

def select_target_column():
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

    #data = drop_non_numeric_data(data)
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

from xgboost import DMatrix

from xgboost import DMatrix

def train_model():
    file_path = file_entry.get()
    model_name = model_entry.get()
    target_column = target_entry.get()

    try:
        data = pd.read_csv(file_path, encoding='latin1')
    except UnicodeDecodeError:
        data = pd.read_csv(file_path, encoding='utf-16')
    try:
        data = drop_non_numeric_data(data)
    except ValueError as e:
        tk.messagebox.showerror("Error", str(e))
        return

    if data.isnull().values.any():
        fill_na = tk.messagebox.askyesno("Missing Values", "The dataset contains missing values. Do you want to fill them?")
        if fill_na:
            data.fillna(0, inplace=True)
            #imputer = SimpleImputer(strategy='mean')
            #data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        else:
            model_name = "XGBClassifier"

    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Preprocess feature names
    X.columns = [clean_feature_name(col) for col in X.columns]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if y.dtype == 'int':
        print('target contains classification labels. Please use a classifier')
    if model_name.lower() == "randomforest":
        model = RandomForestRegressor()
    elif model_name.lower() == "logisticregression":
        model = LogisticRegression()
    elif model_name.lower() == "svm":
        model = SVC()
    elif model_name.lower() == "knn":
        model = KNeighborsClassifier()
    else:
        # Convert data to DMatrix format
        dtrain = DMatrix(X_train, label=y_train, feature_names=list(X_train.columns))
        dtest = DMatrix(X_test, label=y_test, feature_names=list(X_test.columns))

        # Set XGBoost parameters
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'seed': 42
        }

        # Train XGBoost model
        model = xgbr(**params)
        xgb=1
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    plt.plot(y_pred,y_test)
    if xgb==1:
        accuracy = MSE(y_test, y_pred, squared=False)
    else:
        accuracy = model.score(X_test, y_test)

    print('Data shapes: ', X_train.shape, y_train.shape)
    print('Target data type: ', y_train.dtype)

    plt.plot(y_pred, y_test)

    result_label.config(text=f"Model: {model_name}\nAccuracy: {accuracy:.2f}")
# Create the main window
window = tk.Tk()
window.title("Machine Learning Model Trainer")

# Create and pack the widgets
file_label = tk.Label(window, text="Select CSV file:")
file_label.pack()

file_entry = tk.Entry(window, width=50)
file_entry.pack()

browse_button = tk.Button(window, text="Browse", command=browse_file)
browse_button.pack()

target_label = tk.Label(window, text="Target column:")
target_label.pack()

target_entry = tk.Entry(window, width=30)
target_entry.pack()

select_target_button = tk.Button(window, text="Select Target Column", command=select_target_column)
select_target_button.pack()

model_label = tk.Label(window, text="Enter the model name (e.g., RandomForest, LogisticRegression, SVM, KNN):")
model_label.pack()

model_entry = tk.Entry(window, width=30)
model_entry.pack()

train_button = tk.Button(window, text="Train Model", command=train_model)
train_button.pack()

result_label = tk.Label(window, text="")
result_label.pack()

# Start the main event loop
window.mainloop()