import re
import pandas as pd
import streamlit as st

def clean_feature_name(name):
    name = re.sub(r'[^a-zA-Z0-9_]', '', name)
    name = re.sub(r'\s+', '_', name)
    return name

def drop_non_numeric_data(data):
    numeric_columns = data.select_dtypes(include=['number']).columns
    non_numeric_columns = data.columns.difference(numeric_columns)

    if len(non_numeric_columns) > 0:
        st.write("Non-numeric columns found:")
        st.write(non_numeric_columns)

        drop_rows = st.radio("How to handle non-numeric data?", 
                           ["Drop rows", "Drop columns", "Convert to numeric"])
        
        if drop_rows == "Drop rows":
            data = data[numeric_columns]
        elif drop_rows == "Drop columns":
            data = data.select_dtypes(include=['number'])
        else:
            for column in non_numeric_columns:
                try:
                    data[column] = pd.to_numeric(data[column], errors='coerce')
                except ValueError:
                    pass
            data = data.select_dtypes(include=['number'])

    return data 

