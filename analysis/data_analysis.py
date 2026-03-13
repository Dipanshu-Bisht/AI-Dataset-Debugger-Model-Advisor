import pandas as pd
import streamlit as st


@st.cache_data
def dataset_summary(df):
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns)
    }


@st.cache_data
def missing_values(df):
    return df.isnull().sum()


@st.cache_data
def correlation_matrix(df):

    df_copy = df.copy()

    for col in df_copy.select_dtypes(include=["object", "category"]).columns:

        if df_copy[col].nunique() <= 10:
            df_copy[col] = df_copy[col].astype("category").cat.codes

    numeric_df = df_copy.select_dtypes(include=["number"])

    if numeric_df.shape[1] < 2:
        return None

    corr = numeric_df.corr()

    return corr

def dataset_health_score(df):

    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()

    missing_ratio = missing_cells / total_cells

    score = max(0, 100 - (missing_ratio * 100))

    return round(score, 2)