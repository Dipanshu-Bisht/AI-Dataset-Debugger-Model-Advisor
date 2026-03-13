import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def app_title():
    st.title("AI Dataset Debugger & ML Advisor")
    st.write(
        """
        Upload a dataset to automatically analyze data quality,
        detect the machine learning problem type,
        and train baseline models.
        """
    )


def upload_dataset():

    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        # Replace hidden missing values
        df = df.replace(
            [" ", "", "NA", "N/A", "na", "null", "NULL", "?", "-"],
            pd.NA
        )

        # Convert numeric-looking columns automatically
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass

        return df

    return None


def show_dataset_preview(df):
    st.subheader("Dataset Preview")
    st.dataframe(df.head())


def show_dataset_summary(summary):

    st.subheader("Dataset Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", summary["rows"])

    with col2:
        st.metric("Columns", summary["columns"])


def show_missing_values(missing_values):

    st.subheader("Missing Values")

    missing_df = missing_values[missing_values > 0]

    if missing_df.empty:
        st.write("No missing values detected.")
    else:
        st.write(missing_df)


def show_correlation_heatmap(corr_matrix):

    st.subheader("Correlation Heatmap")

    if corr_matrix is None or corr_matrix.empty:
        st.info("Not enough numeric features to compute correlation.")
        return

    corr_matrix = corr_matrix.dropna(axis=0, how="all").dropna(axis=1, how="all")

    if corr_matrix.empty:
        st.info("Correlation could not be computed due to insufficient numeric data.")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        ax=ax
    )

    st.pyplot(fig)

def show_model_results(problem_type, accuracy):

    st.subheader("Model Results")

    st.write(f"Detected Problem Type: **{problem_type}**")

    st.metric(
        label="Best Model Accuracy",
        value=f"{accuracy:.2f}"
    )