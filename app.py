import streamlit as st
import pandas as pd

from analysis.data_analysis import (
    dataset_summary,
    missing_values,
    correlation_matrix,
    dataset_health_score
)

from analysis.model_advisor import (
    detect_problem_type,
    prepare_features,
    train_baseline_model
)

from ui.ui_components import (
    app_title,
    upload_dataset,
    show_dataset_preview,
    show_dataset_summary,
    show_missing_values,
    show_model_results,
    show_correlation_heatmap
)

from insights.insight_generator import generate_dataset_insights


st.set_page_config(
    page_title="AI Dataset Debugger",
    layout="wide"
)


def main():

    app_title()

    st.sidebar.header("Controls")

    df = upload_dataset()

    if df is None:
        st.info("Upload a dataset to begin analysis.")
        return

    # Dataset Preview
    show_dataset_preview(df)

    # Dataset Summary
    summary = dataset_summary(df)
    show_dataset_summary(summary)

    # Dataset Health Score
    score = dataset_health_score(df)

    st.subheader("Dataset Health Score")
    st.metric(label="Health Score", value=f"{score}/100")

    # Missing Values
    missing = missing_values(df)
    show_missing_values(missing)

    # Correlation Heatmap
    corr = correlation_matrix(df)
    show_correlation_heatmap(corr)

    # Sidebar Model Configuration
    st.sidebar.subheader("Model Configuration")

    target_column = st.sidebar.selectbox(
        "Select Target Column",
        df.columns
    )

    problem_type = detect_problem_type(df, target_column)

    X, y = prepare_features(df, target_column)

    st.sidebar.write("Configure settings and run ML analysis.")

    # Run Model
    if st.sidebar.button("Run ML Analysis"):

        with st.spinner("Training baseline models... Please wait"):

            best_model_name, model, accuracy, importance, results = train_baseline_model(X, y)

        # Save results to session
        st.session_state["model_trained"] = True
        st.session_state["best_model_name"] = best_model_name
        st.session_state["accuracy"] = accuracy
        st.session_state["importance"] = importance
        st.session_state["results"] = results
        st.session_state["X_columns"] = X.columns

    # Show results after training
    if "model_trained" in st.session_state:

        best_model_name = st.session_state["best_model_name"]
        accuracy = st.session_state["accuracy"]
        importance = st.session_state["importance"]
        results = st.session_state["results"]
        columns = st.session_state["X_columns"]

        st.subheader("Baseline Model Comparison")

        comparison_df = pd.DataFrame({
            "Model": results.keys(),
            "Accuracy": results.values()
        })

        st.dataframe(comparison_df)

        st.subheader("Best Baseline Model")

        st.write(f"Best Model: **{best_model_name}**")
        st.write(f"Accuracy: **{accuracy:.2f}**")

        show_model_results(problem_type, accuracy)

        importance_df = None

        if importance is not None:

            importance_df = pd.DataFrame({
                "feature": columns,
                "importance": importance
            }).sort_values(by="importance", ascending=False).head(10)

            st.subheader("Top Feature Importance")

            st.bar_chart(
                importance_df.set_index("feature")
            )

        # Insights panel
        with st.expander(" AI Dataset Insights"):

            insights = generate_dataset_insights(
                df,
                target_column,
                best_model_name,
                accuracy,
                importance_df,
                results
        )

            st.subheader("AI Dataset Insights")

            for insight in insights:
                st.write(f"• {insight}")

if __name__ == "__main__":
    main()