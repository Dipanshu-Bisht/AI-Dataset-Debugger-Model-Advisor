def generate_dataset_insights(df, target_column, best_model_name, accuracy, importance_df, results):

    insights = []

    rows, cols = df.shape
    insights.append(f"Dataset contains {rows} rows and {cols} features.")

    # Missing values
    missing = df.isna().sum()
    missing_cols = missing[missing > 0]

    if len(missing_cols) > 0:
        for col, val in missing_cols.items():
            insights.append(f"{val} missing values detected in column '{col}'.")
    else:
        insights.append("No missing values were detected in the dataset.")

    # Class imbalance
    target_counts = df[target_column].value_counts(normalize=True)

    if len(target_counts) == 2:
        perc = (target_counts * 100).round(1)

        insights.append(
            f'Target variable "{target_column}" shows moderate class imbalance '
            f'({perc.iloc[0]}% vs {perc.iloc[1]}%).'
        )

    # Model comparison
    if len(results) > 1:

        sorted_models = sorted(results.items(), key=lambda x: x[1], reverse=True)

        best = sorted_models[0]
        second = sorted_models[1]

        diff = (best[1] - second[1]) * 100

        insights.append(
            f"{best[0]} outperformed {second[0]} by {diff:.1f}%."
        )

    # Feature importance
    if importance_df is not None:

        top_features = (
            importance_df
            .sort_values(by="importance", ascending=False)
            .head(3)["feature"]
            .tolist()
        )

        insights.append(
            "Top predictive features: " + ", ".join(top_features) + "."
        )

    insights.append(
        "Baseline model performance indicates room for improvement through feature engineering or hyperparameter tuning."
    )

    return insights