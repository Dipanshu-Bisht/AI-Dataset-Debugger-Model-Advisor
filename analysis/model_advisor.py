import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier



def detect_problem_type(df, target_column):

    target = df[target_column]

    unique_vals = target.nunique()
    total_vals = len(target)

    # categorical targets
    if target.dtype == "object":
        return "classification"

    # numeric but discrete
    if unique_vals <= 10 and unique_vals / total_vals < 0.1:
        return "classification"

    return "regression"

from sklearn.preprocessing import LabelEncoder

def prepare_features(df, target_column):

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Remove ID-like columns
    for col in X.columns:
        if X[col].nunique() == len(X):
            X = X.drop(columns=[col])

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns

    for col in categorical_cols:

        unique_vals = X[col].nunique()

        if unique_vals <= 10:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        elif unique_vals <= 50:
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
            X = pd.concat([X.drop(columns=[col]), dummies], axis=1)

        else:
            X = X.drop(columns=[col])

    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    if X.shape[1] > 120:
        X = X.iloc[:, :120]

    return X, y

def train_baseline_model(X, y):

    # Speed optimization
    if len(X) > 7000:
        X = X.sample(7000, random_state=42)
        y = y.loc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Random Forest": RandomForestClassifier(n_estimators=30),
        "Decision Tree": DecisionTreeClassifier()
    }

    results = {}

    best_model = None
    best_accuracy = 0
    best_name = ""

    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        results[name] = accuracy

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_name = name

    importance = None

    if hasattr(best_model, "feature_importances_"):
        importance = best_model.feature_importances_

    return best_name, best_model, best_accuracy, importance, results

