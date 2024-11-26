import os
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from utils.model_utils import save_model

def train_model():
    print("\n--- Model Training ---")

    # Step 1: Check if preprocessed data exists
    try:
        X_train = pd.read_csv("data/X_train.csv")
        y_train = pd.read_csv("data/y_train.csv")
        X_test = pd.read_csv("data/X_test.csv")
        y_test = pd.read_csv("data/y_test.csv")
        print("Preprocessed data loaded successfully!")
    except FileNotFoundError:
        print("Error: Preprocessed data not found. Please run the 'Preprocess Data' step first.")
        return

    # Step 2: Select task type
    task_type = input("Select task type: (1) Classification, (2) Regression: ").strip()
    if task_type not in ["1", "2"]:
        print("Invalid task type. Please select 1 or 2.")
        return

    if task_type == "1":
        # Classification models
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        models = {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42)
        }
        metric_name = "Accuracy"
    elif task_type == "2":
        # Regression models
        models = {
            "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
            "Linear Regression": LinearRegression()
        }
        metric_name = "R2 Score"

    # Step 3: Train and evaluate each model
    best_model = None
    best_metric = float("-inf")
    best_model_name = ""

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train.values.ravel())
        print(f"{model_name} training complete!")

        # Evaluate the model
        predictions = model.predict(X_test)
        if task_type == "1":
            # Classification metrics
            from sklearn.metrics import accuracy_score
            metric = accuracy_score(y_test, predictions)
        elif task_type == "2":
            # Regression metrics
            metric = r2_score(y_test, predictions)

        print(f"{model_name} {metric_name}: {metric:.4f}")

        # Update the best model
        if metric > best_metric:
            best_model = model
            best_metric = metric
            best_model_name = model_name

    # Step 4: Save the best model
    if best_model:
        model_path = f"models/{best_model_name.replace(' ', '_').lower()}_model.pkl"
        save_model(best_model, model_path)
        print(f"\nBest Model: {best_model_name} with {metric_name}: {best_metric:.4f}")
        print(f"Model saved as '{model_path}'.")

    print("\nTraining complete!\n")
