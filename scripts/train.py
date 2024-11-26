import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score
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
        models = {
            "Decision Tree": DecisionTreeClassifier,
            "Random Forest": RandomForestClassifier
        }
        metric_name = "Accuracy"
        metric_function = accuracy_score
    elif task_type == "2":
        # Regression models
        models = {
            "Decision Tree Regressor": DecisionTreeRegressor,
            "Random Forest Regressor": RandomForestRegressor
        }
        metric_name = "R2 Score"
        metric_function = r2_score

    # Step 3: Choose model and hyperparameters
    print("\nAvailable Models:")
    for i, model_name in enumerate(models.keys(), 1):
        print(f"{i}. {model_name}")
    model_choice = input("Select a model: ").strip()
    model_name = list(models.keys())[int(model_choice) - 1]
    model_class = models[model_name]

    # Prompt for hyperparameters
    print(f"\nCustomizing hyperparameters for {model_name}...")
    if "Decision Tree" in model_name:
        max_depth = int(input("Enter max_depth (or -1 for no limit): ").strip())
        model = model_class(max_depth=max_depth if max_depth > 0 else None)
    elif "Random Forest" in model_name:
        n_estimators = int(input("Enter n_estimators (default 100): ").strip())
        model = model_class(n_estimators=n_estimators)
    else:
        model = model_class()

    # Step 4: Train and evaluate the model
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train.values.ravel())
    print(f"{model_name} training complete!")

    # Evaluate the model
    predictions = model.predict(X_test)
    metric = metric_function(y_test, predictions)
    print(f"{model_name} {metric_name}: {metric:.4f}")

    # Save the model
    model_path = f"models/{model_name.replace(' ', '_').lower()}_model.pkl"
    save_model(model, model_path)
    print(f"Model saved as '{model_path}'.")

    print("\nTraining complete!\n")
