import pickle
import pandas as pd
import numpy as np

def explain_model():
    print("\n--- Model Explainability ---")

    # Load the model
    model_path = "models/decision_tree_model.pkl"  # Example model path
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"Error: Model not found at {model_path}. Please train a model first.")
        return

    # Load the training data
    X_train_path = "data/X_train.csv"
    try:
        X_train = pd.read_csv(X_train_path)
    except FileNotFoundError:
        print(f"Error: Training data not found at {X_train_path}. Please preprocess the data first.")
        return

    # Explain the model
    if hasattr(model, "feature_importances_"):
        # Tree-based models
        print("\nFeature Importance:")
        importance = model.feature_importances_
        for feature, score in zip(X_train.columns, importance):
            print(f"{feature}: {score:.4f}")
    elif hasattr(model, "coef_"):
        # Linear models
        print("\nModel Coefficients:")
        coefficients = model.coef_.ravel()
        for feature, coef in zip(X_train.columns, coefficients):
            print(f"{feature}: {coef:.4f}")
    else:
        print("This model does not support explainability.")

    print("\nModel explanation complete!\n")
