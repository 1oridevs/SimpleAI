import os
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

def evaluate_model():
    print("\n--- Model Evaluation ---")

    # Step 1: Load the preprocessed test data
    try:
        X_test = pd.read_csv("data/X_test.csv")
        y_test = pd.read_csv("data/y_test.csv")
        print("Test data loaded successfully!")
    except FileNotFoundError:
        print("Error: Test data not found. Please run the 'Preprocess Data' step first.")
        return

    # Step 2: Load the trained model
    model_path = "models/decision_tree_model.pkl"
    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at '{model_path}'. Please train the model first.")
        return

    try:
        print(f"Loading trained model from '{model_path}'...")
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Step 3: Make predictions
    print("Making predictions on the test set...")
    predictions = model.predict(X_test)

    # Step 4: Calculate evaluation metrics
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("\nEvaluation complete!\n")
