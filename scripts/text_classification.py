import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils.model_utils import save_model

def train_text_classification():
    print("\n--- Text Classification Training ---")

    # Step 1: Get dataset file path
    dataset_path = input("Enter the path to your text dataset file (CSV format): ").strip()
    if not os.path.exists(dataset_path):
        print("Error: File not found. Please try again.")
        return

    # Step 2: Load dataset
    try:
        print("Loading dataset...")
        data = pd.read_csv(dataset_path)
        print(f"Dataset loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns.")
    except Exception as e:
        print(f"Error: Unable to load the dataset. {e}")
        return

    # Step 3: Display a preview of the data
    print("\nDataset preview:")
    print(data.head())

    # Step 4: Select text and target columns
    text_column = input("Enter the column name for text data: ").strip()
    target_column = input("Enter the column name for target labels: ").strip()
    if text_column not in data.columns or target_column not in data.columns:
        print(f"Error: '{text_column}' or '{target_column}' not found in the dataset.")
        return

    # Step 5: Preprocess text data
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data[text_column])
    y = data[target_column]

    # Step 6: Split data into training and testing sets
    print("\nSplitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data split successfully!")

    # Step 7: Train the model
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    print("Model training complete!")

    # Step 8: Evaluate the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Step 9: Save the model and vectorizer
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    save_model(model, f"{output_dir}/text_classification_model.pkl")
    save_model(vectorizer, f"{output_dir}/vectorizer.pkl")
    print("\nModel and vectorizer saved successfully!")
