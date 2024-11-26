import os
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data():
    print("\n--- Data Preprocessing ---")
    
    # Step 1: Get dataset file path
    dataset_path = input("Enter the path to your dataset file (CSV format): ").strip()
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

    # Step 4: Clean dataset (optional step for this version)
    print("\nSkipping cleaning for now. Data assumed to be clean.")

    # Step 5: Split dataset into training and testing sets
    try:
        target_column = input("\nEnter the target column name: ").strip()
        if target_column not in data.columns:
            print(f"Error: '{target_column}' not found in the dataset.")
            return

        print("\nSplitting data into training and testing sets...")
        X = data.drop(columns=[target_column])
        y = data[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Data split successfully!")

        # Step 6: Save preprocessed data
        output_dir = "data"
        os.makedirs(output_dir, exist_ok=True)
        X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
        X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
        y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
        y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

        print(f"Preprocessed data saved in the '{output_dir}' directory.")
    except Exception as e:
        print(f"Error during preprocessing: {e}")

    print("\nPreprocessing complete!\n")
