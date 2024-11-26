from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
import os
def load_prebuilt_dataset(dataset_name):
    print(f"\nLoading {dataset_name} dataset...")
    if dataset_name == "Iris":
        data = load_iris(as_frame=True)
        df = pd.concat([data.data, data.target], axis=1)
        df.columns = list(data.feature_names) + ["target"]
    elif dataset_name == "California Housing":
        data = fetch_california_housing(as_frame=True)
        df = pd.concat([data.data, pd.DataFrame(data.target, columns=["target"])], axis=1)
    else:
        print("Dataset not found.")
        return None

    print(f"{dataset_name} dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

def save_prebuilt_dataset(df, target_column):
    """Save the dataset for training and testing."""
    print("Splitting the dataset into training and testing sets...")
    X = df.drop(columns=[target_column])
    y = df[[target_column]]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save the splits to the data directory
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    print(f"Prebuilt dataset saved and split into training and testing sets in the 'data/' directory.")
