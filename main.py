import os
import sys

def main_menu():
    print("Welcome to SimpleAI!")
    print("Select an option to get started:")
    print("1. Preprocess Data")
    print("2. Train a Model")
    print("3. Evaluate a Model")
    print("4. Load Prebuilt Dataset")
    print("5. Explain Model")
    print("6. Exit")

    choice = input("Enter your choice (1-5): ")
    if choice == "1":
        from scripts.preprocess import preprocess_data
        preprocess_data()
    elif choice == "2":
        from scripts.train import train_model
        train_model()
    elif choice == "3":
        from scripts.evaluate import evaluate_model
        evaluate_model()
    elif choice == "4":
        from scripts.prebuilt_datasets import load_prebuilt_dataset, save_prebuilt_dataset
        dataset_name = input("Select a dataset: (1) Iris, (2) California Housing: ").strip()
        dataset_name = "Iris" if dataset_name == "1" else "California Housing"
        df = load_prebuilt_dataset(dataset_name)
        if df is not None:
            save_prebuilt_dataset(df, target_column="target")
    elif choice == "5":
        from scripts.explain_model import explain_model
        explain_model()


    elif choice == "6":
        print("Exiting SimpleAI. Goodbye!")
        sys.exit()

    else:
        print("Invalid choice. Please try again.")
        main_menu()


if __name__ == "__main__":
    main_menu()
