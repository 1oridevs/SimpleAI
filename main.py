import os
import sys

def main_menu():
    print("Welcome to SimpleAI!")
    print("Select an option to get started:")
    print("1. Preprocess Data")
    print("2. Train a Model")
    print("3. Evaluate a Model")
    print("4. Exit")

    choice = input("Enter your choice (1-4): ")
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
        print("Exiting SimpleAI. Goodbye!")
        sys.exit()
    else:
        print("Invalid choice. Please try again.")
        main_menu()


if __name__ == "__main__":
    main_menu()
