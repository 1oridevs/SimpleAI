import pickle

def save_model(model, filepath):
    """Save a trained model to a file."""
    try:
        with open(filepath, "wb") as file:
            pickle.dump(model, file)
        print(f"Model saved to {filepath}.")
    except Exception as e:
        print(f"Error saving model: {e}")
