from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model and vectorizer (for text classification or other models)
MODEL_PATH = "models/text_classification_model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

try:
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
    print(f"Model loaded successfully from {MODEL_PATH}.")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Please train a model first.")
    model = None

try:
    with open(VECTORIZER_PATH, "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    print(f"Vectorizer loaded successfully from {VECTORIZER_PATH}.")
except FileNotFoundError:
    print(f"Error: Vectorizer file not found at {VECTORIZER_PATH}. Please train a text model first.")
    vectorizer = None

    
@app.route("/")
def homepage():
    return "Model Online!"


@app.route("/predict", methods=["POST"])
def predict():
    if not model or not vectorizer:
        return jsonify({"error": "Model or vectorizer not available. Please train a model first."})

    try:
        # Get JSON input
        input_data = request.get_json()
        input_text = input_data.get("text", "")
        if not input_text:
            return jsonify({"error": "No input text provided."})

        # Vectorize the input and make a prediction
        input_vector = vectorizer.transform([input_text])
        prediction = model.predict(input_vector)
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)

