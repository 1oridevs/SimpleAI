from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
model_path = "models/decision_tree_model.pkl"  # Example model path
with open(model_path, "rb") as file:
    model = pickle.load(file)
@app.route("/")
def homepage():
    return "Model Online!"
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON input
        input_data = request.get_json()
        input_df = pd.DataFrame([input_data])  # Convert to DataFrame

        # Make prediction
        prediction = model.predict(input_df)
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
