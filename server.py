from flask import Flask, request, jsonify
from model import ModelHandler
import jwt
from datetime import datetime, timedelta

app = Flask(__name__)

# Secret key for JWT
SECRET_KEY = "iloveyou"

# Instantiate the model handler
model_handler = ModelHandler()

# Middleware for JWT authorization
def verify_jwt(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

@app.before_request
def authorize_request():
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return jsonify({"error": "Unauthorized"}), 401
    token = auth_header.split("Bearer ")[1]
    if not verify_jwt(token):
        return jsonify({"error": "Unauthorized"}), 401

# Routes
@app.route("/load-data", methods=["POST"])
def load_data():
    data = request.json
    file_path = data.get("file_path")
    input_columns = data.get("input_columns", [])
    label_column = data.get("label_column")

    if not file_path or not input_columns or not label_column:
        return jsonify({"error": "file_path, input_columns, and label_column are required"}), 400

    try:
        global inputs, labels
        inputs, labels = model_handler.load_data(file_path, input_columns, label_column)
        return jsonify({"message": "Data loaded successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/train", methods=["POST"])
def train():
    try:
        metrics = model_handler.train_model(inputs, labels)
        return jsonify({
            "message": "Model trained successfully",
            "metrics": metrics,
            "best_params": model_handler.best_params
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/save", methods=["POST"])
def save():
    try:
        model_handler.save_model()
        return jsonify({"message": "Model saved successfully", "path": "model.pkl"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/load", methods=["POST"])
def load():
    try:
        model_handler.load_model()
        return jsonify({"message": "Model loaded successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    input_values = data.get("input_values")

    if input_values is None:
        return jsonify({"error": "input_values is required"}), 400

    try:
        predictions = model_handler.predict(input_values)
        return jsonify({"predictions": predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/reset", methods=["POST"])
def reset():
    try:
        model_handler.reset_model()
        return jsonify({"message": "Model reset successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
