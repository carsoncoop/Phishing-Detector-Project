from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import joblib

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from the extension

# Absolute path to your model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "spam_pipeline.joblib")
model = joblib.load(model_path)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"prediction": "No text provided"}), 400

    # 0 = ham, 1 = spam
    pred = model.predict([text])[0]
    label = "Legit" if pred == 0 else "Spam"
    return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run(debug=True)
