from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import os

app = Flask(__name__)

# Globals
dataset = None
model = None

# Load the trained model once at the start
def load_model():
    global model
    if model is None:
        model = joblib.load('model.pkl')

# Upload Endpoint
@app.route('/upload', methods=['POST'])
def upload_file():
    global dataset
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    try:
        dataset = pd.read_csv(file)
        return jsonify({
            "message": "File uploaded successfully",
            "columns": list(dataset.columns)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Train Endpoint
@app.route('/train', methods=['POST'])
def train_model():
    global dataset, model
    if dataset is None:
        return jsonify({"error": "No dataset uploaded"}), 400

    try:
        # Handle missing values in the target column
        dataset = dataset.dropna(subset=['DefectStatus'])

        # Convert DefectStatus to numeric if it's categorical
        if dataset['DefectStatus'].dtype == 'object':
            dataset['DefectStatus'] = dataset['DefectStatus'].map({'Yes': 1, 'No': 0})

        # Feature selection
        X = dataset[['ProductionVolume', 'DefectRate', 'QualityScore', 'MaintenanceHours']]
        y = dataset['DefectStatus']

        # Split the dataset into training, validation, and testing sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=100)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=100)

        # Model training
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Evaluate on validation data
        y_val_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred)

        # Save the model
        joblib.dump(model, 'model.pkl')

        # Test the model
        y_test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        return jsonify({
            "accuracy": accuracy,
            "f1_score": f1,
            "test_accuracy": test_accuracy
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Predict Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    load_model()  # Load the model once at the start of prediction

    # Example input for prediction
    input_data = request.get_json()
    input_df = pd.DataFrame([input_data])

    # Define top features
    top_features = ["ProductionVolume", "DefectRate", "QualityScore", "MaintenanceHours"]

    # Ensure only top features are used
    input_data_prepared = input_df[top_features]

    # Make prediction
    prediction = model.predict(input_data_prepared)
    confidence = max(model.predict_proba(input_data_prepared)[0])

    # Adjusting logic for prediction based on confidence
    if confidence < 0.8:
        prediction_result = "Non-Defective"  # Confidence is low, classify as Non-Defective
    else:
        prediction_result = "Defective"  # Confidence is high, classify as Defective

    return jsonify({
        "DefectStatus": prediction_result,
        "Confidence": round(confidence * 100, 2)  # Confidence in percentage
    })

if __name__ == '__main__':
    app.run(debug=True)
