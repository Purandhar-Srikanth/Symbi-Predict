from flask import Flask, request, jsonify, render_template, send_from_directory
import os.path
import pickle
import numpy as np
import os

app = Flask(__name__, static_folder='static')

# Load the model and symptoms list
def load_model_and_symptoms():
    model_file = os.path.join(os.path.dirname(__file__), "models", "logistic_regression_model.pkl")
    symptoms_file = os.path.join(os.path.dirname(__file__), "models", "symptoms_list.pkl")
    
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    with open(symptoms_file, 'rb') as f:
        symptoms_list = pickle.load(f)
    
    return model, symptoms_list

# Route for disease prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        symptoms_input = data.get('symptoms', {})
        
        # Load model and symptoms list
        model, symptoms_list = load_model_and_symptoms()
        
        # Create feature vector (initialize all symptoms to 0)
        features = np.zeros(len(symptoms_list))
        
        # Set the values for symptoms that are present
        for symptom, value in symptoms_input.items():
            if symptom in symptoms_list:
                idx = symptoms_list.index(symptom)
                features[idx] = value
        
        # Make prediction
        prediction = model.predict([features])[0]
        
        # Get prediction probabilities
        proba = model.predict_proba([features])[0]
        
        # Normalize probabilities to sum to 100%
        normalized_proba = proba / np.sum(proba)
        max_proba = max(normalized_proba)
        
        # Get top 5 predictions
        classes = model.classes_
        predictions = []
        for i in range(len(classes)):
            predictions.append({
                "disease": classes[i],
                "probability": float(normalized_proba[i])  # Convert numpy float to Python float for JSON
            })
        
        # Sort by probability in descending order
        predictions.sort(key=lambda x: x["probability"], reverse=True)
        top_predictions = predictions[:3]  # Limit to top 3 diseases
        
        # Double-check that probabilities sum to 1.0 (100%)
        total_prob = sum(pred["probability"] for pred in top_predictions)
        if total_prob < 1.0:
            # If the top 5 don't sum to 100%, adjust the last one
            remaining_prob = 1.0 - total_prob
            top_predictions.append({
                "disease": "Other conditions",
                "probability": float(remaining_prob)
            })
        
        return jsonify({
            "prediction": prediction,
            "confidence": float(max_proba),
            "top_predictions": top_predictions
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to get all symptoms
@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    try:
        # Load model and symptoms list
        _, symptoms_list = load_model_and_symptoms()
        
        return jsonify({
            "symptoms": symptoms_list
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route for the main page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Route for the about page
@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

if __name__ == '__main__':
    # Check if model files exist
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    model_file = os.path.join(model_dir, "logistic_regression_model.pkl")
    symptoms_file = os.path.join(model_dir, "symptoms_list.pkl")
    
    if not os.path.exists(model_file) or not os.path.exists(symptoms_file):
        print("Model files not found. Please run train_logistic_regression.py first.")
    else:
        print("Starting Flask server...")
        print("Go to http://localhost:5000 in your browser to use the Disease Prediction System")
        # Run the app
        app.run(debug=True, port=5000)
