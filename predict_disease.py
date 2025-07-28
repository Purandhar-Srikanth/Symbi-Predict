import pickle
import numpy as np
import pandas as pd

def load_model_and_symptoms():
    """
    Load the trained logistic regression model and symptoms list.
    
    Returns:
        tuple: (model, symptoms_list)
    """
    try:
        # Load the model
        with open("E:\\Projects\\Symbi new\\models\\logistic_regression_model.pkl", 'rb') as f:
            model = pickle.load(f)
        
        # Load the symptoms list
        with open("E:\\Projects\\Symbi new\\models\\symptoms_list.pkl", 'rb') as f:
            symptoms_list = pickle.load(f)
        
        return model, symptoms_list
    except FileNotFoundError:
        print("Error: Model files not found. Please run train_logistic_regression.py first.")
        return None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def predict_disease(symptoms_input):
    """
    Predict disease based on input symptoms.
    
    Args:
        symptoms_input: Dictionary with symptom names as keys and 0/1 as values
                      (1 indicates presence of the symptom, 0 indicates absence)
    
    Returns:
        Predicted disease name
    """
    # Load model and symptoms list
    model, symptoms_list = load_model_and_symptoms()
    if model is None or symptoms_list is None:
        return "Failed to load model."
    
    # Create feature vector (initialize all symptoms to 0)
    features = np.zeros(len(symptoms_list))
    
    # Set the values for symptoms that are present
    for symptom, value in symptoms_input.items():
        if symptom in symptoms_list:
            idx = symptoms_list.index(symptom)
            features[idx] = value
        else:
            print(f"Warning: '{symptom}' is not in the known symptoms list and will be ignored.")
    
    # Make prediction
    prediction = model.predict([features])[0]
    
    # Get prediction probabilities
    proba = model.predict_proba([features])[0]
    max_proba = max(proba)
    
    return prediction, max_proba

def get_top_diseases(symptoms_input, top_n=5):
    """
    Get top N most likely diseases based on input symptoms.
    
    Args:
        symptoms_input: Dictionary with symptom names as keys and 0/1 as values
        top_n: Number of top predictions to return
    
    Returns:
        List of tuples (disease, probability)
    """
    # Load model and symptoms list
    model, symptoms_list = load_model_and_symptoms()
    if model is None or symptoms_list is None:
        return "Failed to load model."
    
    # Create feature vector
    features = np.zeros(len(symptoms_list))
    for symptom, value in symptoms_input.items():
        if symptom in symptoms_list:
            idx = symptoms_list.index(symptom)
            features[idx] = value
    
    # Get prediction probabilities
    proba = model.predict_proba([features])[0]
    
    # Get top N predictions
    classes = model.classes_
    predictions = []
    for i in range(len(classes)):
        predictions.append((classes[i], proba[i]))
    
    # Sort by probability in descending order
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N predictions
    return predictions[:top_n]

def interactive_diagnosis():
    """
    Interactive function to input symptoms and get disease prediction.
    """
    # Load model and symptoms list
    model, symptoms_list = load_model_and_symptoms()
    if model is None or symptoms_list is None:
        return
    
    print("Welcome to Disease Prediction System")
    print("===================================")
    print("Please indicate which symptoms you have (1 for yes, 0 for no):")
    
    symptoms_input = {}
    
    # Ask about common symptoms first
    for symptom in symptoms_list:
        while True:
            try:
                response = input(f"Do you have '{symptom}'? (1: Yes, 0: No, s: Skip): ")
                if response.lower() == 's':
                    break
                value = int(response)
                if value in [0, 1]:
                    symptoms_input[symptom] = value
                    break
                else:
                    print("Please enter 0 or 1")
            except ValueError:
                print("Please enter a valid number (0 or 1) or 's' to skip")
        
        # After every 5 symptoms, check if we have enough information
        if len(symptoms_input) % 5 == 0 and len(symptoms_input) > 0:
            prediction, confidence = predict_disease(symptoms_input)
            print(f"\nBased on symptoms so far, potential diagnosis: {prediction} (Confidence: {confidence:.2f})")
            
            if confidence > 0.8:  # High confidence threshold
                continue_response = input("We have a high-confidence prediction. Continue with more symptoms? (y/n): ")
                if continue_response.lower() != 'y':
                    break
    
    # Final prediction
    top_diseases = get_top_diseases(symptoms_input)
    
    print("\nDiagnosis Results:")
    print("=================")
    for disease, probability in top_diseases:
        print(f"{disease}: {probability:.4f} ({probability*100:.1f}%)")
    
    print("\nNote: This is an automated prediction and should not replace professional medical advice.")
    print("Please consult with a healthcare professional for proper diagnosis and treatment.")

if __name__ == "__main__":
    # Example of direct prediction
    print("Example 1: Direct prediction with specific symptoms")
    example_symptoms = {
        "fever": 1,
        "cough": 1,
        "shortness of breath": 1,
        "fatigue": 1,
        "headache": 1
    }
    
    try:
        prediction, confidence = predict_disease(example_symptoms)
        print(f"Predicted disease: {prediction} (Confidence: {confidence:.4f})")
    except Exception as e:
        print(f"Error in prediction: {e}")
    
    print("\nExample 2: Interactive diagnosis")
    response = input("Would you like to try the interactive diagnosis system? (y/n): ")
    if response.lower() == 'y':
        interactive_diagnosis()
    else:
        print("Exiting program.")
