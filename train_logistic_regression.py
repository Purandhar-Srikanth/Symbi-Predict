import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

# Path to the CSV file
csv_file = "E:\\Projects\\Symbi new\\diseaseandsymptoms.csv"

print("Loading dataset...")
# Read the CSV file
data = pd.read_csv(csv_file)

# Extract the features (symptoms) and target (diseases)
X = data.iloc[:, 1:].values  # All columns except the first one (which contains disease names)
y = data.iloc[:, 0].values   # First column contains disease names

# Split the dataset into training and testing sets
print("Splitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
print(f"Number of features (symptoms): {X_train.shape[1]}")
print(f"Number of unique diseases: {len(np.unique(y))}")

# Initialize and train the logistic regression model
print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto')
model.fit(X_train, y_train)

# Make predictions on the test set
print("Making predictions on test set...")
y_pred = model.predict(X_test)

# Evaluate the model
print("\nModel Evaluation:")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Save the model
print("\nSaving the trained model...")
model_directory = "E:\\Projects\\Symbi new\\models"
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

model_file = os.path.join(model_directory, "logistic_regression_model.pkl")
with open(model_file, 'wb') as f:
    pickle.dump(model, f)

# Save the list of symptoms (feature names)
symptoms = data.columns[1:].tolist()  # All column names except the first one
symptoms_file = os.path.join(model_directory, "symptoms_list.pkl")
with open(symptoms_file, 'wb') as f:
    pickle.dump(symptoms, f)

print(f"Model saved to {model_file}")
print(f"Symptoms list saved to {symptoms_file}")

# Create a simple prediction function to demonstrate usage
print("\nCreating a sample prediction function...")

def predict_disease(symptoms_dict):
    """
    Make a prediction based on symptoms.
    
    Args:
        symptoms_dict: A dictionary with symptom names as keys and 0/1 as values
                      (1 indicates presence of the symptom, 0 indicates absence)
    
    Returns:
        Predicted disease name
    """
    # Create a feature vector
    features = np.zeros(len(symptoms))
    for symptom, value in symptoms_dict.items():
        if symptom in symptoms:
            idx = symptoms.index(symptom)
            features[idx] = value
    
    # Make prediction
    prediction = model.predict([features])[0]
    return prediction

print("\nSample prediction code generated and ready to use.")
print("You can now use the saved model to predict diseases based on symptoms.")
print("\nExample usage:")
print("1. Load the model and symptoms list")
print("2. Create a symptoms dictionary with values 1 for present symptoms and 0 for absent symptoms")
print("3. Call the predict_disease function to get a predicted disease")
