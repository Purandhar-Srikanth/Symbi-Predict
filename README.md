# Disease Prediction System using Logistic Regression

This system uses logistic regression to predict diseases based on symptoms. It consists of two main Python scripts:

## Files

1. `train_logistic_regression.py` - Trains a logistic regression model using the disease and symptoms dataset
2. `predict_disease.py` - Provides functions to use the trained model for disease prediction
3. `diseaseandsymptoms.csv` - The original dataset (your existing file)

## System Requirements

- Python 3.6 or higher
- Required Python packages:
  - pandas
  - numpy
  - scikit-learn
  - pickle

## Installation

If Python is not installed, you need to install it first:
1. Download Python from [python.org](https://www.python.org/downloads/)
2. During installation, make sure to check "Add Python to PATH"

After installing Python, install the required packages:

```
pip install pandas numpy scikit-learn
```

## How to Use

### Step 1: Train the Model

Run the training script:

```
python train_logistic_regression.py
```

This will:
- Load the dataset from the CSV file
- Train a logistic regression model
- Evaluate the model performance
- Save the trained model and symptoms list to the `models` folder

### Step 2: Predict Diseases

Use the prediction script:

```
python predict_disease.py
```

This script provides:
1. A direct prediction example using a predefined set of symptoms
2. An interactive diagnosis system where you can input symptoms and get predictions

### Custom Prediction

You can also import the functions from `predict_disease.py` into your own scripts:

```python
from predict_disease import predict_disease, get_top_diseases

# Create a dictionary of symptoms
symptoms = {
    "fever": 1,
    "cough": 1,
    "headache": 0,
    # Add more symptoms as needed...
}

# Get prediction
prediction, confidence = predict_disease(symptoms)
print(f"Predicted disease: {prediction} (Confidence: {confidence:.2f})")

# Get top 5 most likely diseases
top_diseases = get_top_diseases(symptoms, top_n=5)
for disease, probability in top_diseases:
    print(f"{disease}: {probability:.4f}")
```

## Important Notes

- This system is for educational purposes only and should not replace professional medical advice
- Consult with healthcare professionals for proper diagnosis and treatment
- The accuracy of predictions depends on the quality and size of the training dataset
