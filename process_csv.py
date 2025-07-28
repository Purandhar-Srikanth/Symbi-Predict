import csv

# Path to the CSV file
csv_file = "E:\\Projects\\Symbi new\\diseaseandsymptoms.csv"

# Read the CSV file
with open(csv_file, 'r') as file:
    csv_reader = csv.reader(file)
    
    # Read the header row (first row) which contains the disease column and symptom columns
    headers = next(csv_reader)
    
    # Extract all symptoms (skip the first column which is "diseases")
    symptoms = headers[1:]
    
    # Sort symptoms alphabetically
    sorted_symptoms = sorted(symptoms)
    
    # Create a set to store unique diseases
    diseases = set()
    
    # Read all rows to extract diseases
    for row in csv_reader:
        if row and len(row) > 0:
            diseases.add(row[0])
    
    # Sort diseases alphabetically
    sorted_diseases = sorted(diseases)

# Write symptoms to a file
with open("E:\\Projects\\Symbi new\\symptoms.txt", 'w') as symptoms_file:
    for symptom in sorted_symptoms:
        symptoms_file.write(symptom + "\n")

# Write diseases to a file
with open("E:\\Projects\\Symbi new\\diseases.txt", 'w') as diseases_file:
    for disease in sorted_diseases:
        diseases_file.write(disease + "\n")

print("Files created successfully:")
print("1. symptoms.txt - Contains all symptoms in alphabetical order")
print("2. diseases.txt - Contains all diseases in alphabetical order")
