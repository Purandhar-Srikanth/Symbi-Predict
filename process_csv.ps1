# Path to the CSV file
$csvFile = "E:\Projects\Symbi new\diseaseandsymptoms.csv"

# Read the CSV file
$csvData = Import-Csv -Path $csvFile

# Get the header row which contains all symptoms
$headers = $csvData | Get-Member -MemberType NoteProperty | 
    Where-Object { $_.Name -ne "diseases" } | 
    Select-Object -ExpandProperty Name

# Sort symptoms alphabetically
$sortedSymptoms = $headers | Sort-Object

# Extract all unique diseases
$diseases = $csvData | Select-Object -ExpandProperty diseases -Unique

# Sort diseases alphabetically
$sortedDiseases = $diseases | Sort-Object

# Write symptoms to a file
$sortedSymptoms | Out-File -FilePath "E:\Projects\Symbi new\symptoms.txt"

# Write diseases to a file
$sortedDiseases | Out-File -FilePath "E:\Projects\Symbi new\diseases.txt"

Write-Host "Files created successfully:"
Write-Host "1. symptoms.txt - Contains all symptoms in alphabetical order"
Write-Host "2. diseases.txt - Contains all diseases in alphabetical order"
