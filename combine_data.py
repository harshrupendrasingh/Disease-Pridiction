import pandas as pd

disease_symptoms = pd.read_csv('chatbot\Dataset\disease symptoms\dataset_symptoms.csv')
precaution_symptom = pd.read_csv('chatbot\Dataset\disease symptoms\symptom_precaution.csv')
symptoms_severity = pd.read_csv('chatbot\Dataset\disease symptoms\Symptom-severity.csv')
disease_description = pd.read_csv('chatbot\Dataset\disease symptoms\symptom_Description.csv')

# Merge the datasets
merged_data = disease_symptoms.merge(precaution_symptom, on='Disease', how='left')
merged_data = merged_data.merge(symptoms_severity, left_on='Symptom_1', right_on='Symptom', how='left')
merged_data = merged_data.merge(disease_description, on='Disease', how='left')

# Drop the Symptom column
merged_data.drop('Symptom', axis=1, inplace=True)

# One-hot encode the symptom and precaution columns
symptoms_dummies = pd.get_dummies(merged_data[['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'Symptom_5', 'Symptom_6', 'Symptom_7', 'Symptom_8', 'Symptom_9', 'Symptom_10', 'Symptom_11', 'Symptom_12', 'Symptom_13', 'Symptom_14', 'Symptom_15', 'Symptom_16', 'Symptom_17']])
precautions_dummies = pd.get_dummies(merged_data[['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']])

# Concatenate the one-hot encoded columns back to the dataset
merged_data = pd.concat([merged_data, symptoms_dummies, precautions_dummies], axis=1)

# Drop the original symptoms and precautions columns
merged_data.drop(['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'Symptom_5', 'Symptom_6', 'Symptom_7', 'Symptom_8', 'Symptom_9', 'Symptom_10', 'Symptom_11', 'Symptom_12', 'Symptom_13', 'Symptom_14', 'Symptom_15', 'Symptom_16', 'Symptom_17', 'Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4'], axis=1, inplace=True)

# Save the merged dataset
merged_data_file = 'merged_data.csv'
merged_data.to_csv(merged_data_file, index=False)
