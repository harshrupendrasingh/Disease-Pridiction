from disease_detection import train_and_predict_disease

user_symptoms = "fever itching vomiting"
predicted_diseases, probabilities = train_and_predict_disease(user_symptoms, top_n=5)
for disease, probability in zip(predicted_diseases, probabilities):
    print(f"Disease: {disease}, Probability: {probability:.2f}")
