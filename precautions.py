import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

data = pd.read_csv('chatbot\Dataset\disease symptoms\symptom_precaution.csv')

features = data['Disease']
target = data[['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]

data.fillna(0, inplace=True)

label_encoder = LabelEncoder()
features_encoded = label_encoder.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(features_encoded, target, test_size=0.2, random_state=42)

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train.reshape(-1, 1), y_train)

y_pred = rf_regressor.predict(X_test.reshape(-1, 1))

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
