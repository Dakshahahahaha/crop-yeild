import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Load your dataset
df = pd.read_csv("crop_yield.csv")  # Make sure this file is in the same folder

# Drop missing values
df.dropna(inplace=True)

# Encode categorical columns
label_encoders = {}
for col in ['State', 'Crop', 'Season']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target
X = df[['State', 'Crop_Year', 'Crop', 'Season', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']]
y = df['Production']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
with open("rf_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save the scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save label encoders
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print("✅ Model, scaler, and label encoders saved!")
