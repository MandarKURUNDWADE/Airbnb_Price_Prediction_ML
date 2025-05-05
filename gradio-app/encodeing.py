import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib

# Load training data (use the correct path)
data = pd.read_csv('airbnb_data.csv')

# Identify categorical features
categorical_features = ['property_type', 'room_type', 'bed_type', 'cancellation_policy', 'city']

# Create and fit the encoder
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(data[categorical_features])

# Save the encoder
joblib.dump(encoder, 'encoder.pkl')
print("Encoder saved successfully!")
