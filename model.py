# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib  # For saving the model

# Load the dataset
df = pd.read_csv("electricity.csv")

# Data Preprocessing
X = df[['Electricity Consumption']]  # Assuming this is the feature column
y = df['Electricity Price']  # Assuming this is the target column

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'electricity_price_prediction_model.pkl')
