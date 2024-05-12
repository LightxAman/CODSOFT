import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset with specified encoding
movie_data = pd.read_csv("IMDb Movies India.csv", encoding='latin1')

# Display the first few rows of the dataset
print(movie_data.head())

# Check for missing values
print(movie_data.isnull().sum())

# Drop rows with missing values or handle them accordingly
movie_data.dropna(inplace=True)

# Convert categorical variables into dummy/indicator variables
movie_data = pd.get_dummies(movie_data, columns=['genre', 'director', 'actor1', 'actor2', 'actor3'])

# Split the data into features (X) and target variable (y)
X = movie_data.drop(['rating'], axis=1)
y = movie_data['rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict ratings for the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Visualize the predicted vs actual ratings
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs Predicted Ratings")
plt.show()
