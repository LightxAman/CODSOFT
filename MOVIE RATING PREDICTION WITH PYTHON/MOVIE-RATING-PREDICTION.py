import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
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

# Remove commas from the 'Votes' column and convert it to numeric
movie_data['Votes'] = movie_data['Votes'].str.replace(',', '').astype(float)

# Split the 'Genre' column into multiple dummy variables
genres = movie_data['Genre'].str.get_dummies(sep=', ')
movie_data = pd.concat([movie_data, genres], axis=1)

# Drop unnecessary columns
movie_data.drop(['Name', 'Year', 'Duration', 'Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3'], axis=1, inplace=True)

# Split the data into features (X) and target variable (y)
X = movie_data.drop(['Rating'], axis=1)
y = movie_data['Rating']

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

# Calculate RMSE and MAE
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)

mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

# Calculate R-squared
r_squared = model.score(X_test, y_test)
print("R-squared:", r_squared)

# Visualize the predicted vs actual ratings
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs Predicted Ratings")
plt.show()

# Residual plot
plt.figure(figsize=(10, 6))
sns.residplot(y_test, y_pred, lowess=True, line_kws={'color': 'red', 'lw': 1})
plt.title('Residual Plot')
plt.xlabel('Predicted Ratings')
plt.ylabel('Residuals')
plt.show()

# Plot model coefficients
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
coefficients = coefficients.sort_values(by='Coefficient', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=coefficients)
plt.xlabel('Coefficient')
plt.ylabel('Feature')
plt.title('Model Coefficients')
plt.show()

# Plot feature importance
feature_importance = pd.Series(model.coef_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_importance.index)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()
