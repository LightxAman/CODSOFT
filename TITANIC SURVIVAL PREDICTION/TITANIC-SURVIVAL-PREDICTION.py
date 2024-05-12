# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Step 2: Load and Explore Data
# Load the Titanic dataset
data = pd.read_csv("Titanic-Dataset.csv")

# Display the first few rows of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Explore statistical summary
print(data.describe())

# Visualize survival distribution
sns.countplot(data['Survived'])
plt.title("Survival Distribution")
plt.show()

# Explore survival by gender
sns.countplot(data['Survived'], hue=data['Sex'])
plt.title("Survival by Gender")
plt.show()

# Explore survival by passenger class
sns.countplot(data['Survived'], hue=data['Pclass'])
plt.title("Survival by Passenger Class")
plt.show()

# Boxplot of age by passenger class
sns.boxplot(x='Pclass', y='Age', data=data)
plt.title("Age Distribution by Passenger Class")
plt.show()

# Step 3: Data Preprocessing
# Fill missing values
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Encode categorical variables
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Select features and target variable
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = data['Survived']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Training and Evaluation
# Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

# Print Random Forest Classifier results
print("Random Forest Classifier Accuracy:", rf_accuracy)
print(classification_report(y_test, rf_pred))

# Plot Random Forest Classifier Confusion Matrix
sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d')
plt.title("Random Forest Classifier Confusion Matrix")
plt.show()

# Logistic Regression
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)

# Print Logistic Regression results
print("Logistic Regression Accuracy:", lr_accuracy)
print(classification_report(y_test, lr_pred))

# Plot Logistic Regression Confusion Matrix
sns.heatmap(confusion_matrix(y_test, lr_pred), annot=True, fmt='d')
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# Step 5: Cross-validation
# Random Forest Cross-validation
rf_cv_scores = cross_val_score(rf_model, X, y, cv=5)
print("Random Forest CV Mean Accuracy:", np.mean(rf_cv_scores))

# Logistic Regression Cross-validation
lr_cv_scores = cross_val_score(lr_model, X, y, cv=5)
print("Logistic Regression CV Mean Accuracy:", np.mean(lr_cv_scores))
