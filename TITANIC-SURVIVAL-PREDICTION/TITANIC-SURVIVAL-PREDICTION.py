import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

# Step 1: Import Libraries
warnings.filterwarnings('ignore')

# Step 2: Load and Explore Data
data = pd.read_csv("Titanic-Dataset.csv")
print(data.head())
print(data.isnull().sum())
print(data.describe())

sns.countplot(data['Survived'])
plt.title("Survival Distribution")
plt.show()

# Visualize survival by gender and passenger class using seaborn.catplot()
sns.catplot(x="Sex", hue="Survived", col="Pclass", data=data, kind="count")
plt.suptitle("Survival by Gender and Passenger Class")
plt.show()

# Step 3: Data Preprocessing
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Training and Evaluation
# Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print("Random Forest Classifier Accuracy:", rf_accuracy)
print(classification_report(y_test, rf_pred))
sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d')
plt.title("Random Forest Classifier Confusion Matrix")
plt.show()

# Logistic Regression
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
print("Logistic Regression Accuracy:", lr_accuracy)
print(classification_report(y_test, lr_pred))
sns.heatmap(confusion_matrix(y_test, lr_pred), annot=True, fmt='d')
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# Step 5: Cross-validation
rf_cv_scores = cross_val_score(rf_model, X, y, cv=5)
print("Random Forest CV Mean Accuracy:", np.mean(rf_cv_scores))
lr_cv_scores = cross_val_score(lr_model, X, y, cv=5)
print("Logistic Regression CV Mean Accuracy:", np.mean(lr_cv_scores))
