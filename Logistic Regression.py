# Import libraries
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Iris dataset
iris = load_iris("iris_flower_classificattion.csv
                 ")
X = iris.data  # Features
y = iris.target  # Target (0: Setosa, 1: Versicolor, 2: Virginica)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Logistic Regression model
log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
log_reg.fit(X_train, y_train)

# Make predictions
y_pred = log_reg.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Example prediction for a new sample
sample = [[5.1, 3.5, 1.4, 0.2]]  # Example flower measurements
prediction = log_reg.predict(sample)
print("\nPredicted Class for sample:", iris.target_names[prediction[0]])
