from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib  # Import for model saving

import pandas as pd

# Load the dataset
loan_data = pd.read_csv('Loan Status Prediction.csv')

# Preprocess categorical variables (assuming 'purpose' is categorical)
label_encoder = LabelEncoder()
loan_data['purpose'] = label_encoder.fit_transform(loan_data['purpose'])

# Define features and target variable
X = loan_data.drop(['Loan Repayment Status', 'Predicted Loan Repayment Status'], axis=1)
y = loan_data['Predicted Loan Repayment Status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier()

# Train the model on the training data
rf_classifier.fit(X_train, y_train)

# Predict on the test data
y_pred = rf_classifier.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Generate classification report
report = classification_report(y_test, y_pred)

# Save the trained model (replace 'loan_prediction_model.pkl' with your desired filename)
joblib.dump(rf_classifier, 'loan_prediction_model.pkl')

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
