import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import pickle


# Dataset: 3 Load the dataset [Parkinson's Disease Dataset]
df = pd.read_csv('D:\\MDTE21\\Multi Diseses Prediction\\parkinsons - parkinsons.csv')
df.info()

# Drop irrelevant columns
df.drop('name', axis=1, inplace=True)

# Save the preprocessed dataset
df.to_csv('preprocessed_parkinsons_disease.csv', index=False)
print("Preprocessing complete. The preprocessed dataset is saved as 'preprocessed_parkinsons_disease.csv'.")


# Load the preprocessed dataset
df = pd.read_csv('D:\\MDTE21\\Multi Diseses Prediction\\preprocessed_parkinsons_disease.csv')

# Split the dataset into features and target variable
X = df.drop('status', axis=1)
y = df['status']
# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model: 1 Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Predict and evaluate the Logistic Regression model
y_pred_lr = lr.predict(X_test)

# Get probability estimates for the positive class
y_proba_lr = lr.predict_proba(X_test)[:, 1]  

# Evaluate the logistic regression model
accuracy = accuracy_score(y_test, y_pred_lr)
precision = precision_score(y_test, y_pred_lr)
recall = recall_score(y_test, y_pred_lr)
f1 = f1_score(y_test, y_pred_lr)
roc_auc = roc_auc_score(y_test, y_proba_lr)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_lr)
# Classification report
report = classification_report(y_test, y_pred_lr)

# Display
print("\nLogistic Regression Model Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")
print("\n📊 Confusion Matrix:")
print(cm)
print("\n📃 Classification Report:")
print(report)

# Model: 2 Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Predict and evaluate the Random Forest model
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

# Evaluate the random forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_proba_rf)

# Confusion matrix for Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
# Classification report for Random Forest
report_rf = classification_report(y_test, y_pred_rf)

# Display Random Forest results
print("\nRandom Forest Model Evaluation:")
print(f"Accuracy: {accuracy_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall: {recall_rf:.4f}")
print(f"F1 Score: {f1_rf:.4f}")
print(f"ROC AUC Score: {roc_auc_rf:.4f}")
print("\n📊 Confusion Matrix:")
print(cm_rf)
print("\n📃 Classification Report:")
print(report_rf)


# Model: 3 XGBoost
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)

# Predict and evaluate the XGBoost model
y_pred_xgb = xgb_model.predict(X_test)
y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

# Evaluate the XGBoost model
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb)
recall_xgb = recall_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)
roc_auc_xgb = roc_auc_score(y_test, y_proba_xgb)

# Confusion matrix for XGBoost
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
# Classification report for XGBoost
report_xgb = classification_report(y_test, y_pred_xgb)

# Display XGBoost results
print("\nXGBoost Model Evaluation:")
print(f"Accuracy: {accuracy_xgb:.4f}")
print(f"Precision: {precision_xgb:.4f}")
print(f"Recall: {recall_xgb:.4f}")
print(f"F1 Score: {f1_xgb:.4f}")
print(f"ROC AUC Score: {roc_auc_xgb:.4f}")
print("\n📊 Confusion Matrix:")
print(cm_xgb)
print("\n📃 Classification Report:")
print(report_xgb)


# Save logistic regression model
with open('D3_logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(lr, f)

# Save random forest model
with open('D3_random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf, f)

# Save XGBoost model
with open('D3_xgboost_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

# Save the scaler to a .pkl file
with open('D3_scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)