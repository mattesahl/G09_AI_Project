import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
)
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV   #


#For visualisation
plt.rcParams['figure.figsize'] = (10, 6.6)
sns.set_style('whitegrid')

#Load dataset
original_dataset = pd.read_csv("stroke_dataset.csv")
data = original_dataset.copy()

#Check ifu missing any data
print(f"Number of missing data:\n{data.isnull().sum()}")

#Label
categorical_columns = ['smoking_status', 'gender', 'ever_married', 'Residence_type', 'work_type']
for col in categorical_columns:
    data[col] = LabelEncoder().fit_transform(data[col])


#Correlation
correlation_matrix = data.corr()
plt.figure(figsize=(8, 6.4))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

#Data distribution
print(f"Distribution of target variable:\n{data['stroke'].value_counts()}")

#Oversampling
X = data.drop('stroke', axis=1)
y = data['stroke']
smote = SMOTE(random_state=10)
X_resampled, y_resampled = smote.fit_resample(X, y)

#Traning/test data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

#Random Forest
rf_model = RandomForestClassifier(n_estimators=1000, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

#Prediction
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

#Classification report
print("\nClassification report:")
print(classification_report(y_test, y_pred))

#Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Stroke", "Stroke"], yticklabels=["No Stroke", "Stroke"])
plt.title("Confusion Matrix")
plt.xlabel("Prediction")
plt.ylabel("Actual")
plt.show()

#ROC-curve
y_score = rf_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = roc_auc_score(y_test, y_score)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC-curve (AUC = {roc_auc:.2f})", color="darkorange")
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.show()

#Cross validated AUC
scores = cross_val_score(rf_model, X_resampled, y_resampled, cv=5, scoring='roc_auc')
print(f"Cross-validated AUC: {scores.mean():.2f}")

#Determine important features
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title("Feature Importance")
plt.show()


