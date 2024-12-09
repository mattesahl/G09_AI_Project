# Stroke prediction - AI Project by group G09
## Group members
| Name              | Department| University                                | Email                                |
|-------------------|------------|---------------------------------------------|--------------------------------------|
| Mattias Sahlstrand      |   | Hanyang University | mattias@hanyang.ac.kr                  |
| Pontus Donnér  |    |Hanyang University | pontus_donner@hotmail.com                       |
| Belen Herranz Campusano |    | Hanyang University |  100495930@alumnos.uc3m.es  |
| Hugo Nicolay |    | Hanyang University | nicolayhugo1@gmail.com   |

## Research idea
**Title:** Stroke prediction  
**Proposal:** (this text will be deleted)

During this project, we will analyze a dataset (www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) in order to build an AI model that can predict the likelihood of patients getting a stroke. The dataset that we will use contains information about around 5,000 patients, and each patient is described with 11 different clinical features. These features are the patient’s gender, age, average glucose level, BMI, residence type, smoking status, kind of work, marriage status, and also information if the patient has hypertension or a heart disease. By combining this dataset with machine learning techniques, we hope to create our AI model to gain reliable predictions. Furthermore, this project and our AI model might also help us to understand which of the 11 features might correlate most with stroke. This knowledge could potentially inform healthcare providers and policymakers in designing better preventive measures. Additionally, it may serve as a foundation for further research into early intervention strategies tailored to high-risk groups.

## Introduction 

## Dataset

## :gear:	Methodology
### Imports
First, we import the necessary libraries, which included tools for data manipulation (pandas), visualization (matplotlib and seaborn), machine learning (scikit-learn), and oversampling techniques (imblearn). 

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
)
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
```

We configured the visualization settings to ensure consistency and readability in our plots.
```pyhton
plt.rcParams['figure.figsize'] = (10, 6.6)
sns.set_style('whitegrid')
```

Moreover, our dataset is loaded using pandas and we create a copy of the original dataset to ensure that the original data is not altered.
```pyhton
# Load the dataset
original_dataset = pd.read_csv("stroke_dataset.csv")
data = original_dataset.copy()
```

### Data Preprocessing
#### Handling missing values
The BMI column contained approximately 200 missing values. Since the dataset has over 5,000 subjects, we decided to remove these rows rather than calculate the missing values with the mean of the others. We discarted imputation because it would not represent real data.
In addition, we kept the category with individuals with an unknown smoking status because this group was huge, so excluding them would result in a great loss of information.
The ID column was also dropped because it is an identifier that does not contribute to stroke prediction.
#### Encoding categorical variables
As the initial dataset included some categorical columns, such as smoking_status, gender, ever_married, Residence_type, and work_type. Those columns were encoded into numerical values using the LabelEncoder function, which transforms categories into integer numbers that can be manage by machine learning algorithms.
```pyhton
# Encoding categorical columns
categorical_columns = ['smoking_status', 'gender', 'ever_married', 'Residence_type', 'work_type']
for col in categorical_columns:
    data[col] = LabelEncoder().fit_transform(data[col])
```

### Correlation Matrix
To analyze relationships between variables, we calculated the correlation matrix. This provided insights into how features were related to each other and to the main variable, stroke. A heatmap was used to visualize the correlations.

```python
# Calculating and visualizing the correlation matrix
correlation_matrix = data.corr()
plt.figure(figsize=(8, 6.4))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
```

### Oversampling
The dataset contains a great majority of cases were non-stroke, therefore, to address the severe class imbalance, we decided to use SMOTE (Synthetic Minority Oversampling Technique).
This technique generates synthetic examples for the minority class, in this case 'stroke cases', and balances the dataset to prevent the model from being biased towards the majority class.
```pyhton
# Separating features and target variable
X = data.drop('stroke', axis=1)
y = data['stroke']

# Applying SMOTE for oversampling
smote = SMOTE(random_state=10)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

### Model Selection
As the machine learning model of our porject, we have chosen Random Forest Classifier, first, cause it was a model worked in class, but mainly for its robustness, ability to handle mixed data types, and interpretability. 
Random Forest is an ensemble learning technique that builds multiple decision trees and outputs the majority vote for classification tasks. It is well-suited for our dataset, as it naturally handles class imbalance and provides feature importance.

### Training and Evaluation
We trained the model using 1,000 estimators, set the random state for reproducibility, and enabled parallel processing to improve efficiency.
```pyhton
# Training the Random Forest model
rf_model = RandomForestClassifier(n_estimators=1000, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
```

To evaluate the model's performance, we split the dataset into training and testing sets using an 80-20 split. This allowed us to measure the model's ability to generalize to unseen data. The metrics used for evaluation included:
1. Accuracy Score: The proportion of correctly predicted cases.
2. Classification Report: Metrics such as precision, recall, and F1-score for each class.
3. Confusion Matrix: Visual representation of true positives, true negatives, false positives, and false negatives.
4. ROC Curve and AUC: Indicates the trade-off between true positive and false positive rates, with AUC summarizing the model's performance.
```pyhton
# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Making predictions
y_pred = rf_model.predict(X_test)

# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classification report
print("\nClassification report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Stroke", "Stroke"], yticklabels=["No Stroke", "Stroke"])
plt.title("Confusion Matrix")
plt.xlabel("Prediction")
plt.ylabel("Actual")
plt.show()

# ROC curve and AUC
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
```
### Feature Importance Analysis
To understand which features were most important for predicting strokes, we analyzed the feature importance scores provided by the Random Forest Classifier. This analysis helps identify the key factors contributing to stroke risk.
```pyhton
# Extracting feature importances
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Visualizing feature importances
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title("Feature Importance")
plt.show()
```

## Evaluation & Analysis

## :books: Related Work
### Stroke Prediction with Machine Learning
Stroke Prediction Dataset on Kaggle: This is the dataset we were working with. The Kaggle page contains a detailed description of the dataset, including the characteristics of the variables and some prior analyses.
[Stroke Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

Saumya Gupta’s Stroke Prediction - Detailed EDA & 7 ML Models: Reference from another Kaggle user that worked with the same dataset
[Stroke Prediction - Detailed EDA & 7 ML Models](https://www.kaggle.com/code/saumyagupta2025/stroke-prediction-detailed-eda-7-ml-models)

### Using Random Forest for Prediction
Random Forest in Scikit-learn: The official Scikit-learn documentation explains how to implement and use the Random Forest algorithm for classification.
[Random Forest in Scikit-learn](https://scikit-learn.org/stable/modules/ensemble.html#random-forest)

Random Forest in Practice: A detailed explanation and practical guide on how to implement Random Forests in Python using Scikit-learn, from setting up the environment to interpreting results.
[Random Forest Practical Guide​](https://willkoehrsen.github.io/data%20science/machine%20learning/random-forest-simple-explanation/)

### Risk Factors for Strokes
Stroke Risk Factors: An overview of the main risk factors for stroke, including age, hypertension, diabetes, lifestyle choices, and family history. The CDC provides detailed information about how these factors contribute to stroke risk and offers advice on how to reduce the likelihood of having a stroke.
[CDC's Stroke Risk Factors](https://www.cdc.gov/stroke/risk-factors/index.html)
## Conclusion
