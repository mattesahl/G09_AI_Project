# Stroke prediction - AI Project by group G09
## Group members
| Name              | Department| University                                | Email                                |
|-------------------|------------|---------------------------------------------|--------------------------------------|
| Mattias Sahlstrand      | Mechanical Engineering  | Hanyang University | mattias@hanyang.ac.kr                  |
| Pontus Donnér  |    |Hanyang University | pontus_donner@hotmail.com                       |
| Belen Herranz Campusano |  Computer Science  | Hanyang University |  100495930@alumnos.uc3m.es  |
| Hugo Nicolay |    | Hanyang University | nicolayhugo1@gmail.com   |

## Research idea
**Title:** Stroke prediction  

## Introduction 

Strokes are one of the biggest health concerns in the world, ranking as the second leading cause of death according to the World Health Organization. Understanding the factors that contribute to stroke risk is crucial in predicting and potentially preventing such incidents. 

To address this, we’ve developed a data-trained model capable of analyzing health and lifestyle data to predict the likelihood of a patient experiencing a stroke. By using machine learning, our goal is to create a reliable tool that uses datas from the past to make accurate predictions and provide valuable insights for healthcare professionals and researchers. 

This project aims to simplify the process of risk assessment and offer a data-driven approach to stroke prediction. 


## Dataset

**Dataset Description:**

For this project, we are using a dataset that contains 5,110 entries with 12 columns. Each column contains  a specific detail about an individual, such as their health status and lifestyle choices, which are useful factors  in order to predict  stroke risks.



Here are the 12 features included in the dataset:

Id: Unique identifier for each entry.

Gender: Gender of the individual.

Age: Age in years.

Hypertension: Indicates if the individual has hypertension (0 = No, 1 = Yes).

Heart_Disease: Indicates if the individual has heart disease (0 = No, 1 = Yes).

Ever_Married: Marital status (Yes/No).

Work_Type: Type of employment (e.g., Private, Self-employed, etc.).

Residence_Type: Living area (Urban/Rural).

Avg_Glucose_Level: Average glucose level in blood (mg/dL).

BMI: Body Mass Index.

Smoking_Status: Smoking experience (Smokes, Used to smoke, Never smoked).

Stroke: Indicates whether the individual has had a stroke (0 = No, 1 = Yes).

**Why this dataset?**

This dataset was chosen because it includes a lot of  various datas, which is important to increase the accuracy of the model. The combination of numerical and categorical features makes it ideal for machine learning applications.

**Observations and adjustments:**

During our analysis of the dataset, we noticed that some columns, such as BMI, contain missing values, while others, like Smoking_Status, include unknown data points. Handling these gaps will be the first task  in building a robust predictive model. Additionally, by focusing on the most impactful features, we want  to improve the model's accuracy. 

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

## :mag: Evaluation & Analysis
After we run our code, the output in the terminal looks like this:   
![image](https://github.com/user-attachments/assets/000f5c91-e053-401c-a175-06c4a4a57638)    
**Figure 1. Output in terminal.**

As we can see in figure 1, the dataset is complete, and we have data for all variables of all the patients that are left in the dataset. We can also see the distribution of the target variable in our dataset, where 4700 patients did not have a stroke and 209 people did have it. We addressed this imbalance by using SMOTE (Synthetic Minority Oversampling Technique) to generate additional samples for the minority class. This was done in the oversampling part of the code. We did this to improve the training process since it otherwise would have favored the majority class (“no stroke”).

### Model performance
From the terminal output, we can see that the Random Forest model performed with an accuracy of 95,64% on the test set. If we look at our classification report, the model achieved a precision of 97% for the majority part (people who didn’t have a stroke) and 94% for the minority part (people who did have a stroke). It is similar numbers, although reversed, for the recall score which was 94% for the majority part and 98% for the minority part. This resulted in a f1-score of 0.96 for “Stroke” and 0.95 for “No stroke”. Overall, our model performed with an accuracy of 96% on the test data (based on 1880 patients). Furthermore, the calculated Cross-validated AUC of 0.95 confirms that the model we have built is performing with a great level of accuracy.

This level of accuracy might seem suspiciously high, but when looking at other people analyzing the same dataset (but using a different method) this is not uncommon at all. Rather, in many instances we found people achieving above 97% accuracy. Therefore, we believe that our method is working well. It is also positive to see that the accuracy of predicting a stroke is so similar to the accuracy of predicting no stroke.

### Confusion Matrix
![image](https://github.com/user-attachments/assets/a23175f7-2ecc-47f9-adc8-942f0e26dbed)     
**Figure 2. The Confusion Matrix.**

Figure 2 (The Confusion Matrix) shows how the model predicted the test data. As we can see, the model correctly classified 867 patients with no stroke and 931 patients who had a stroke. In 23 instances the model predicted a no stroke when the correct answer was stroke (false negative) and similarly did it in 59 cases predicted a stroke when it should have been no stroke (false positive).

### Receiver Operating Characteristic (ROC) Curve
![image](https://github.com/user-attachments/assets/8810de23-ca29-4645-9aa0-e45c0c83b924)      
**Figure 3. Receiver Operating Characteristic (ROC) Curve.**

The ORC curve visualizes the relationship between the false positive rate and the true positive rate. That is a measurement of the correctness of the model. In our model, the AUC (Area Under the Curve) is calculated as 0.99. This shows an excellent model performance. The sketched blue line is where the ROC curve would be if we had blindly guessed on a large dataset.

### Feature Importance
![image](https://github.com/user-attachments/assets/a12f2af1-aa33-4942-9aa7-683a3c795d1d)     
**Figure 4. Feature importance.**

Figure 4 illustrates which feature has the most influence on a patient having a stroke. It shows that the patient’s age is the most important factor, with a calculated importance of over 40% in this case. This is very logical since it is known that stroke is more common to occur for older people. The two features following age are the patient’s average glucose level and then BMI. The differences between these three features and the rest are quite large, indicating that these are the three most important features. It is interesting to see how low importance hypertension and heart disease were calculated to have in our data (it seems to have a lower importance than whether the patient was married).

### Correlation Heatmap
![image](https://github.com/user-attachments/assets/9f204254-f299-4fa1-a3c8-a648ad278cdd)      
**Figure 5. Correlation Heatmap.**

The Correlation Heatmap (figure 5) quantifies the linear relationships between the different variables. If we look at our target variable (stroke), it is easy to see that it had the highest correlation with age. This indicates a moderate positive correlation, which means that the older a person becomes, the bigger the likelihood of the person having a stroke. The same can be said for hypertension, average glucose level, and heart disease, which were the three variables showing a 0.14 correlation with stroke. 

### Combined analysis
When combining the correlation heatmap and the feature importance we can gain some interesting insights. At first, we can see that age has a strong correlation with stroke as well as the highest importance of all the variables. This reinforces that age plays a critical role in stroke prediction.
Furthermore, we discovered that heart disease and hypertension, which had the lowest importance when predicting stroke out of all variables, still had the second highest linear correlation. This could be a good example that correlation does not imply causation. Our theory is that heart disease and hypertension have a high correlation with other features that improve the risk of stroke. This could be for example age or average glucose levels. When these features are included in the model, they can overshadow the predictive contribution of heart disease and hypertension, reducing their individual feature importance. 
An interesting observation is that BMI, which has a modest linear correlation with stroke of 0.04, is the third most important variable. This indicates that BMI captures information that is significant when combining it with other features. It is also worth pointing out that gender has a low score of importance, as well as a low linear correlation with stroke suggesting that stroke is equally common for men and women.



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
In conclusion, we have in this project successfully created a model that, with the application of machine learning techniques predicts whether a person is likely to have a stroke. This prediction is based on ten different clinical features: the patient’s gender, age, average glucose level, BMI, residence type, smoking status, kind of work, marriage status, and also information if the patient has hypertension or heart disease. In order to do so, we first performed preprocessing steps on the comprehensive dataset of 5110 patients (for example handling that the data set had missing values and solving problems with class imbalance through SMOTE) before training a Random Forest model on 80% of the data set. Our trained model achieved a prediction accuracy of 96% when we tested it on the remaining 20% of the dataset. This indicates that the model in an effective way can distinguish between stroke and non-stroke cases when given specific clinical features. The findings from our analysis indicate that age, average glucose level, and BMI are critical factors influencing stroke risk, where age is the most significant predictor.
