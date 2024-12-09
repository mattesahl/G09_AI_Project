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
### Libraries
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

Additionally, we configured the visualization settings to ensure consistency and readability in our plots.

## Evaluation & Analysis

## Related Work

## Conclusion
