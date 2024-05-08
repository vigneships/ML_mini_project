#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 14:26:18 2024

@author: it
"""

import pandas as pd
import numpy as np
import joblib   
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
df = pd.read_csv('/home/it/Downloads/11-dataset.csv')


df.head()
print(f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
df['casualty_severity'].value_counts().plot(kind='bar')
plt.title('Casualty Severity Counts')
plt.xlabel('Casualty Severity')
plt.ylabel('Count')
plt.show()
plt.figure(figsize=(10,6))
sns.boxplot(x='casualty_severity', y='age_of_casualty', data=df)
plt.title('Age of Casualty vs Casualty Severity')
plt.xlabel('Casualty Severity')
plt.ylabel('Age of Casualty')
plt.show()
sns.histplot(df['age_of_casualty'], kde=True)
plt.title('Age of Casualty Distribution')
plt.xlabel('Age of Casualty')
plt.ylabel('Frequency')
plt.show()
features = ['accident_year', 'vehicle_reference', 'casualty_reference', 'casualty_class', 'sex_of_casualty', 'age_of_casualty', 'age_band_of_casualty', 'casualty_severity', 'pedestrian_location', 'pedestrian_movement', 'car_passenger', 'bus_or_coach_passenger', 'pedestrian_road_maintenance_worker', 'casualty_type', 'casualty_home_area_type', 'casualty_imd_decile']
df = df[features]
df = df[features].copy()
le = LabelEncoder()
categorical_features = ['casualty_class', 'sex_of_casualty', 'car_passenger', 'bus_or_coach_passenger', 'pedestrian_road_maintenance_worker', 'casualty_type', 'casualty_home_area_type']
for feature in categorical_features:
    df[feature] = le.fit_transform(df[feature])

# Standard Scaling for numerical variables
scaler = StandardScaler()
numerical_features = ['accident_year', 'vehicle_reference', 'casualty_reference', 'age_of_casualty', 'age_band_of_casualty', 'pedestrian_location', 'pedestrian_movement', 'casualty_imd_decile']
for feature in numerical_features:
    df[feature] = scaler.fit_transform(df[feature].values.reshape(-1, 1))

df.head()
X = df.drop('casualty_severity', axis=1)
y = df['casualty_severity']
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Precision
precision = precision_score(y_test, y_pred, average='weighted')
print(f'Precision: {precision}')

# Recall
recall = recall_score(y_test, y_pred, average='weighted')
print(f'Recall: {recall}')

# F1 Score
f1 = f1_score(y_test, y_pred, average='weighted')
print(f'F1 Score: {f1}')
cv_scores = cross_val_score(model, X, y, cv=10)

print(f'Cross Validation Scores: {cv_scores}')
print(f'Average CV Score: {cv_scores.mean()}')
# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [2, 4],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Create a base model
rf = RandomForestClassifier()

# Define a function to fit the model with a specific set of hyperparameters
def fit_model(params):
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return score, params

# Create a list of all possible combinations of hyperparameters
param_list = list(ParameterGrid(param_grid))

# Fit the model with all combinations of hyperparameters and track progress with tqdm
results = Parallel(n_jobs=-1, verbose=1)(
    delayed(fit_model)(params) for params in tqdm(param_list)
)

# Find the best hyperparameters based on the test score
best_score, best_params = max(results, key=lambda x: x[0])

print(f"Best parameters: {best_params}")
print(f"Best score: {best_score}")


# Save the model to a file
joblib.dump(model, 'model.pkl')

print("Model dumped!")
# Load the model from the file
best_estimator_from_joblib = joblib.load('model.pkl')
features_used_in_model = df.drop('casualty_severity', axis=1).columns.tolist()

# Get coefficients from the model
coefficients = best_estimator_from_joblib.coef_[0]

# Now use 'features_used_in_model' in your DataFrame creation
coefficients_df = pd.DataFrame({
    'Feature': features_used_in_model,
    'Coefficient': coefficients
})

# Sort the DataFrame by absolute value of coefficients
coefficients_df = coefficients_df.reindex(coefficients_df.Coefficient.abs().sort_values(ascending=False).index)

print(coefficients_df)

