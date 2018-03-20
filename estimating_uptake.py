import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline, make_pipeline

from feature_engineering import *

if __name__ == "__main__":
    feature_target_file = '../output/bank_all_features_target.csv'
    full_df = pd.read_csv(feature_target_file)
    
    # Extract features and target
    features_df, target = extract_features_target(full_df)
    
    # Train test split
    RANDOM_STATE = 42
    X_train, X_test, y_train, y_test = train_test_split(features_df.values, target, random_state=RANDOM_STATE, test_size=0.2)
    
    # Pipeline for logistic regression
    # Find optimal parameter C
    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    param_grid = {'logisticregression__C': [0.01, 0.1, 1, 10, 100]}
    grid = GridSearchCV(pipe, param_grid, cv=5)
    grid.fit(X_train, y_train)
    print("Best estimator: \n{}".format(grid.best_estimator_))
    print(classification_report(y_test, grid.predict(X_test), target_names=["not subscribed", "subscribed"]))
    
    # Random Forest: Parameters found by GridSearchCV (see Model-Tuning.ipynb)
    rf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight={0: 1, 1:3}, max_depth=10, max_features=50, n_estimators=50)
     
    # Parameter tuning for Gradient boosting
    gbrt = GradientBoostingClassifier(random_state=RANDOM_STATE, max_depth=5, n_estimators=200)
    gbrt.fit(X_train, y_train)
    print("Classification Report for GradientBoostingClassifier")
    print(classification_report(y_test, gbrt.predict(X_test), target_names=["not subscribed", "subscribed"]))
    
    
    
    
    
    
    
    