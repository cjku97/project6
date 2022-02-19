import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
"""
Write your logreg unit tests here. Some examples of tests we will be looking for include:
* check that fit appropriately trains model & weights get updated
* check that predict is working

More details on potential tests below, these are not exhaustive
"""

def test_updates(): 
    # Simulating a dataset
	num_points = 1000
	# Chose values for params (w vector) where w[-1] is bias term
	w = [0.8, 2.6, 1]
	# Generate example dataset and dependent variable y
	X = np.random.rand(num_points, len(w)-1)
	y = (w[-1] + np.expand_dims(X.dot(w[:-1]), 1) + np.random.rand(num_points, 1)*0.1).flatten()
	y = (y > np.mean(y)) * 1
	# Split into training and validation sets
	split = int(0.6*num_points)
	X_train = X[:split]
	X_val = X[split:]
	y_train = y[:split]
	y_val = y[split:]
	# Generate LogisticRegression class instance
	log_model = logreg.LogisticRegression(num_feats=len(w)-1, max_iter=10, tol=1e-6, learning_rate=0.001, batch_size=2)
	
	# Check that predict returns values between 0 and 1
	model_pred = log_model.make_prediction(X)
	assert min(model_pred) >= 0
	assert max(model_pred) <= 1
	
	# Check that your gradient is right shape
	grad = log_model.calculate_gradient(X,y)
	assert grad.shape == (len(w)-1,)

def test_predict():
	# load data with default settings
    X_train, X_val, y_train, y_val = utils.loadDataset(features=['Penicillin V Potassium 500 MG', 'Computed tomography of chest and abdomen', 
                                    'Plain chest X-ray (procedure)',  'Low Density Lipoprotein Cholesterol', 
                                    'Creatinine', 'AGE_DIAGNOSIS'], split_percent=0.8, split_state=42)

    # scale data since values vary across features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform (X_val)
    
    # Generate LogisticRegression class instance
    log_model = logreg.LogisticRegression(num_feats=6, max_iter=20, tol=1e-6, learning_rate=0.001, batch_size=12)
    # Train model
    log_model.train_model(X_train, y_train, X_val, y_val)
    
    # Check that loss decreases
    assert log_model.loss_history_val[0] > log_model.loss_history_val[-1]
    
    # Evaluate
    model_prob = log_model.make_prediction(X_val)
    model_pred = (model_prob >= 0.5) * 1
    tn, fp, fn, tp = confusion_matrix(y_val, model_pred).ravel()
    auc_score = roc_auc_score(y_val, model_pred)
    fpr, tpr, thresholds = roc_curve(y_val, model_prob)
    
    # Check that accuracy is acceptable
    assert (tp + tn)/(tp + fp + tn + fn) > 0.8
    # Check that AUC is acceptable
    assert auc_score > 0.8
    # Check that precision and recall are acceptable
    assert tp/(tp + fp) > 0.8
    assert tp/(tp + fn) > 0.8
	
	
	