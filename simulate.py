import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

def main():
	
	# Simulating a dataset
	num_points = 8000
	# Chose values for params (w vector) where w[-1] is bias term
	w = [2, 2.6, 0.8, 4.5, 1]

	# Generate example dataset and dependent variable y
	X = np.random.rand(num_points, len(w)-1)
	y = (w[-1] + np.expand_dims(X.dot(w[:-1]), 1) + np.random.rand(num_points, 1)*0.1).flatten()
	# print(y)
	y = (y > np.mean(y)) * 1
	# print(y)

	# Split into training and validation sets
	split = int(0.6*num_points)
	X_train = X[:split]
	X_val = X[split:]
	y_train = y[:split]
	y_val = y[split:]

	# Generate LogisticRegression class instance
	log_model = logreg.LogisticRegression(num_feats=len(w)-1, max_iter=10, tol=0.0001, learning_rate=0.01, batch_size=10)

	# Train model
	log_model.train_model(X_train, y_train, X_val, y_val)
	
	# Evaluate
	log_model.plot_loss_history()
	print(f"Ground Truth: w --> {w}")
	print(f"Learned Params: w --> {log_model.W}")
	model_prob = log_model.make_prediction(X_val)
	model_pred = (model_prob >= 0.5) * 1
	print(confusion_matrix(y_val, model_pred))
	print(classification_report(y_val, model_pred))
	
	# ROC Curve
	logit_roc_auc = roc_auc_score(y_val, model_pred)
	fpr, tpr, thresholds = roc_curve(y_val, model_prob)
	plt.figure()
	plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="lower right")
	plt.savefig('Simulate_Log_ROC')
	plt.show()

	


if __name__ == "__main__":
    main()
