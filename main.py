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

    # load data with default settings
    X_train, X_val, y_train, y_val = utils.loadDataset(features=['Penicillin V Potassium 500 MG', 'Computed tomography of chest and abdomen', 
                                    'Plain chest X-ray (procedure)',  'Low Density Lipoprotein Cholesterol', 
                                    'Creatinine', 'AGE_DIAGNOSIS'], split_percent=0.8, split_state=42)

    # scale data since values vary across features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform (X_val)
    print(X_train.shape, X_val.shape, y_val.shape, y_train.shape)

    
    # for testing purposes once you've added your code
    # CAUTION: hyperparameters have not been optimized
	
	# Generate LogisticRegression class instance
    log_model = logreg.LogisticRegression(num_feats=6, max_iter=20, tol=1e-6, learning_rate=0.001, batch_size=12)
    
    # Train model
    log_model.train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate
    log_model.plot_loss_history()
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
    # plt.savefig('Log_ROC')
    plt.show()


if __name__ == "__main__":
    main()
