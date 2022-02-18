import numpy as np
import pandas as pd
from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler

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

    log_model = logreg.LogisticRegression(num_feats=6, max_iter=20, tol=1e-6, learning_rate=0.001, batch_size=12)
    log_model.train_model(X_train, y_train, X_val, y_val)
    log_model.plot_loss_history()
    print("done")
    print(f"Learned Params: w --> {log_model.W}")
    model_pred = log_model.make_prediction(X_val)
    print(model_pred[0:10]) # should these be exactly 0 or 1 right? or just between 0 and 1?
    print(y_val[0:10])


if __name__ == "__main__":
    main()
