import numpy as np


def eval_surrogate_model(y_train,y_pred, y_test):
    # rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    # mae = np.mean(np.abs(y_test - y_pred))
    # SSE = np.sum(np.square(y_test - y_pred))    
    # SST = np.sum(np.square(y_test - y_train.mean()))
    # r2 = 1 - SSE/SST
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2, axis=0)) 
    mae = np.mean(np.abs(y_test - y_pred), axis=0)  
    SSE = np.sum(np.square(y_test - y_pred), axis=0) 
    SST = np.sum(np.square(y_test - np.mean(y_train, axis=0)), axis=0) 
    r2 = 1 - SSE / SST 

    return rmse, mae, r2