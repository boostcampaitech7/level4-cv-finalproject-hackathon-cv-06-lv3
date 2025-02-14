import numpy as np

def eval_search_model(x_train, x_opt, x_test):

    print(x_opt[0])
    print(x_test[0])
    print(x_train.mean(axis=0))
    rmse = np.sqrt(np.mean((x_test - x_opt) ** 2,axis=0))
    mae = np.mean(np.abs(x_test - x_opt),axis=0)
    SSE = np.sum(np.square(x_test - x_opt),axis=0)    
    SST = np.sum(np.square(x_test - x_train.mean(axis=0)),axis=0)

    r2 = 1 - SSE/SST
    return rmse, mae, r2