import numpy as np

def eval_search_model(x_train, x_opt, x_test):

    if x_opt.ndim == 2:
        print(x_opt[0])
        print(x_test[0])
        print(x_train.mean(axis=0))
        rmse = np.sqrt(np.mean((x_test - x_opt) ** 2,axis=0))
        mae = np.mean(np.abs(x_test - x_opt),axis=0)
        SSE = np.sum(np.square(x_test - x_opt),axis=0)    
        SST = np.sum(np.square(x_test - x_train.mean(axis=0)),axis=0)
        r2 = 1 - SSE/SST
    if x_opt.ndim == 3:
        r2 = []
        rmse = []
        mae = []
        for i in range(x_opt.shape[0]):
            rmse.append(np.sqrt(np.mean((x_test - x_opt[i]) ** 2,axis=0)))
            mae.append(np.mean(np.abs(x_test - x_opt[i]),axis=0))
            SSE = np.sum(np.square(x_test - x_opt[i]),axis=0)    
            SST = np.sum(np.square(x_test - x_train.mean(axis=0)),axis=0)
            r2.append((1 - SSE/SST)[i])
        r2 = np.array(r2)
        rmse = np.array(rmse)
        mae = np.array(mae)
    return rmse, mae, r2
