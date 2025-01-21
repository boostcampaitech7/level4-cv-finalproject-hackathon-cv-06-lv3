import torch
from tabpfn import TabPFNRegressor

def tabpfn_train(train_data, val_data):

    X_train, y_train = train_data
    X_test, y_test = val_data

    model = TabPFNRegressor(device='cpu')
    model.fit(X_train, y_train)

    return model

def tabpfn_predict(model, val_data):

    X_test, y_test = val_data
    y_pred = model.predict(X_test)
    return y_pred