import lightgbm as lgb


def lightgbm_load_data(X_train, X_test, y_train, y_test):
    
    train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    val_data = lgb.Dataset(X_test, label=y_test, reference=train_data,free_raw_data=False)
 
    return train_data, val_data

