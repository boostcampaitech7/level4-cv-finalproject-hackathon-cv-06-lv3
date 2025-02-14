import torch
import numpy as np



class SimpleNN_dataloader(torch.utils.data.Dataset):

    def __init__(self,X,y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        
        X = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor([self.y[idx]], dtype=torch.float32)
        # y = torch.log(y)
        return X, y


def simpleNN_load_data(X_train,X_test,y_train,y_test):

    train_data = SimpleNN_dataloader(X_train,y_train)
    test_data = SimpleNN_dataloader(X_test,y_test)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=10, shuffle=False)
    return train_loader, test_loader