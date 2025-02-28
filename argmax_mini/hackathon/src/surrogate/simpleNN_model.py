import torch
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import numpy as np

class simpleNN_model(torch.nn.Module):
    def __init__(self,input_size,output_size=1):
        super(simpleNN_model,self).__init__()
        self.fc1 = torch.nn.Linear(input_size,16)
        self.fc2 = torch.nn.Linear(16,32)
        self.fc3 = torch.nn.Linear(32,output_size)

    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.device = next(self.parameters()).device  # device 속성 자동 설정
        return self


def simpleNN_train(train_loader,val_loader):

    model = simpleNN_model(input_size=train_loader.dataset.X.shape[1])
    model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs:=200):
        train_loss = 0
        val_loss = 0

        for data,target in train_loader:
            optimizer.zero_grad()
            data = data.to(model.device)
            target = target.to(model.device)
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {train_loss/len(train_loader):.4f}")

            for data,target in val_loader:
                data = data.to(model.device)
                target = target.to(model.device)
                output = model(data)
                loss = loss_fn(output, target)
                val_loss += loss.item()
            print(f"Validation Loss: {val_loss/len(val_loader):.4f}")

    return model

def simpleNN_evaluate(model,train_loader,test_loader):
    pass
    # y_mean = train_loader.dataset.y.mean()
    
    # model.eval()
    # # loss_fn = torch.nn.MSELoss()

    # with torch.no_grad():
    #     SSE = 0
    #     SST = 0
    #     for data,target in test_loader:
    #         data = data.cuda()
    #         target = target.cuda()
    #         output = model(data)
    #         output = output.numpy()
    #         target = target.numpy()
    #         # print(output.shape)
    #         # print(target.shape)
    #         SSE += np.sum((target - output)**2)
    #         SST += np.sum((target - y_mean)**2)
    #         # print(((target - output).shape))
    #         # print(((target - y_mean).shape))
        
    #     # print(SSE)
    #     # print(SST)

    #     r2 = 1 - SSE/SST
    #     rmse = np.sqrt(SSE/len(test_loader))
    #     mae = np.mean(np.abs(target - output))

    # return rmse, mae, r2

def simpleNN_predict(model,X_test) -> np.ndarray:
    model.eval()
    if isinstance(X_test, torch.Tensor):
        with torch.no_grad():
            X_test = X_test.to(model.device)
            output = model(X_test)
        # return output.numpy()
        return output.cpu().numpy()  # GPU에서 CPU로 복사 후 numpy로 변환
    elif isinstance(X_test, torch.utils.data.DataLoader):
        y_pred = []
        for data,target in X_test:
            with torch.no_grad():
                data = data.to(model.device)
                output = model(data)
                output = output.detach().cpu().numpy()
                # target = target.numpy()
                y_pred.append(output)
        y_pred = np.concatenate(y_pred, axis=0).squeeze()
        return y_pred
    elif isinstance(X_test, np.ndarray):
        with torch.no_grad():
            X_test = torch.tensor(X_test, dtype=torch.float32).to(model.device)
            output = model(X_test)
        return output.cpu().numpy()
