import torch
import numpy as np
from tqdm import tqdm

def fgsm_attack(image,epsilon,data_grad):

    sign_data_grad = data_grad.sign()
    perturbed_image = image - epsilon*sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def backprob_search(model, pred_func, X_train, val_data):
    
    assert isinstance(model, torch.nn.Module)
    
    x_opt = []
    x_bound,y_bound = torch.Tensor(X_train.min(axis=0)),torch.Tensor(X_train.max(axis=0))
    x_bound = x_bound.to(model.device)
    y_bound = y_bound.to(model.device)

    model.eval()
    for x,y in tqdm(val_data):

        x = x.to(model.device)
        y = y.to(model.device)

        y_opt = y.detach()
        # init_x = torch.randn(x.shape, device='cuda',requires_grad=True)
        init_x = torch.Tensor(x.mean(axis=0).repeat(x.shape[0],1)).to(model.device)
        init_x.requires_grad = True
        optimizer = torch.optim.Adam([init_x], lr=0.1)

        y_min = 1e6
        x_min = None
        for param in model.parameters():
            param.requires_grad = False

        for i in range(10000):
            optimizer.zero_grad()
            output = model(init_x)

            loss = torch.mean((output - y_opt)**2)
            
            loss.backward()
            optimizer.step()
            # init_x = torch.clamp(init_x, x_bound, y_bound)

            if loss.item() < y_min:
                # print(loss.item())
                y_min = loss.item()
                x_min = init_x
                # print(x_min.shape)
            

        x_opt.append(x_min.detach().cpu().numpy())


    x_opt = np.concatenate(x_opt, axis=0)
    # print(x_opt.shape)
    return x_opt

        