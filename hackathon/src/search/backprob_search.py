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
    # X_train = X_train.to_numpy()
    
    x_opts = []
    min_bound, max_bound = torch.Tensor(X_train.min(axis=0)), torch.Tensor(X_train.max(axis=0))
    min_bound = min_bound.to(model.device)
    max_bound = max_bound.to(model.device)
    y_std = val_data.dataset.y.std()

    model.eval()
    for i in range(X_train.shape[1]):
        x_opt = []
        print(f'column {i}')
        for x, y in tqdm(val_data):
            x = x.to(model.device)
            y = y.to(model.device)
            y_opt = y.detach()
            
            specific_axes = [k for k in range(x.shape[1]) if k != i]
            init_x = torch.Tensor(x.mean(axis=0).repeat(x.shape[0], 1)).to(model.device)
            init_x[:, specific_axes] = x[:, specific_axes]
            
            init_x.requires_grad = True
            optimizer = torch.optim.Adam([init_x], lr=0.1)

            y_min = None
            x_min = None
            
            for param in model.parameters():
                param.requires_grad = False

            for i in range(10000):
                init_x.data = torch.clamp(init_x.data, min_bound, max_bound)
                optimizer.zero_grad()
                output = model(init_x)
                
                diff = (output - y_opt) ** 2
                mask = diff * (y_std ** 2) > 0.005

                if i > 0 and not mask.any():
                    print(i)
                    break 

                loss = torch.mean(diff[mask]) if mask.any() else torch.mean(diff)
                
                loss.backward()
                init_x.grad[:, specific_axes] = 0
                optimizer.step()
                # init_x.data[:, specific_axes] = x[:, specific_axes]

                if y_min is None or loss.item() < y_min:
                    y_min = loss.item()
                    x_min = init_x.clone().detach()

            x_opt.append(x_min.detach().cpu().numpy())
        x_opts.append(np.concatenate(x_opt, axis=0))

    x_opts = np.stack(x_opts)
    print(x_opts.shape)
    return x_opts


