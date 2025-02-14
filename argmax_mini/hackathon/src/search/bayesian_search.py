from bayes_opt import BayesianOptimization
import numpy as np

def objective(model,predict_func,x,target):

    pred = predict_func(model,x)[0]
    
    return -abs(pred-target)


def bayesian_search(model,predict_func,x_train,y_train,y_test):

    
    min_vals = x_train.min(axis=0)
    max_vals = x_train.max(axis=0)
    mean_vals = x_train.mean(axis=0)

    pbounds = {f'x{i}': (min_vals[i], max_vals[i]) for i in range(x_train.shape[1])}

    for target,control in zip(y_test,y_train):

        def wrapped_objective(**kwargs):
            x_array = np.array([kwargs[f'x{i}'] for i in range(len(kwargs))])
            x_array = x_array.reshape(1,-1)
            # print(x_array.shape)
            # print(target.shape)
            return objective(model, predict_func, x_array, target)

        optimizer = BayesianOptimization(
            f=wrapped_objective,
            pbounds=pbounds,
            random_state=42,
            verbose=2
        )

        optimizer.maximize(
            init_points=10,  # 초기 랜덤 포인트 수
            n_iter=30,       # 최적화 반복 횟수
        )

        # optimizer.max - control
        # optimizer.max['params']

        print(optimizer.max['params'])
        break # TODO 최적화 오래걸림...

    return optimizer.max