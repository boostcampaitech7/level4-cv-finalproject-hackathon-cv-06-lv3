from deap import base, creator, tools
import numpy as np
import random

import faiss
from deap import algorithms

from tqdm import tqdm
import contextlib
import os
import fireducks.pandas as pd
from hackathon.src.search.ga_function import mutGaussian_mutUniformInt\
                                            ,cx_simulated_binary_w_cx_uniform\
                                            ,k_means_selection\
                                            ,lexicographic_selection        


def k_means_search_deploy(model, pred_func, X_train, X_test, y_test,\
                          all_var_names, control_var_names, optmize_dict, importance,\
                            bounds, scalers, user_request_target):
    """
    # all_var_names : target 변수 제외 모든 변수 이름 [numpy X와 같은 순서]
    # control_var_names : control 변수 이름 
    
    ## len(control_var_names) > len(optmize_dict) == len(importance) 
    # optmize_dict : minimize, maximize 
    # importance : 중요도 순서 (1 부터 중복 없이 ranking)

    # bounds 
    """
    is_norminal = [False]*len(control_var_names)
    for i, key in enumerate(control_var_names):
        if type(scalers[key]).__name__ == 'LabelEncoder':
            is_norminal[i] = True
    print("is_norminal",is_norminal)

    # 제어 변수 인덱스 결정 및 중요도 순 정렬

    control_set = set(control_var_names)  
    control_index = [i for i, v in enumerate(all_var_names) if v in control_set] # var 중에 control
    
    # pop idx : control variable만 0부터 재정렬한 idx 
    control_index_to_pop_idx = {v: i for i, v in enumerate(control_index)}
    
    # control중 importance 순서    
    sorted_control_index_by_importance = sorted([i for i in control_index if all_var_names[i] in importance.keys()], key=lambda x: importance[all_var_names[x]])
    # poppulation 열 index를 중요도 순서로 정렬 
    sorted_pop_idx_by_importance = [control_index_to_pop_idx[i] for i in sorted_control_index_by_importance]
    # optimize를 importance 순서로 정렬 
    # sorted_optimize_dict_by_vars_idx = {all_var_names[k]: optmize_dict[all_var_names[k]] for k in [i for i in control_index if all_var_names[i] in importance.keys()]}
    sorted_optimize_dict_by_importance = {all_var_names[k]: optmize_dict[all_var_names[k]] for k in sorted_control_index_by_importance}


    # population 변수 인덱스와 optimize 매핑  
    pop_index_to_optimize = {control_index_to_pop_idx[i]: optmize_dict[all_var_names[i]] for i in sorted_control_index_by_importance}
    # print('control_index_to_optimize',control_index_to_optimize)
    print('pop_index_to_optimize',pop_index_to_optimize)
    scale_factor_x = np.std(X_train, axis=0)
    rounding_digits_x = np.clip(np.ceil(-np.log10(scale_factor_x/100)), 2, 10).astype(int)[sorted_control_index_by_importance]
    
    scale_factor_y = np.std(y_test, axis=0)
    rounding_digits_y = np.clip(np.ceil(-np.log10(scale_factor_y/100)), 2, 10).astype(int)

    rounding_digits = np.concatenate([rounding_digits_y,rounding_digits_x], axis=0)

    vectorized_round = np.vectorize(round)

    #TODO bounds 적용 
    if bounds:
        x_min = np.array([value[0] for key, value in bounds.items()]).reshape(-1)
        x_max = np.array([value[1] for key, value in bounds.items()]).reshape(-1)
    else:
        x_min,x_max = np.min(X_train, axis=0)[control_index], np.max(X_train, axis=0)[control_index]
        
    print("x_min : ",x_min)
    print("x_max : ",x_max)
    n_features = X_train.shape[1]
    weights =  (1.0,) * y_test.shape[-1]
    weights += tuple(1.0 if opt == 'maximize' else -1.0 for opt in sorted_optimize_dict_by_importance.values())
    print('weights',weights)
    creator.create('FitnessMax', base.Fitness, weights=weights) # model pred + control optim
    creator.create('Individual', np.ndarray, fitness=creator.FitnessMax)

    # def generate_individual():
    #     # return np.random.uniform(x_min, x_max)
    #     mean = (x_max + x_min) / 2
    #     std_dev = (x_max - x_min) / 6  # 99.7% 확률로 x_min과 x_max 사이에 생성 std 3! 
    #     return np.random.randn(n_features) * std_dev + mean

    def generate_individual(is_nominal, pop_index_to_optimize):
        # is_nominal을 bool 타입의 NumPy 배열로 변환
        is_nominal = np.array(is_nominal, dtype=bool)
        
        # 각 변수에 대해, x_max가 최소 x_min + 1 이상이 되도록 보장하여
        # 제어 변수의 최소값과 최대값이 너무 가까워서 발생하는 오류를 방지함
        local_x_min = x_min.copy()
        local_x_max = np.maximum(x_min + 1, x_max)
        
        # 초기 individual 생성:
        # - 범주형 변수인 경우 randint를 사용 (size를 지정하여 배열 생성)
        # - 연속형 변수인 경우 uniform 분포를 사용하여 난수 생성
        individual = np.where(
            is_nominal,
            np.random.randint(local_x_min, local_x_max + 1, size=len(local_x_min)),
            np.random.uniform(local_x_min, local_x_max)
        )
        
        # 지수 분포 조정을 위한 scale_factor 계산
        # (local_x_max - local_x_min) * (5/3)는 값의 분포를 조절하는 역할을 함
        scale_factor = (local_x_max - local_x_min) * (5 / 3)
        
        # 최적화할 변수들에 대해 각 인덱스별로 값 조정
        for i in pop_index_to_optimize.keys():
            # 인덱스가 individual의 범위를 초과하면 건너뜀
            if i >= len(individual):
                continue
            
            # 'maximize'인 경우: x_max에서 지수 분포 난수를 빼서 값을 조정
            if pop_index_to_optimize[i] == 'maximize':
                adjustment = np.random.exponential(scale_factor[i])
                individual[i] = local_x_max[i] - adjustment
            else:
                # 'minimize'인 경우: x_min에 지수 분포 난수를 더해서 값을 조정
                adjustment = np.random.exponential(scale_factor[i])
                individual[i] = local_x_min[i] + adjustment
            
            # individual의 값이 local_x_min과 local_x_max 사이에 있도록 범위 제한(clip)
            individual[i] = np.clip(individual[i], local_x_min[i], local_x_max[i])
        
        return individual



    toolbox = base.Toolbox()
    toolbox.register('attr_float', generate_individual, is_nominal=is_norminal, pop_index_to_optimize=pop_index_to_optimize)
    # min_max 차원이 8개이기에 n을 1로 설정 하면 8개의 변수를 가진 ind 생성!
    toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    # res = {"pred_x":[]} # , "test_x":[], "test_y":[]}
    res = {}
    for control_var in control_var_names:
        res[f"pred_x_{control_var}"] = []
    for idx, (gt_x, gt_y) in tqdm(enumerate(zip(X_test, y_test))):

        def fitness(population):
            
            population = np.array(population)
            input_data = np.array(gt_x).reshape(1,-1).repeat(len(population), axis=0)
            input_data[:,control_index] = population
            y_pred = pred_func(model=model, X_test=input_data)

            fit_res = []
            # print(user_request_target)
            # fit_res.append(-(y_pred - gt_y)**2)
            # print(-(y_pred - user_request_target)**2)
            # print(y_pred)
            target_fit = -(y_pred - user_request_target.reshape(1,-1))**2
            fit_res.append(target_fit)

            for i in sorted_pop_idx_by_importance:
                    imp_fit = population[:,i:i+1]
                    fit_res.append(imp_fit)

            fit_res = np.concatenate(fit_res, axis=1)
            fit_res = vectorized_round(fit_res, rounding_digits)
            return fit_res

        toolbox.register('evaluate', fitness)
        
        INDPB = 0.2 # 변수별 변이 확률
        cxpb = 0.5 # 교차 확률
        mutpb = 0.5 # 돌연변이 확률
        
        # 선택 방법, 사용하지 않음
        toolbox.register('select', tools.selTournament)
        # 개체 생성 
        population = toolbox.population(n=1000)

        ETA_CX = 2.0
        sigma_list = [(ub - lb)/(6.0) for (lb,ub) in zip(x_min, x_max)]
        # 교차 방법 
        toolbox.register('mate', cx_simulated_binary_w_cx_uniform\
                         , eta=ETA_CX, indpb=INDPB, is_nominal=is_norminal)
        
        mu = [0.0]*(len(x_min))
        for i in range(len(is_norminal)):
            if is_norminal[i]:
                mu[i] = x_min[i]
                sigma_list[i] = x_max[i]
        mu = np.array(mu)
        sigma_list = np.array(sigma_list)

        # 돌연변이 방법
        toolbox.register('mutate', mutGaussian_mutUniformInt, mu=mu, sigma=sigma_list,\
                          indpb=INDPB, is_nominal=is_norminal)

        # 유전 알고리즘 세대 반복 
        for gen in range(1,101):    

            offspring = algorithms.varAnd(population, toolbox, cxpb, mutpb)

            offspring = [creator.Individual(np.clip(np.array(ind), x_min, x_max)) for ind in offspring]
            population = offspring+population

            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            fitness_scores = toolbox.evaluate(invalid_ind)
            for ind, fit in zip(invalid_ind, fitness_scores):
                ind.fitness.values = tuple(fit)
            population = k_means_selection(population, k=len(population)//3)

        # population = tools.selBest(population, k=1)
        population = lexicographic_selection(population, k=1)

        for i in range(len(control_index)):
            if is_norminal[i]:
                res[f"pred_x_{control_var_names[i]}"].append(int(population[0][i]))
            else:
                res[f"pred_x_{control_var_names[i]}"].append(float(population[0][i]))
    
    return pd.DataFrame(res)