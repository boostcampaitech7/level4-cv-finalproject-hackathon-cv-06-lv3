from deap import base, creator, tools
import numpy as np
import random

import faiss
from deap import algorithms

from tqdm import tqdm
import contextlib
import os
import fireducks.pandas as pd

from src.search.ga_func import cx_simulated_binary_w_cx_uniform,\
                    mutGaussian_mutUniformInt,\
                    k_means_selection,\
                    lexicographic_selection 


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

    # nominal 변수 여부 결정 
    is_norminal = [False]*len(control_var_names)
    for i, key in enumerate(control_var_names):
        if type(scalers[key]).__name__ == 'LabelEncoder':
            is_norminal[i] = True
    print("is_norminal",is_norminal)

    # 제어 변수 인덱스 찾기 
    control_set = set(control_var_names)  
    control_index = [i for i, v in enumerate(all_var_names) if v in control_set] # var 중에 control
    
    # pop idx : control variable만 0부터 재정렬한 idx 
    control_index_to_pop_idx = {v: i for i, v in enumerate(control_index)}
    
    # control idx 기준 importance 순서로 정렬     
    sorted_control_index_by_importance = sorted([i for i in control_index if all_var_names[i] in importance.keys()], key=lambda x: importance[all_var_names[x]])
    
    # poppulation index 기준 importance 순서로 정렬 
    sorted_pop_idx_by_importance = [control_index_to_pop_idx[i] for i in sorted_control_index_by_importance] 
    
    # optimize (minimize, maximize)를 importance 순서로 정렬 
    sorted_optimize_dict_by_importance = {all_var_names[k]: optmize_dict[all_var_names[k]] for k in sorted_control_index_by_importance}

    # population 변수 인덱스와 optimize 매핑  
    pop_index_to_optimize = {control_index_to_pop_idx[i]: optmize_dict[all_var_names[i]] for i in sorted_control_index_by_importance}

    # round 처리 이유 : sorting으로 중요도를 고려하기 때문에,
    #                  중요도 높은 변수의 비교를 완화하여여 
    #                  중요도 낮은 변수를 고려하기 위해
    #                  
    # 연속형 변수 소수점 자리 결정 
    scale_factor_x = np.std(X_train, axis=0)
    rounding_digits_x = np.clip(np.ceil(-np.log10(scale_factor_x/100)), 2, 10).astype(int)[sorted_control_index_by_importance]
    
    # target 변수 소수점 자리 결정 
    scale_factor_y = np.std(y_test, axis=0)
    rounding_digits_y = np.clip(np.ceil(-np.log10(scale_factor_y/100)), 2, 10).astype(int)

    # 소수점 자리 결정 
    rounding_digits = np.concatenate([rounding_digits_y,rounding_digits_x], axis=0)

    vectorized_round = np.vectorize(round)

    # bounds 적용 
    if bounds:
        x_min = np.array([value[0] for key, value in bounds.items()]).reshape(-1)
        x_max = np.array([value[1] for key, value in bounds.items()]).reshape(-1)
    else:
        x_min,x_max = np.min(X_train, axis=0)[control_index], np.max(X_train, axis=0)[control_index]
        
    print("x_min : ",x_min)
    print("x_max : ",x_max)
    n_features = X_train.shape[1]

    #  minimize, maximize를 고려한 weight 생성 
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

    # 개체 생성
    # 범주형 변수 여부 고려
    # 최적화 변수 고려
    # 초기값 생성 
    def generate_individual(is_nominal,pop_index_to_optimize):
        is_nominal = np.array(is_nominal, dtype=bool)

        individual = np.where(
        is_nominal,
        np.random.randint(x_min, x_max + 1),  # 범주형이면 randint 사용
        np.random.uniform(x_min, x_max)       # 연속형이면 uniform 사용
        )
        # min max optimize 고려 해서 초기값 생성 보류 
        scale_factor = (x_max - x_min) / 3*5  # 범위 조절 (x_max - x_min)/3 -> 분모 범위내에서 샘플 95% 확률로 포함 
        for i in pop_index_to_optimize.keys():
            if pop_index_to_optimize[i] == 'maximize':
                individual[i] = x_max[i] - np.random.exponential(scale_factor[i])
            else:
                individual[i] = x_min[i] + np.random.exponential(scale_factor[i])
            
            individual[i] = np.clip(individual[i], x_min[i], x_max[i])

        return individual

    # 개체 생성 
    toolbox = base.Toolbox()
    toolbox.register('attr_float', generate_individual, is_nominal=is_norminal, pop_index_to_optimize=pop_index_to_optimize)
    # min_max 차원이 8개이기에 n을 1로 설정 하면 8개의 변수를 가진 ind 생성!
    toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    # 결과 저장 Dict  
    res = {}
    for control_var in control_var_names:
        res[f"pred_x_{control_var}"] = []

    for idx, (gt_x, gt_y) in tqdm(enumerate(zip(X_test, y_test))):

        # 개체 평가 
        def fitness(population):

            population = np.array(population)
            input_data = np.array(gt_x).reshape(1,-1).repeat(len(population), axis=0)
            input_data[:,control_index] = population
            y_pred = pred_func(model=model, X_test=input_data)

            fit_res = []
            # target 변수 평가
            target_fit = -(y_pred - user_request_target.reshape(1,-1))**2
            fit_res.append(target_fit)

            # 중요도 고려 최적화 변수 평가 (minimize, maximize)
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
        
        # 돌연변이 방법 
        mu = [0.0]*(len(x_min))
        for i in range(len(is_norminal)):
            if is_norminal[i]:
                mu[i] = x_min[i]
                sigma_list[i] = x_max[i]
        mu = np.array(mu)
        sigma_list = np.array(sigma_list)
        toolbox.register('mutate', mutGaussian_mutUniformInt, mu=mu, sigma=sigma_list,\
                          indpb=INDPB, is_nominal=is_norminal)

        # 유전 알고리즘 세대 반복 
        for gen in range(1,100):    
            # Crossover, Mutation 
            offspring = algorithms.varAnd(population, toolbox, cxpb, mutpb)

            # 범위 제한 
            offspring = [creator.Individual(np.clip(np.array(ind), x_min, x_max)) for ind in offspring]
            population = offspring+population

            # 평가 
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            fitness_scores = toolbox.evaluate(invalid_ind)
            for ind, fit in zip(invalid_ind, fitness_scores):
                ind.fitness.values = tuple(fit)

            population = k_means_selection(population, k=len(population)//3)

        # 중요도, 최적화 변수 고려 선택 
        population = lexicographic_selection(population, k=1)

        # 결과 저장 
        for i in range(len(control_index)):
            if is_norminal[i]:
                res[f"pred_x_{control_var_names[i]}"].append(int(population[0][i]))
            else:
                res[f"pred_x_{control_var_names[i]}"].append(float(population[0][i]))
    
    
    return pd.DataFrame(res)