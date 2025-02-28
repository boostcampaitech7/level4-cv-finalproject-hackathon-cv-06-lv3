import random

import numpy as np
import torch
from tqdm import tqdm
from deap import base, creator, tools


def ga_deap_search(model, pred_func, X_train, X_test, y_test, max_gen=10):
    """
    Genetic Algorithm (GA) 기반 탐색 함수

    Args:
        model (object): 예측 모델
        pred_func (function): 예측 함수
        X_train (numpy.ndarray): 학습 데이터
        X_test (numpy.ndarray): 테스트 데이터
        y_test (numpy.ndarray): 테스트 라벨
        max_gen (int): 최대 세대 수 (기본값: 10)

    Returns:
        numpy.ndarray: 최적화된 입력 값 리스트
    """
    test = X_test
    gt_ys = y_test

    # 입력 데이터의 최소 및 최대 값 계산
    x_min = np.min(X_train, axis=0)
    x_max = np.max(X_train, axis=0)

    res = []
    for idx, gt_y in tqdm(enumerate(gt_ys), desc="GA Optimization Loop"):

        def fitness(population):
            """
            적합도 함수 정의

            Args:
                population (list): 개체군 리스트

            Returns:
                numpy.ndarray: 적합도 값
            """
            population = np.concatenate(population, axis=0)
            y_pred = pred_func(model=model, X_test=population)
            y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)
            return -((y_pred_tensor - gt_y) ** 2)

        # DEAP 객체 생성 (FitnessMax 및 Individual)
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.uniform, x_min, x_max)
        toolbox.register(
            "individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=1
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # 진화 연산 등록
        toolbox.register("evaluate", fitness)
        toolbox.register("select", tools.selBest, k=5)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)

        # 초기 개체군 생성
        pop_size = 50
        population = toolbox.population(n=pop_size)

        for gen in range(max_gen):
            # 개체군 평가
            fitness_scores = toolbox.evaluate(population)
            for ind, fit in zip(population, fitness_scores):
                ind.fitness.values = (fit,)

            if len(population) == 1:
                break

            # 부모 선택 및 복제
            parents = toolbox.select(population, k=len(population))
            offspring = tools.selBest(parents, k=len(population))
            offspring = list(map(toolbox.clone, offspring))

            # 교배 수행 (확률: 70%)
            for i in range(1, len(offspring), 2):
                if random.random() < 0.7:
                    toolbox.mate(offspring[i - 1], offspring[i])

            # 돌연변이 적용 (확률: 20%)
            for child in offspring:
                if random.random() < 0.2:
                    toolbox.mutate(child)

            # 개체의 적합도 초기화 (재계산 필요)
            for ind in offspring:
                del ind.fitness.values

            # 자식 개체를 새로운 개체군으로 설정
            population[:] = offspring

        # 최적 개체 선택 및 변환
        all_individual = tools.selBest(population, k=len(population))
        all_individual = np.array(all_individual)
        squeezed_all_individual = all_individual.squeeze(axis=1)

        # GT 데이터와 거리 계산
        gt_x = test[idx]
        expanded_gt_x = np.tile(gt_x, (50, 1))
        differences = squeezed_all_individual - expanded_gt_x
        distances = np.linalg.norm(differences, axis=1)
        res_idx = np.argmin(distances)
        x_pred = all_individual[res_idx]

        res.append(x_pred)

    return np.concatenate(res, axis=0)
