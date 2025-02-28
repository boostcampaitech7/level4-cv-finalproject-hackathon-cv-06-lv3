import random

import numpy as np
from deap import base, creator, tools
from tqdm import tqdm


def adaptive_niche_size(gen, max_gen, initial_sigma, min_sigma, decay_constant=5.0):
    """
    세대에 따라 적응적으로 niche size를 조정하는 함수

    Args:
        gen (int): 현재 세대 (generation)
        max_gen (int): 전체 세대 수 (maximum number of generations)
        initial_sigma (float): 초기 niche size 값
        min_sigma (float): 최소 niche size 값 (sigma가 이 값 이하로 감소하지 않음)
        decay_constant (float): niche size 감소 속도를 조절하는 상수 (기본값: 5.0)

    Returns:
        (float) 현재 세대에 해당하는 niche size 값
    """

    # 지수적 감소를 적용하여 niche size 계산
    sigma = initial_sigma * np.exp(-decay_constant * gen / max_gen)

    # 최소 niche size보다 작아지지 않도록 제한
    return max(sigma, min_sigma)


def fitness_sharing(population, sigma, alpha):
    """
    적용된 적합도 공유(Fitness Sharing) 방법을 사용하여 개체들의 적합도를 조정합니다.

    Args:
        population (list): 개체 리스트 (각 개체는 .fitness.values 속성을 가져야 함)
        sigma (float): 공유 거리 임계값 (개체 간 거리 비교 기준)
        alpha (float): 공유 강도 계수
    """
    # 개체 리스트를 하나의 배열로 병합
    population_array = np.concatenate(population, axis=0)

    # 모든 개체 간의 유클리드 거리 행렬 계산
    distances = np.linalg.norm(
        population_array[:, np.newaxis, :] - population_array[np.newaxis, :, :], axis=2
    )

    # 공유 함수 적용: 거리가 sigma 미만인 경우 공유 값을 계산, 그렇지 않으면 0
    sh_values = np.where(distances < sigma, 1 - (distances / sigma) ** alpha, 0)

    # 각 개체별 공유 계수 계산
    sharing_factors = np.sum(sh_values, axis=1)

    # 개체들의 적합도를 공유 계수로 나누어 조정
    for ind, sharing_factor in zip(population, sharing_factors):
        if sharing_factor > 0.0:
            adjusted_sharing_factor = max(
                sharing_factor, 1e-6
            )  # 0으로 나누는 문제 방지
            ind.fitness.values = (ind.fitness.values[0] / adjusted_sharing_factor,)


def ga_adaptive_niching_search(
    model,
    pred_func,
    X_train,
    X_test,
    y_test,
    max_gen=100,
    initial_sigma=2.5,
    min_sigma=0.5,
    decay_constant=2.0,
):
    """
    유전자 알고리즘 기반의 적응형 니칭 검색을 수행하는 함수

    Args:
        model: 예측을 수행할 모델
        pred_func: 예측 함수
        X_train (np.array): 훈련 데이터
        X_test (np.array): 테스트 데이터
        y_test (np.array): 테스트 데이터의 정답값
        max_gen (int): 최대 세대 수 (default: 100)
        initial_sigma (float): 초기 시그마 값 (default: 2.5)
        min_sigma (float): 최소 시그마 값 (default: 0.5)
        decay_constant (float): 시그마 감소 계수 (default: 2.0)

    Returns:
        np.array: 최적의 예측값 배열
    """

    test = X_test
    gt_ys = y_test
    n_features = X_train.shape[1]

    # 훈련 데이터의 최소, 최대값 계산
    x_min = np.min(X_train, axis=0)
    x_max = np.max(X_train, axis=0)

    res = []
    for idx, gt_y in tqdm(enumerate(gt_ys), total=len(gt_ys)):

        def fitness(population):
            """개체군의 적합도를 평가하는 함수"""
            population = np.concatenate(population, axis=0)
            y_pred = pred_func(model=model, X_test=population)
            fit_fun = -np.square(
                y_pred - gt_y
            )  # 오차의 음수 값 사용 (최대화 문제이므로)
            return fit_fun

        # DEAP creator를 재정의 (기존 정의 삭제 후 재생성)
        try:
            del creator.FitnessMax
            del creator.Individual
        except:
            pass

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()

        def generate_individual():
            """개체 생성 함수: 평균과 표준편차를 활용한 무작위 샘플링"""
            mean = (x_max + x_min) / 2
            std_dev = (x_max - x_min) / 6
            return np.random.randn(n_features) * std_dev + mean

        # DEAP Toolbox 초기화
        toolbox.register("attr_float", generate_individual)
        toolbox.register(
            "individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=1
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", fitness)
        toolbox.register("select", tools.selTournament, tournsize=2)
        toolbox.register("mate", tools.cxBlend, alpha=0.7)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.6)

        pop_size = 300
        population = toolbox.population(n=pop_size)

        # 초기 개체의 적합도 평가
        invalid_inds = [ind for ind in population if not ind.fitness.valid]
        fitness_scores = toolbox.evaluate(invalid_inds)
        for ind, fit in zip(invalid_inds, fitness_scores):
            ind.fitness.values = (fit,)

        # 진화 과정
        for gen in range(max_gen):
            if len(population) == 1:
                break

            # 부모 선택 및 자손 생성
            parents = toolbox.select(population, k=len(population))
            offspring = list(map(toolbox.clone, parents))

            # 교배 연산
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.3:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # 변이 연산
            for child in offspring:
                if random.random() < 0.5:
                    toolbox.mutate(child)
                    del child.fitness.values

            # 값 범위 제한
            offspring = [
                creator.Individual(np.clip(np.array(ind), x_min, x_max))
                for ind in offspring
            ]

            # 적합도 재계산
            invalid_offspring = [ind for ind in offspring if not ind.fitness.valid]
            fit_vals = toolbox.evaluate(invalid_offspring)
            for ind, fv in zip(invalid_offspring, fit_vals):
                ind.fitness.values = (fv,)

            # 부모 + 자손 합쳐서 새로운 세대 선정
            combined = population + offspring
            next_population = toolbox.select(combined, k=pop_size)

            # 적응형 니칭 크기 계산
            sigma = adaptive_niche_size(
                gen, max_gen, initial_sigma, min_sigma, decay_constant
            )

            # 피트니스 공유 적용
            fitness_sharing(next_population, sigma, alpha=1.0)

            population[:] = next_population

        # 최적 개체 선택
        best_individual = tools.selBest(population, k=1)[0]
        best_individual = best_individual[0]
        x_pred = np.array(best_individual).reshape(1, 8)

        res.append(x_pred)

    return np.concatenate(res, axis=0)
