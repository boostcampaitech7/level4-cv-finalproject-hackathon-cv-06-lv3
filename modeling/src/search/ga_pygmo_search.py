# import numpy as np
# import torch

# import pygmo as pg
# from tqdm import tqdm


# def ga_pygmo_search(model, pred_func, X_train, X_test, y_test):
#     """
#     PyGMO를 이용한 유전 알고리즘(Genetic Algorithm, GA) 기반 탐색 함수

#     Args:
#         model (torch.nn.Module): PyTorch 모델
#         pred_func (function): 예측을 수행하는 함수
#         X_train (numpy.ndarray): 훈련 데이터
#         X_test (numpy.ndarray): 테스트 데이터
#         y_test (numpy.ndarray): 테스트 데이터의 실제 값

#     Returns:
#         numpy.ndarray: 최적화된 입력 데이터 배열
#     """
#     test = X_test
#     gt_ys = y_test

#     # 입력 데이터의 최소, 최대 범위 계산
#     x_min = np.min(X_train, axis=0)
#     x_max = np.max(X_train, axis=0)

#     class SphereProblem:
#         """
#         PyGMO 최적화 문제 정의 클래스
#         """

#         def __init__(self, model, gt_y, x_min, x_max):
#             self.model = model
#             self.gt_y = gt_y
#             self.x_min = x_min
#             self.x_max = x_max

#         def fitness(self, x):
#             """
#             개체(입력 데이터)에 대한 손실 함수 평가
#             """
#             x_tensor = (
#                 torch.tensor(x, dtype=torch.float32).unsqueeze(0).to("cuda")
#             )  # 배치 차원 추가

#             with torch.no_grad():
#                 y_pred = self.model(x_tensor)
#                 y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)

#             # 손실 함수: 예측값과 실제값의 차이를 제곱 후 음수화 (최대화 문제로 변환)
#             fit_fun = -((y_pred_tensor.item() - self.gt_y) ** 2)
#             return [fit_fun]

#         def get_bounds(self):
#             """
#             검색 공간의 경계를 반환
#             """
#             return (self.x_min.tolist(), self.x_max.tolist())

#     def batch_evaluate_population(model, pop, gt_y, pred_func):
#         """
#         현재 개체군의 모든 개체를 일괄 평가하는 함수

#         Args:
#             model (torch.nn.Module): PyTorch 모델
#             pop (pg.population): 현재 개체군
#             gt_y (float): 목표 예측값
#             pred_func (function): 예측 수행 함수

#         Returns:
#             pg.population: 업데이트된 개체군
#         """
#         x_tensors = (
#             torch.tensor(pop.get_x(), dtype=torch.float32).reshape(-1, 8).to("cuda")
#         )

#         with torch.no_grad():
#             y_preds = pred_func(model=model, X_test=x_tensors)
#             y_pred_tensor = torch.tensor(y_preds, dtype=torch.float32)

#         # 손실 함수 계산 (최대화 문제로 변환)
#         fitness_values = -((y_pred_tensor - gt_y) ** 2).cpu().numpy()

#         # 새로운 개체군 생성
#         new_pop = pg.population(pop.problem)
#         for i, fit_val in enumerate(fitness_values):
#             individual = np.array(pop.get_x()[i]).flatten()
#             fit_val = float(fit_val)
#             new_pop.push_back(individual, [fit_val])

#         return new_pop

#     res = []  # 최적화된 결과를 저장할 리스트

#     for gt_y in tqdm(gt_ys):
#         """
#         각 목표값(gt_y)에 대해 최적화 수행
#         """
#         prob = pg.problem(SphereProblem(model, gt_y, x_min, x_max))

#         # 유전 알고리즘 설정 (세대 수: 100, 교차 확률: 0.7, 변이 확률: 0.2)
#         algo = pg.algorithm(pg.sga(gen=100, cr=0.7, eta_c=1.0, m=0.2, param_m=1.0))

#         # 초기 개체군 생성 (크기: 50)
#         pop = pg.population(prob, size=50)

#         # 개체군을 평가하여 업데이트
#         pop = batch_evaluate_population(model, pop, gt_y, pred_func)

#         # 최적의 개체 선택
#         best_individual = pop.champion_x

#         x_pred = np.array(best_individual).reshape(1, 8)
#         res.append(x_pred)

#     return np.concatenate(res, axis=0)
