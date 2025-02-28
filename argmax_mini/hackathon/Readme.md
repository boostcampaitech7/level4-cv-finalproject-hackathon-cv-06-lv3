# Modeling Goal 
-----
Tabular Data에 대해 특정 output Y값에 대한 input X를 추천하는 자동화된 모델 개발


Surrogate Model 학습과, Surrogate Model을 환경으로 하여 특정 Y값에 대한 최적화된 X를 찾는 Search Model 구현 

# 요구사항 및 구현
----

- 하나의 Y값에 대해 여러 X값이 존재하며, 자동화된 모델링 필요 
	- X로 Y를 예측하는 Surrogate Model과, Surrogate Model을 활용하여 특정 Y로 X를 예측하는 Search Model로 구성


- 데이터 셋이 들어올 때 자동화된 학습 및 10분 이하의 Response Time  고려 
	- Hyperparameter Tuning이 적거나 없는 Surrogate 모델을 선정하여 사용 
		- TabPFN - hyperparameter tuning이 요구되지 않는 foundation model 
		- CatBoost - Optuna를 통해 간단한 hyperparameter tuning 

- 특정 Y에 대한 여러 X를 찾을 수 있어야 함 
	- 탐색(Exploration)과 활용(Exploitation) 중에서 탐색에 초점을 맞춘 모델을 사용
	- Genetic Algorithm을 선정, 니치 보호 기법을 K-means를 사용하여 구현 



# How To Use
----
[jupyter 파일](https://github.com/boostcampaitech7/level4-cv-finalproject-hackathon-cv-06-lv3/blob/user_request_ipynb/argmax_mini/how_to_use_model.ipynb) 제공


``` python

import argparse
import numpy as np
import fireducks.pandas as pd
import os

from hackathon import search_model
from hackathon.src.datasets.data_loader import load_data
import hackathon.src.dynamic_pipeline as dynamic_pipeline
from hackathon import surrogate_model


df = pd.read_csv('./hackathon/data/concrete.csv')
target = 'strength' # Y의 col 이름  
user_request_target = 40.0 # 유저가 원하는 Y값
model = 'tabpfn' # Surrogate model : catboost, tabpfn
control_name = ['cement', 'slag', 'ash', 'water'] # user가 수정을 원하는 X값
control_range = df[control_name] # user가 원하는 X값의 범위 
control_range = {col: (df[col].min(), df[col].max()) for col in control_range}

importance = {'cement': 1, 'water': 2,}# user가 원하는 X값의 최적화 순위
optimize = {'cement': 'minimize','water': 'minimize'} # user가 원하는 최적화 방향

# 데이터 전처리 후 저장 
ret_dict = dynamic_pipeline.preprocess_dynamic(df.copy())
ret_dict[0].to_csv('./temp.csv', index=False)


# Surrogate Model 학습 
args = argparse.Namespace(
    target = [target],
    data_path='./temp.csv',
    model=model, # catboost, tabpfn
    prj_id = 42,
    seed=42
)

surrogate_model.main(args, ret_dict[-1])

args = argparse.Namespace(
    model=model,
    search_model='k_means',
    data_path='./temp.csv',
    control_name=control_name,
    control_range= control_range,
    target=[target],
    importance=importance,
    optimize=optimize,
    prj_id=42,
    seed=42,
    user_request_target=[user_request_target],
    model_path='./temp/surrogate_model/model'
)

x_opt = search_model.main(args,ret_dict[-1])


```

