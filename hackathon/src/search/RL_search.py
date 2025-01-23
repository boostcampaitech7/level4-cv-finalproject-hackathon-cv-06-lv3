import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def RL_search(model):
    def set_seed(seed=42):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed(3232)

    class DQNetwork(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(DQNetwork, self).__init__()
            self.fc = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim)
            )

        def forward(self, x):
            return self.fc(x)


    # 데이터 준비
    df = pd.read_csv('./data/concrete.csv')
    init = df.iloc[:,:-1].mean().values
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 스케일링 및 스플릿
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df.iloc[:, :-1])
    data = pd.DataFrame(data_scaled, columns=df.columns[:-1])
    data['strength'] = df['strength']

    # Train/Test Split
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # DQN 에이전트 설정
    input_dim = train_data.shape[1]  # feature 개수
    action_dim = input_dim - 1  # 각 feature에 대한 액션
    q_network = DQNetwork(input_dim, action_dim).to(device)  # +1은 타겟 strength 포함
    optimizer = optim.Adam(q_network.parameters(), lr=1e-3, weight_decay=0.1)
    criterion = nn.MSELoss()

    # 학습 설정
    num_epochs = 2000
    batch_size = 32
    min_rew = 100
    traim_mean_loss = []
    train_mean_reward = []
    val_mean_loss = []
    val_mean_reward = []
    val_min_mean_reward = []
    for epoch in range(num_epochs):
        q_network.train()
        train_loss_sum = 0.0
        train_loss_count = 0
        train_total_reward = 0.0
        train_reward_count = 0

        for i in range(0, len(train_data), batch_size):
            batch = train_data.iloc[i:i + batch_size]
            features = batch.iloc[:, :-1].values
            actual_strengths = batch['strength'].values

            initial_state = np.repeat(np.expand_dims(init, axis=0), features.shape[0], axis=0)

            # LightGBM 예측
            scaled_initial_state = scaler.transform(pd.DataFrame(initial_state, columns=scaler.feature_names_in_))
            predicted_strengths = model.predict(scaled_initial_state)

            # DQN 입력 구성
            dqn_input = np.hstack([initial_state, actual_strengths[:,np.newaxis] - predicted_strengths[:,np.newaxis]])
            dqn_input_tensor = torch.tensor(dqn_input, dtype=torch.float32).to(device)

            # 액션 예측
            predicted_actions_tensor = q_network(dqn_input_tensor)

            # 실제 액션 계산 (실제 feature - 초기 상태)
            actual_actions_tensor = torch.tensor(scaler.inverse_transform(features) - initial_state, dtype=torch.float32).to(device)

            # 업데이트된 상태 계산
            updated_state = initial_state + predicted_actions_tensor.cpu().detach().numpy()
            scaled_updated_state = scaler.transform(pd.DataFrame(updated_state, columns=scaler.feature_names_in_))
            updated_strengths = model.predict(scaled_updated_state)

            # 리워드 계산
            rewards = updated_strengths - actual_strengths  # 리워드 계산
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)

            # DQN 손실 계산 (리워드 활용)
            loss = criterion(predicted_actions_tensor, actual_actions_tensor)
            #loss = (loss * rewards_tensor.mean())  # 리워드를 손실에 반영

            # 역전파 및 옵티마이저 업데이트
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_loss_count += len(rewards)
            train_total_reward += abs(rewards).sum().item()
            train_reward_count += len(rewards)

                # initial_state = updated_state[abs(rewards)>1]
                # actual_strengths = actual_strengths[abs(rewards)>1]
                # features = features[abs(rewards)>1]

        train_loss_mean = train_loss_sum / train_loss_count
        train_reward_mean = train_total_reward / train_reward_count
        traim_mean_loss.append(train_loss_mean)
        train_mean_reward.append(train_reward_mean)
        # ===== Validation =====
        q_network.eval()
        val_loss_sum = 0.0
        val_loss_count = 0
        val_total_reward = 0.0
        val_reward_count = 0
        val_min_reward = 0.0
        val_min_reward_count = 0
        with torch.no_grad():
            for i in range(0, len(test_data), batch_size):
                batch = test_data.iloc[i:i + batch_size]
                features = batch.iloc[:, :-1].values
                actual_strengths = batch['strength'].values

                # 초기 상태를 중간값으로 설정
                initial_state = np.repeat(np.expand_dims(init, axis=0), features.shape[0], axis=0)
                re = np.full(features.shape[0], 100)
                for j in range(10):
                    scaled_initial_state = scaler.transform(pd.DataFrame(initial_state, columns=scaler.feature_names_in_))
                    predicted_strengths = model.predict(scaled_initial_state)

                    # DQN 입력 구성
                    dqn_input = np.hstack([initial_state, actual_strengths[:,np.newaxis] - predicted_strengths[:,np.newaxis]])
                    dqn_input_tensor = torch.tensor(dqn_input, dtype=torch.float32).to(device)

                    # DQN으로 액션 예측
                    predicted_actions_tensor = q_network(dqn_input_tensor)

                    # 실제 액션 계산 (실제 feature - 초기 상태)
                    actual_actions_tensor = torch.tensor(scaler.inverse_transform(features) - initial_state, dtype=torch.float32).to(device)

                    # 업데이트된 상태 계산
                    updated_state = initial_state + predicted_actions_tensor.cpu().detach().numpy()
                    scaled_updated_state = scaler.transform(pd.DataFrame(updated_state, columns=scaler.feature_names_in_))
                    updated_strengths = model.predict(scaled_updated_state)

                    # 리워드 계산
                    rewards = updated_strengths - actual_strengths
                    rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)

                    # DQN 손실 계산
                    loss = criterion(predicted_actions_tensor, actual_actions_tensor)
                    #loss = (loss * rewards_tensor.mean())
                    
                    val_loss_sum += loss.item()
                    val_loss_count += len(rewards)
                    val_total_reward += abs(rewards).sum().item()
                    val_reward_count += len(rewards)
                    re = np.minimum(re, abs(rewards))
                    initial_state = updated_state
                val_min_reward += re.sum().item()
                val_min_reward_count += len(re)
        val_loss_mean = val_loss_sum / val_loss_count
        val_reward_mean = val_total_reward / val_reward_count
        val_min_reward_mean = val_min_reward/val_min_reward_count
        val_mean_loss.append(val_loss_mean)
        val_mean_reward.append(val_reward_mean)
        val_min_mean_reward.append(val_min_reward_mean)
        if val_min_reward_mean < min_rew: 
            min_rew = val_min_reward_mean
            torch.save(q_network.state_dict(), './model.pth') 
        # Epoch 결과 출력
        print(f"Epoch {epoch + 1}/{num_epochs}\n"
            f"Train Loss Mean: {train_loss_mean:.4f}, Train Total Reward: {train_total_reward:.4f}, Train Mean Reward: {train_reward_mean:.4f}, "
            f"Val Loss Mean: {val_loss_mean:.4f}, Val Total Reward: {val_total_reward:.4f}, Val Mean Reward: {val_reward_mean:.4f}, Val min Mean Reward: {val_min_reward_mean:.4f}")
    test(df, 7, model, q_network, scaler, init, device)

def test(df, idx, model, q_network, scaler, init, device):
    idx = 230
    tar_d = df.iloc[idx, :-1]
    tar = df.iloc[idx]['strength']
    print(f'y: {tar}, x: {tar_d}')

    now = model.predict(scaler.transform(pd.DataFrame(init.reshape(1,8), columns=scaler.feature_names_in_)))
    input = np.hstack([init.reshape(1,8),np.array([tar])[:,np.newaxis] - now[:,np.newaxis]])
    input_tensor = torch.tensor(input, dtype=torch.float32).to(device)
    ac = q_network(input_tensor).cpu().detach().numpy()
    print(f'예측 된 액션: {ac}')
    fe = ac+init.reshape(1,8)
    print(f'최종 x: {fe}')
    init_fe = scaler.transform(pd.DataFrame(init.reshape(1,8), columns=scaler.feature_names_in_))
    init_st = model.predict(init_fe)
    up_fe = scaler.transform(pd.DataFrame(fe, columns=scaler.feature_names_in_))
    up_st = model.predict(up_fe)
    print(f'목표 x: {tar_d}')
    print(f'초기 상태 y: {init_st}, 액션 이후 y: {up_st}, 목표 y: {tar}')