import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
from gymnasium.utils.save_video import save_video

# Инициализация среды
env = gym.make("CartPole-v1", render_mode="rgb_array")
env.reset()
n_actions = env.action_space.n
state_dim = env.observation_space.shape


# Построение нейронной сети
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = PolicyNetwork(state_dim[0], n_actions)
print("Модель создана успешно")


# Функция 1: predict_probs
def predict_probs(states):
    """
    Predict action probabilities given states.
    :param states: numpy array of shape [batch, state_shape]
    :returns: numpy array of shape [batch, n_actions]
    """
    with torch.no_grad():
        # Убедимся, что states - это numpy array перед конвертацией
        if isinstance(states, list):
            states = np.array(states)
        states_tensor = torch.tensor(states, dtype=torch.float32)
        logits = model(states_tensor)
        probs = torch.softmax(logits, dim=-1).numpy()

    assert probs is not None, "probs is not defined"
    return probs


# Тестирование predict_probs
test_states = np.array([env.reset()[0] for _ in range(5)])
test_probas = predict_probs(test_states)
assert isinstance(test_probas, np.ndarray), "you must return np array and not %s" % type(test_probas)
assert tuple(test_probas.shape) == (test_states.shape[0], env.action_space.n), "wrong output shape: %s" % np.shape(
    test_probas)
assert np.allclose(np.sum(test_probas, axis=1), 1), "probabilities do not sum to 1"
print("predict_probs тест пройден!")


# Функция генерации сессии
def generate_session(env, t_max=1000):
    states, actions, rewards = [], [], []
    s, info = env.reset()

    for t in range(t_max):
        action_probs = predict_probs(np.array([s]))[0]
        a = np.random.choice(n_actions, p=action_probs)
        new_s, r, done, truncated, info = env.step(a)

        states.append(s)
        actions.append(a)
        rewards.append(r)

        s = new_s
        if done:
            break

    return states, actions, rewards


# Функция 2: get_cumulative_rewards
def get_cumulative_rewards(rewards, gamma=0.99):
    """
    Take a list of immediate rewards r(s,a) for the whole session
    and compute cumulative returns (a.k.a. G(s,a) in Sutton '16).
    """
    cumulative_rewards = []
    G = 0.0

    for r in reversed(rewards):
        G = r + gamma * G
        cumulative_rewards.insert(0, G)

    assert cumulative_rewards is not None, "cumulative_rewards is not defined"
    return cumulative_rewards


# Тестирование get_cumulative_rewards
assert len(get_cumulative_rewards(list(range(100)))) == 100
assert np.allclose(
    get_cumulative_rewards([0, 0, 1, 0, 0, 1, 0], gamma=0.9),
    [1.40049, 1.5561, 1.729, 0.81, 0.9, 1.0, 0.0])
assert np.allclose(
    get_cumulative_rewards([0, 0, 1, -2, 3, -4, 0], gamma=0.5),
    [0.0625, 0.125, 0.25, -1.5, 1.0, -4.0, 0.0])
assert np.allclose(
    get_cumulative_rewards([0, 0, 1, 2, 3, 4, 0], gamma=0),
    [0, 0, 1, 2, 3, 4, 0])
print("get_cumulative_rewards тест пройден!")


# Вспомогательная функция
def to_one_hot(y_tensor, ndims):
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    y_one_hot = torch.zeros(y_tensor.size()[0], ndims).scatter_(1, y_tensor, 1)
    return y_one_hot


# Функция 3: get_loss (ИСПРАВЛЕННАЯ ВЕРСИЯ)
def get_loss(states, actions, rewards, gamma=0.99, entropy_coef=1e-2):
    """
    Compute the loss for the REINFORCE algorithm.
    """
    # Оптимизация: преобразуем список в numpy array перед созданием тензора
    states = torch.tensor(np.array(states), dtype=torch.float32)  # ИСПРАВЛЕНИЕ ЗДЕСЬ
    actions = torch.tensor(np.array(actions), dtype=torch.int32)  # Также оптимизируем actions
    cumulative_returns = np.array(get_cumulative_rewards(rewards, gamma))
    cumulative_returns = torch.tensor(cumulative_returns, dtype=torch.float32)

    # predict logits, probas and log-probas using an agent.
    logits = model(states)
    assert logits is not None, "logits is not defined"

    probs = torch.softmax(logits, dim=-1)
    assert probs is not None, "probs is not defined"

    log_probs = torch.log_softmax(logits, dim=-1)
    assert log_probs is not None, "log_probs is not defined"

    assert all(isinstance(v, torch.Tensor) for v in [logits, probs, log_probs]), \
        "please use compute using torch tensors and don't use predict_probs function"

    # select log-probabilities for chosen actions, log pi(a_i|s_i)
    log_probs_for_actions = log_probs[range(len(actions)), actions]
    assert log_probs_for_actions is not None, "log_probs_for_actions is not defined"

    J_hat = torch.mean(log_probs_for_actions * cumulative_returns)
    assert J_hat is not None, "J_hat is not defined"

    # Compute loss here. Don't forget entropy regularization with `entropy_coef`
    entropy = -torch.sum(probs * log_probs, dim=-1).mean()
    assert entropy is not None, "entropy is not defined"

    loss = -J_hat - entropy_coef * entropy
    assert loss is not None, "loss is not defined"

    return loss


# Оптимизатор и функция обучения
optimizer = torch.optim.Adam(model.parameters(), 1e-3)


def train_on_session(states, actions, rewards, gamma=0.99, entropy_coef=1e-2):
    optimizer.zero_grad()
    loss = get_loss(states, actions, rewards, gamma, entropy_coef)
    loss.backward()
    optimizer.step()
    return np.sum(rewards)


# Обучение модели
print("Начинаем обучение...")
for i in range(100):
    sessions = [generate_session(env) for _ in range(10)]
    rewards = [train_on_session(*session, entropy_coef=1e-3) for session in sessions]

    if i % 10 == 0:
        mean_reward = np.mean(rewards)
        print(f"Итерация {i}: средняя награда: {mean_reward:.3f}")

    if np.mean(rewards) > 400:
        print("Достигнута хорошая производительность!")
        break

# Создание видео
env_for_video = gym.make("CartPole-v1", render_mode="rgb_array_list")
episode_index = 0
step_starting_index = 0

obs, info = env_for_video.reset()

for step_index in range(200):
    probs = predict_probs(np.array([obs]))[0]
    action = np.random.choice(n_actions, p=probs)
    obs, reward, terminated, truncated, info = env_for_video.step(action)
    done = terminated or truncated

    if done or step_index == 199:
        frames = env_for_video.render()
        os.makedirs("videos", exist_ok=True)
        save_video(
            frames, "videos",
            fps=env_for_video.metadata.get("render_fps", 30),
            step_starting_index=step_starting_index,
            episode_index=episode_index,
        )
        episode_index += 1
        step_starting_index = step_index + 1
        obs, info = env_for_video.reset()

env_for_video.close()
print("Видео сохранено в папке 'videos'!")

env.close()
print("Все задания выполнены успешно!")