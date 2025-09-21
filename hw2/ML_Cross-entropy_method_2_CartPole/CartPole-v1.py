import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import json
import warnings

warnings.filterwarnings('ignore')


# Функция для выбора элитных сессий
def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    if len(rewards_batch) == 0:
        return [], []

    reward_threshold = np.percentile(rewards_batch, percentile)
    elite_states = []
    elite_actions = []

    for i in range(len(rewards_batch)):
        if rewards_batch[i] >= reward_threshold:
            elite_states.extend(states_batch[i])
            elite_actions.extend(actions_batch[i])

    return elite_states, elite_actions


# Создание среды CartPole-v1 (актуальная версия)
env = gym.make("CartPole-v1", render_mode="rgb_array")
n_actions = env.action_space.n

# Создание и инициализация агента
agent = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation='tanh',
    solver='adam',
    max_iter=1000,
    random_state=42,
    warm_start=True  # Позволяет дообучать модель без сброса весов
)

# Инициализация агента
initial_state, _ = env.reset()
initial_states = [initial_state for _ in range(n_actions)]
initial_actions = list(range(n_actions))
agent.fit(initial_states, initial_actions)


# Функция для генерации сессий
def generate_session_cartpole(t_max=500):  # Увеличим лимит для v1
    states, actions = [], []
    total_reward = 0
    s, info = env.reset()

    for t in range(t_max):
        # Предсказание вероятностей действий
        probs = agent.predict_proba([s])[0]
        a = np.random.choice(n_actions, p=probs)

        new_s, r, done, truncated, info = env.step(a)

        states.append(s)
        actions.append(a)
        total_reward += r

        s = new_s
        if done or truncated:
            break

    return states, actions, total_reward


# Параметры обучения
n_sessions = 100
percentile = 70
log = []
best_mean_reward = -float('inf')

# Обучение агента
try:
    for i in range(100):
        sessions = [generate_session_cartpole() for _ in range(n_sessions)]
        states_batch, actions_batch, rewards_batch = zip(*sessions)

        elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile)

        # Проверка, что есть элитные данные для обучения
        if len(elite_states) > 0 and len(elite_actions) > 0:
            agent.fit(elite_states, elite_actions)

        mean_reward = np.mean(rewards_batch)
        threshold = np.percentile(rewards_batch, percentile) if len(rewards_batch) > 0 else 0
        log.append([mean_reward, threshold])

        # Сохранение лучшей модели
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward

        print(f"Iteration: {i}, Mean reward: {mean_reward:.3f}, Threshold: {threshold:.3f}")

        if mean_reward >= 475:  # CartPole-v1 имеет максимальную награду 500
            print("You Win!")
            break
except KeyboardInterrupt:
    print("Training interrupted by user")

# Генерация данных для отправки
sessions = [generate_session_cartpole() for _ in range(n_sessions)]
states_batch, actions_batch, rewards_batch = zip(*sessions)

sessions_to_send = []
for session in sessions:
    observations = [x.tolist() if hasattr(x, 'tolist') else x for x in session[0]]
    actions = session[1]  # actions уже являются целыми числами
    sessions_to_send.append((observations, actions))

# Сохранение в файл
with open('sessions_to_send.json', 'w') as iofile:
    json.dump(sessions_to_send, iofile, ensure_ascii=True, indent=4)

env.close()

# Визуализация результатов обучения
if log:
    plt.figure(figsize=(10, 6))
    rewards, thresholds = zip(*log)
    plt.plot(rewards, label='Mean Reward')
    plt.plot(thresholds, label='Threshold')
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_progress.png')
    plt.show()