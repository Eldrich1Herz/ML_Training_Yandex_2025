import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from template_crossentropy import generate_session, select_elites, update_policy

# Инициализация среды
env = gym.make("Taxi-v3")
n_states = env.observation_space.n
n_actions = env.action_space.n

# Инициализация политики
policy = np.ones((n_states, n_actions)) / n_actions

# Параметры обучения
n_sessions = 250
percentile = 50
learning_rate = 0.5
log = []

# Обучение
for i in range(100):
    sessions = [generate_session(env, policy) for _ in range(n_sessions)]
    states_batch, actions_batch, rewards_batch = zip(*sessions)

    elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile)
    new_policy = update_policy(elite_states, elite_actions, n_states, n_actions)

    policy = learning_rate * new_policy + (1 - learning_rate) * policy

    # Визуализация прогресса
    mean_reward = np.mean(rewards_batch)
    threshold = np.percentile(rewards_batch, percentile)
    log.append([mean_reward, threshold])

    print(f"Iteration: {i}, Mean reward: {mean_reward:.3f}, Threshold: {threshold:.3f}")

    if mean_reward > -50:
        print("You Win!")
        break

env.close()

# Визуализация результатов
plt.figure(figsize=[10, 4])
plt.subplot(1, 2, 1)
plt.plot(list(zip(*log))[0], label='Mean rewards')
plt.plot(list(zip(*log))[1], label='Reward thresholds')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.hist(rewards_batch, bins=20)
plt.vlines([np.percentile(rewards_batch, percentile)], [0], [100], label="percentile", color='red')
plt.legend()
plt.grid()

plt.show()