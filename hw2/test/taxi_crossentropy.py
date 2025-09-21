import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Константы для Taxi-v3
n_states = 500
n_actions = 6


def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    """Выбор элитных состояний и действий"""
    reward_threshold = np.percentile(rewards_batch, percentile)
    elite_states, elite_actions = [], []

    for i in range(len(rewards_batch)):
        if rewards_batch[i] >= reward_threshold:
            elite_states.extend(states_batch[i])
            elite_actions.extend(actions_batch[i])

    return elite_states, elite_actions


def update_policy(elite_states, elite_actions):
    """Обновление политики на основе элитных данных"""
    new_policy = np.zeros((n_states, n_actions))

    for state, action in zip(elite_states, elite_actions):
        new_policy[state][action] += 1

    row_sums = new_policy.sum(axis=1)
    for i in range(n_states):
        if row_sums[i] == 0:
            new_policy[i] = np.ones(n_actions) / n_actions
        else:
            new_policy[i] /= row_sums[i]

    return new_policy


def generate_session(env, policy, t_max=10000):
    """Генерация игровой сессии"""
    states, actions = [], []
    total_reward = 0.0
    s, _ = env.reset()

    for t in range(t_max):
        a = np.random.choice(n_actions, p=policy[s])
        new_s, r, done, _, _ = env.step(a)

        states.append(s)
        actions.append(a)
        total_reward += r
        s = new_s

        if done:
            break

    return states, actions, total_reward


def train_taxi():
    """Основная функция обучения для Taxi-v3"""
    env = gym.make("Taxi-v3")
    policy = np.ones([n_states, n_actions]) / n_actions

    n_sessions = 250
    percentile = 50
    learning_rate = 0.5
    log = []

    for i in range(100):
        sessions = [generate_session(env, policy) for _ in range(n_sessions)]
        states_batch, actions_batch, rewards_batch = zip(*sessions)

        elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile)
        new_policy = update_policy(elite_states, elite_actions)

        policy = learning_rate * new_policy + (1 - learning_rate) * policy

        mean_reward = np.mean(rewards_batch)
        threshold = np.percentile(rewards_batch, percentile)
        log.append([mean_reward, threshold])

        print(f"Iteration {i}: mean reward = {mean_reward:.3f}, threshold = {threshold:.3f}")

        if mean_reward > -50:
            print("You Win!")
            break

    env.close()
    return policy


if __name__ == "__main__":
    trained_policy = train_taxi()