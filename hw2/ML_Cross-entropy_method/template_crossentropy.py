# coding: utf-8

import numpy as np

n_states = 500  # for Taxi-v3
n_actions = 6  # for Taxi-v3


def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    reward_threshold = np.percentile(rewards_batch, percentile)
    elite_states = []
    elite_actions = []

    for i in range(len(rewards_batch)):
        if rewards_batch[i] >= reward_threshold:
            elite_states.extend(states_batch[i])
            elite_actions.extend(actions_batch[i])

    return elite_states, elite_actions


def update_policy(elite_states, elite_actions, n_states=n_states, n_actions=n_actions):
    new_policy = np.zeros((n_states, n_actions))

    for state, action in zip(elite_states, elite_actions):
        new_policy[state, action] += 1

    row_sums = new_policy.sum(axis=1)

    for state in range(n_states):
        if row_sums[state] == 0:
            new_policy[state] = np.ones(n_actions) / n_actions
        else:
            new_policy[state] /= row_sums[state]

    return new_policy


def generate_session(env, policy, t_max=int(10 ** 4)):
    states, actions = [], []
    total_reward = 0.

    s, info = env.reset()

    for t in range(t_max):
        a = np.random.choice(np.arange(policy.shape[1]), p=policy[s])
        new_s, r, done, truncated, info = env.step(a)

        states.append(s)
        actions.append(a)
        total_reward += r

        s = new_s
        if done:
            break

    return states, actions, total_reward