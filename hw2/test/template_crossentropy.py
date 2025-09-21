import numpy as np

n_states = 500
n_actions = 6


def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    """
    Select states and actions from games that have rewards >= percentile
    :param states_batch: list of lists of states, states_batch[session_i][t]
    :param actions_batch: list of lists of actions, actions_batch[session_i][t]
    :param rewards_batch: list of rewards, rewards_batch[session_i]
    :param percentile: percentile threshold for elite selection

    :returns: elite_states,elite_actions, both 1D lists of states and respective actions from elite sessions
    """
    reward_threshold = np.percentile(rewards_batch, percentile)
    elite_states = []
    elite_actions = []

    for i in range(len(rewards_batch)):
        if rewards_batch[i] >= reward_threshold:
            elite_states.extend(states_batch[i])
            elite_actions.extend(actions_batch[i])

    return elite_states, elite_actions


def update_policy(elite_states, elite_actions, n_states=n_states, n_actions=n_actions):
    """
    Given old policy and a list of elite states/actions from select_elites,
    return new updated policy where each action probability is proportional to

    policy[s_i,a_i] ~ #[occurences of si and ai in elite states/actions]

    Don't forget to normalize policy to get valid probabilities and handle 0/0 case.
    In case you never visited a state, set probabilities for all actions to 1./n_actions

    :param elite_states: 1D list of states from elite sessions
    :param elite_actions: 1D list of actions from elite sessions
    :param n_states: number of states in the environment
    :param n_actions: number of actions in the environment

    :returns: new_policy: np.array of shape (n_states, n_actions)
    """
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


def generate_session(env, policy, t_max=int(10 ** 4)):
    """
    Play game until end or for t_max ticks.
    :param env: gym environment
    :param policy: an array of shape [n_states,n_actions] with action probabilities
    :param t_max: maximum number of steps
    :returns: list of states, list of actions and sum of rewards
    """
    states, actions = [], []
    total_reward = 0.0

    s, info = env.reset()

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