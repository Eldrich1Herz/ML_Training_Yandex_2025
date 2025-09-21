import numpy as np
import gymnasium as gym
from template_crossentropy import select_elites, update_policy, generate_session


# Тестирование функций
def test_functions():
    # Тест select_elites
    states_batch = [[1, 2, 3], [4, 2, 0, 2], [3, 1]]
    actions_batch = [[0, 2, 4], [3, 2, 0, 1], [3, 3]]
    rewards_batch = [3, 4, 5]

    elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile=30)
    print("Test select_elites passed!")

    # Тест update_policy
    new_policy = update_policy(elite_states, elite_actions)
    print("Test update_policy passed!")

    # Тест generate_session
    env = gym.make("Taxi-v3")
    policy = np.ones([500, 6]) / 6
    states, actions, reward = generate_session(env, policy)
    print("Test generate_session passed!")
    env.close()


if __name__ == "__main__":
    test_functions()