import gym
import numpy as np
import matplotlib.pyplot as plt

# The mountain car environment has got two observation values: x-axis pos. [-1.2, 0.6] and velocity [-0.07, 0.07]
# (via env.observation_space.high, env.observation_space.low)
# Possible actions are: push left (0), no push (1), push right(2) (via env.action_space.n & documentation)
# Goal: reach the flag
env = gym.make("MountainCar-v0")


LEARNING_RATE = 0.5
DISCOUNTING_FACTOR = 0.9

epsilon = 1
epsilon_decay = 0.005

EPISODES = 4000
SHOWTIME = 10

# for demonstration purposes
STATISTICS_PER_EPISODE = 100
reward_list = []
aggregated_rewards = {'ep': [], 'rew': []}

# discretization due to granular observation values (10 x 10 shape)
DISCRETE_OBSERVATION_SPACE = [10, 10]
discrete_step_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OBSERVATION_SPACE

print(discrete_step_size)


def convert_to_discrete_value(state):
    state = (state - env.observation_space.low) / discrete_step_size
    return tuple(state.astype(np.int))


# Forming a 10 x 10 x 3 matrix
q_table = np.random.uniform(low=-1, high=0, size=(DISCRETE_OBSERVATION_SPACE + [env.action_space.n]))


for episode in range(EPISODES):
    reward_per_episode = 0

    discrete_state = convert_to_discrete_value(env.reset())
    done = False

    if episode % SHOWTIME == 0:
        render_current_episode = True
        print("rendered in episode: ", episode)
    else:
        render_current_episode = False

    while not done:
        if np.random.random() > epsilon:
            # retreive q-table action
            action = np.argmax(q_table[discrete_state])
        else:
            # randomize action
            action = env.action_space.sample()

        new_state, reward, done, _ = env.step(action)
        # add current reward to cumulated reward
        reward_per_episode += reward
        new_discrete_state = convert_to_discrete_value(new_state)

        if render_current_episode:
            env.render()

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            # q-learing formula
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNTING_FACTOR * max_future_q)
            q_table[discrete_state + (action,)] = new_q

        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action,)] = 0
            print("nailed it: ", episode)

        discrete_state = new_discrete_state

        # perform decaying
        epsilon -= epsilon_decay

    reward_list.append(reward_per_episode)
    if not episode % STATISTICS_PER_EPISODE and episode is not 0:
        average_reward = sum(reward_list[-STATISTICS_PER_EPISODE:]) / STATISTICS_PER_EPISODE
        aggregated_rewards['ep'].append(episode)
        aggregated_rewards['rew'].append(average_reward)

env.close()

plt.plot(aggregated_rewards['ep'], aggregated_rewards['rew'], label="rewards")
plt.legend(loc=4)
plt.show()