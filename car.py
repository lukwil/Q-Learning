import gym
import numpy as np


# The mountain car environment has got two observation values: x-axis pos. [-1.2, 0.6] and velocity [-0.07, 0.07]
# Possible actions are: push left (0), no push (1), push right(2)
env = gym.make("MountainCar-v0")


LEARNING_RATE = 0.1
DISCOUNTING_FACTOR = 0.95
EPISODES = 25000

SHOWTIME = 5000

# discretization due to granular observation values (20 x 20 shape)
DISCRETE_OBSERVATION_SPACE = [20] * 2
discrete_step_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OBSERVATION_SPACE

print(discrete_step_size)

#test
def convert_to_discrete_value(state):
    discrete_state = (state - env.observation_space.low) / discrete_step_size
    return tuple(discrete_state.astype(np.int))


# Forming a 20 x 20 x 3 matrix
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OBSERVATION_SPACE + [env.action_space.n]))


for episode in range(EPISODES):
    if episode % SHOWTIME == 0:
        render = True
        print(episode)
    else:
        render = False

    discrete_state = convert_to_discrete_value(env.reset())
    done = False

    while not done:
        action = np.argmax(q_table[discrete_state])
        new_state, reward, done, _ = env.step(action)
        new_discrete_state = convert_to_discrete_value(new_state)

        if render:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNTING_FACTOR * max_future_q)
            q_table[discrete_state + (action,)] = new_q
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action,)] = 0
            print("geschafft: ", episode)
env.close()



