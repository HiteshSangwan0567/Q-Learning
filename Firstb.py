import gym
import numpy as np

env = gym.make("MountainCar-v0")

Edit > Untabify Region
LEARNING_RATE = 0.1
DISCOUNT = 0.95 #Measure of how Important the future action is
               #Measure of Weights
EPISODES = 25000
#print(env.observation_space.high)
#print(env.observation_space.low)
#print(env.action_space.n)

DISCRETE_OS_SIZE = [20,20]
# divide into 20 buckets/ chunks
#How big are the chunks
discrete_os_win_size = (env.observation_space.high - env.observation_space.low/DISCRETE_OS_SIZE)

#print(discrete_os_win_size)

q_table = np.random.uniform(low=-2, high=0, size = (DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/ discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

discrete_state = get_discrete_state(env.reset())

done = False

while not done:
	action = np.argmax(q_table[discrete_state])
	new_state, reward, done, _ = env.step(action)

	new_discrete_state = get_discrete_state(new_state)
	#print(reward, new_state)#\\ blow out our memory\\ more episodes...\\reward will be -1 for all till it reaches the flag
	env.render()
	if not done:
            
             max_future_q=np.max(q_table[new_discrete_state])
             current_q = q_table[discrete_state + (action,)]
        
             new_q = (1 - LEARNING_RATE)* current_q + LEARNING_RATE * (reward + DISCOUNT *max_future_q)
             q_table[discrete_state + (action, )] = new_q
             
        elif new_state[0] >= env.goal_position:

             q_table[discrete_state + (action,)] = 0
             
        discrete_state = new_discrete_state
     # 1. build the Q table...\\
     # Convert these contnous values to discrete values
     # Buckets of different sizes.

env.close()
