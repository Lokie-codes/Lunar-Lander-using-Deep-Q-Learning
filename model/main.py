import gym
import tensorflow as tf
from tensorflow import keras

# Define the model
model = keras.Sequential()
model.add(keras.layers.Dense(64, activation='relu', input_shape=(8,)))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(4, activation='linear'))

# Compile the model
model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.001))

def select_action(state, epsilon):
  """Select an action using an epsilon-greedy policy."""
  if np.random.rand() < epsilon:
    # Select a random action
    return np.random.randint(4)
  else:
    # Select the action with the highest predicted Q value
    q_values = model.predict(state)
    return np.argmax(q_values)

def update_model(state, action, reward, next_state, done):
  """Update the model based on the observed reward and the next state."""
  # Predict the Q values for the next state
  q_values_next = model.predict(next_state)

  # Calculate the target Q value
  if done:
    # The episode has ended, so the target is the final reward
    target = reward
  else
    # The episode has not ended, so the target is the reward plus the maximum
    # predicted Q value for the next state
    target = reward + gamma * np.max(q_values_next)

    # Predict the Q values for the current state
    q_values = model.predict(state)

    # Update the Q value for the action taken
    q_values[0][action] = target

    # Fit the model to the updated Q values
    model.fit(state, q_values, verbose=0)

# Create the Lunar Lander environment
env = gym.make('LunarLander-v2')

# Set the discount factor
gamma = 0.99

# Set the exploration rate
epsilon = 1.0

# Set the number of episodes to run
num_episodes = 1000

# Keep track of the rewards
rewards = []

# Run the game for the specified number of episodes
for episode in range(num_episodes):
  # Reset the environment and get the initial state
  state = env.reset()
  state = np.expand_dims(state, axis=0)

  # Initialize variables for the episode
  total_reward = 0
  done = False

  # Run the game until the episode is finished
  while not done:
    # Select an action using the action selection function
    action = select_action(state, epsilon)

    # Take a step in the environment
    next_state, reward, done, _ = env.step(action)
    next_state = np.expand_dims(next_state, axis=0)

    # Update the model based on the observed reward and the next state
    update_model(state, action, reward, next_state, done)

    # Update the variables for the next iteration
    state = next_state
    total_reward += reward

  # Reduce the exploration rate over time
  epsilon = max(0.1, epsilon * 0.99)

  # Add the episode reward to the list of rewards
  rewards.append(total_reward)

  # Print the episode reward
  print(f'Episode {episode}: {total_reward}')

# Plot the rewards
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
