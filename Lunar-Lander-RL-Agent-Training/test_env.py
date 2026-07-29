import gymnasium as gym

env = gym.make("LunarLander-v3")

print("Observation Space:", env.observation_space)
print("Action Space:", env.action_space)

env.close()