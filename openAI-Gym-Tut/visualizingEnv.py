import gym
import time

env = gym.make('CartPole-v0')

env.reset()

for step in range(1000):
    env.render()
    env.step(env.action_space.sample())
    time.sleep(0.1)