import gym
import collections
from tensorboardX import SummaryWriter

import random

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 20
epsilon = 0.2

class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.values = collections.defaultdict(float)

    def sample_env(self):
        action = self.env.action_space.sample()
        old_state = self.state
        new_state, reward, is_done, _ = self.env.step(action)

        self.state = self.env.reset() if is_done else new_state

        return (old_state, action, reward, new_state)

    def best_value_and_action(self, state, rand=False):
        best_value, best_action = None, None

        if rand:
            if random.random()<epsilon:
                action = self.env.action_space.sample()
                act_value = self.values[(state, action)] 
                return act_value, action

        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action

        return best_value, best_action

    def value_update(self, s, a, r, next_s):
        best_v, _ = self.best_value_and_action(next_s, True)
        new_val = r + GAMMA * best_v
        old_val = self.values[(s, a)]
        self.values[(s, a)] = old_val * (1-ALPHA) + new_val*ALPHA

    def play_episode(self, env):
        total_reward = 0
        state = env.reset()
        while True:
            _, action = self.best_value_and_action(state)
            o, r, d, _ = env.step(action)
            total_reward += r
            if d:
                break
            state = o

        return total_reward


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-q-learning")

    iter_num = 0

    best_reward = 0.0
    while True:
        iter_num += 1
        s, a, r, next_s = agent.sample_env()
        agent.value_update(s,a,r,next_s)
        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_num)
        if reward > best_reward:
            print(f"Best reward updated: {best_reward}->{reward}")
            best_reward = reward
        if reward > 0.8:
            print(f"Solved in {iter_num} iterations.")
            break
writer.close()
