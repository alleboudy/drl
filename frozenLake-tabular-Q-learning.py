import gym
import collections
from tensorboardX import SummaryWriter


"""
Here, we only keep Q(s,a) for states and action we explore
so,
1- we take a step in the environment with the best possible action in the current state
2- update our Q value for that state
3- play some test episode accumulating their average reward while not touching the Q table
depending on the average rewards we might do another sample step or terminate.

"""



ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 20

class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.q_values  =collections.defaultdict(float)


    def sample_env(self):
        """
        Takes a step in the environment with a sampled action
        """
        action = self.env.action_space.sample()
        old_state = self.state
        new_state, reward, is_done, _ = self.env.step(action)
        self.state = self.env.reset() if is_done else new_state
        return (old_state, action, reward, new_state)

    def best_value_and_action(self,state):
        """
        For a given statem returns the best action with the heights Q value
        Will be used for:
            1- calculating the Q value of the landing state to update the current
            2- when testing, to pick the best action
        """
        best_value, best_action = None, None
        for action in range(self.env.action_space.n):
            action_value = self.q_values[(state,action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value,best_action

    def q_value_update(self, s, a, r, next_s):
        best_v, _ =self.best_value_and_action(next_s)
        new_val = r + GAMMA*best_v
        old_val = self.q_values[(s,a)]
        self.q_values[(s,a)] = old_val*(1 - ALPHA ) + new_val *ALPHA


    def play_episode(self, env):
        """
        Plays a full episode on a given test environment and returns its total reward
        """
        total_reward = 0.0
        state = env.reset()
        while True:
            _,action = self.best_value_and_action(state)
            new_state, reward, is_done,_ =env.step(action)
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

if __name__ == '__main__':
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-q-learning")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no +=1
        if iter_no%100==0:
            print('Iteration #{}'.format(iter_no))
        s,a,r, next_s = agent.sample_env()
        agent.q_value_update(s,a,r,next_s)

        reward = 0
        for _ in range(TEST_EPISODES):
            reward+= agent.play_episode(test_env)
        reward/=TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward>=0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()
    agent.env.close()
    test_env.close()








