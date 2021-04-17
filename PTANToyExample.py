import gym
import ptan


class ToyEnv(gym.Env):

    def __init__(self, name):
        super(ToyEnv, self).__init__()
        self.name = name
        self.observation_space = gym.spaces.Discrete(n=5)
        self.action_space = gym.spaces.Discrete(n=3)
        self.step_index = 0

    def reset(self):
        self.step_index = 0
        return self.step_index

    def step(self, action):
        is_done = self.step_index == 10
        print("Env: ", self.name)

        if is_done:
            return self.step_index % self.observation_space.n, 0, is_done, {}

        self.step_index += 1

        return self.step_index % self.observation_space.n, float(action), self.step_index == 10, {}


class DullAgent(ptan.agent.BaseAgent):
    """
    Agent always returns the fixed action
    """

    def __init__(self, action: int):
        self.action = action

    def __call__(self, observations, state=None):
        return [self.action for _ in observations], state


env1 = ToyEnv("Env1")
env2 = ToyEnv("Env2")
agent = DullAgent(action=1)


exp_source = ptan.experience.ExperienceSource(
    env=[env1, env2], agent=agent, steps_count=4)
for idx, exp in enumerate(exp_source):
    if idx > 4:
        break
    print(exp)
