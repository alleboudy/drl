import torch
import torch.nn as nn

import numpy as np

import ptan
import gym
import torch.optim as optim


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x.float())


TGT_NET_SYNC = 10
REPLAY_SIZE = 10000
GAMMA = 0.99
EPS_DECAY = 0.99
env = gym.make("CartPole-v0")
BATCH_SIZE = 32
loss = nn.MSELoss()


net = Net(env.observation_space.shape[0], 128, env.action_space.n)

tgt_net = ptan.agent.TargetNet(net)

selector = ptan.actions.ArgmaxActionSelector()
selector = ptan.actions.EpsilonGreedyActionSelector(
    epsilon=1, selector=selector)

agent = ptan.agent.DQNAgent(net, selector)

exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)

buffer = ptan.experience.ExperienceReplayBuffer(
    exp_source, buffer_size=REPLAY_SIZE)

optimizer = optim.Adam(net.parameters(), lr=0.01)

step = 0
episode = 0
batch = None
solved = False

device = "cpu"
while True:
    step += 1
    buffer.populate(1)
    for reward, steps in exp_source.pop_rewards_steps():
        episode += 1
        print("%d: episode %d done, reward=%.3f, epsilon=%.2f" %
              (step, episode, reward, selector.epsilon))
        solved = reward > 150

    if solved:
        print("solved!")
        break
    if len(buffer) < 2 * BATCH_SIZE:
        continue
    batch = buffer.sample(BATCH_SIZE)

    states, actions, rewards, next_states_tuple = list(zip(*batch))

    #dones = [True if s is None else False for s in next_states]
    dones = []
    next_states = []
    for i in range(len(next_states_tuple)):
        if next_states_tuple[i] is None:
            dones.append(True)
            next_states.append(states[0])
        else:
            dones.append(False)
            next_states.append(next_states_tuple[i])

    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)
    next_states_v = torch.tensor(next_states).to(device)

    next_states_values = tgt_net.target_model(next_states_v.float()).max(1)[0]
    next_states_values[done_mask] = 0.0

    next_states_values = next_states_values.detach()

    tgt_q_v = rewards_v + next_states_values * GAMMA

    optimizer.zero_grad()
    q_v = net(states_v.float())
    q_v = q_v.gather(1, actions_v.unsqueeze(-1)).squeeze(1)

    loss_v = loss(q_v, tgt_q_v)
    loss_v.backward()
    optimizer.step()
    selector.epsilon *= EPS_DECAY
    if step % TGT_NET_SYNC == 0:
        tgt_net.sync()
