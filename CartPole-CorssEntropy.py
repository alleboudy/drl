import torch
from torch import nn, optim
from dataclasses import dataclass
import numpy as np
import gym

import tensorboardX


HIDDEN_SIZE = 128
BATCH_SIZE = 128
PERCENTILE = 70


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class EpisodeStep:
    def __init__(self, observation, action):
        self.observation = observation
        self.action = action


@dataclass
class Episode:
    def __init__(self, reward, steps):
        self.reward = reward
        self.steps = steps


def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    s_m = nn.Softmax(dim=1)

    while True:
        obs_v = torch.FloatTensor([obs])
        act_probs_v = s_m(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        o, r, d, _ = env.step(action)
        episode_reward += r
        episode_steps.append(EpisodeStep(observation=obs, action=action))

        if d:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            o = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []

        obs = o
def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for example in batch:
        if example.reward < reward_bound:
            continue
        train_act.extend(map(lambda s: s.action, example.steps))
        train_obs.extend(map(lambda s: s.observation, example.steps))

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)

    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    obs_size = env.observation_space.shape[0]
    print(f"env.observation_space: {env.observation_space.shape}")
    n_actions = env.action_space.n
    net = Net(obs_size, HIDDEN_SIZE, n_actions)

    loss = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=0.01)

    writer = tensorboardX.SummaryWriter()

    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)

        optimizer.zero_grad()
        action_scores_v = net(obs_v)

        loss_v = loss(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" %
              (iter_no, loss_v.item(), reward_m, reward_b))

        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        if reward_m > 199:
            print("Solved!")
            break

    writer.close()