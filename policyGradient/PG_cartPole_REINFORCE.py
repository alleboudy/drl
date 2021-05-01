import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import ptan
import torch.nn.functional as F

GAMMA = 0.99
LEARNING_RATE = 0.01
EPISODES_TO_TRAIN = 4


class PGN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(PGN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)


def calc_qvals(rewards):
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)
    return list(reversed(res))


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    writer = SummaryWriter(comment="-cartpole-reinforce")
    net = PGN(env.observation_space.shape[0], env.action_space.n)
    agent = ptan.agent.PolicyAgent(
        net, preprocessor=ptan.agent.float32_preprocessor, apply_softmax=True)

    expe_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=GAMMA)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    done_episodes = 0

    batch_episodes = 0

    cur_rewards = []

    batch_states, batch_actions, batch_qvals = [], [], []

    for step_idx, exp in enumerate(expe_source):
        batch_states.append(exp.state)
        batch_actions.append(exp.action)
        cur_rewards.append(exp.reward)

        if exp.last_state is None:  # end of episode
            batch_qvals.extend(calc_qvals(cur_rewards))
            cur_rewards.clear()
            batch_episodes += 1

        # the following is for reporting
        new_rewards = expe_source.pop_total_rewards()
        if new_rewards:
            done_episodes += 1
            reward = new_rewards[0]
            total_rewards.append(reward)

        if len(total_rewards) >= 100:
            mean_rewards = float(np.mean(total_rewards[-100:]))
            print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d" %
                (step_idx, reward, mean_rewards, done_episodes))
            writer.add_scalar("episodes", done_episodes, step_idx)
            if mean_rewards > 195:
                print("Solved in %d steps and %d episodes!" %
                    (step_idx, done_episodes))
                break

        # then training
        if batch_episodes < EPISODES_TO_TRAIN:
            continue
        optimizer.zero_grad()
        state_v = torch.FloatTensor(batch_states)
        batch_actions_t = torch.LongTensor(batch_actions)
        batch_qvals_v = torch.FloatTensor(batch_qvals)
        print("batch_qvals_v: ",batch_qvals_v.shape)
        print("batch_states: ",state_v.shape)
        print("batch_actions_t: ",batch_actions_t.shape)


        logits_v = net(state_v)
        log_prob_v = F.log_softmax(logits_v, dim=1)
        print('log_prob_v: ', log_prob_v.shape)
        print('log_prob_v[:, batch_actions_t]: ', log_prob_v[:, batch_actions_t].shape)
        print('log_prob_v[range(len(batch_states)), batch_actions_t]: ', log_prob_v[range(len(batch_states)), batch_actions_t].shape)

        log_prob_actions_v = batch_qvals_v * log_prob_v[range(len(batch_states)), batch_actions_t]  # for each step, take the value for the action taken in that step
        print("log_prob_actions_v: ", log_prob_actions_v.shape)
        # notice range(len(batch_states)) for the first index, because 

        loss_v=-log_prob_actions_v.mean()#why mean not sum?
        loss_v.backward()
        optimizer.step()
        batch_episodes=0
        batch_states.clear()
        batch_actions.clear()
        batch_qvals.clear()


    writer.close()
