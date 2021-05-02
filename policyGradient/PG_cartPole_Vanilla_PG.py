import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
#from tensorboardX import SummaryWriter

import ptan
import torch.nn.functional as F

GAMMA = 0.99
LEARNING_RATE = 0.001
EPISODES_TO_TRAIN = 8
REWARD_STEPS = 10
ENTROPY_BETA = 0.01


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


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    #writer = SummaryWriter(comment="-cartpole-reinforce")
    net = PGN(env.observation_space.shape[0], env.action_space.n)
    agent = ptan.agent.PolicyAgent(
        net, preprocessor=ptan.agent.float32_preprocessor, apply_softmax=True)

    # notice the steps_count is used, so, the r returned is the discounted reward of the s,a over steps_count
    # this is instead of waiting for a whole episode to end to calculate the Q
    # in the AC method, will use another network to estimate the Q
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    done_episodes = 0

    batch_episodes = 0

    cur_rewards = []
    reward_sum = 0.0
    bs_smoothed = entropy = l_entropy = l_policy = l_total = None

    batch_states, batch_actions, batch_scales = [], [], []
    for step_idx, exp in enumerate(exp_source):
        reward_sum += exp.reward
        # this running average is to fix the gradient high variance
        # the A3C will get a better baseline
        baseline = reward_sum/(step_idx+1)
        #writer.add_scalar("baseline", baseline, step_idx)
        batch_states.append(exp.state)
        batch_actions.append(int(exp.action))
        batch_scales.append(exp.reward-baseline)
        cur_rewards.append(exp.reward)

        if exp.last_state is None:  # end of episode
            batch_episodes += 1

        # the following is for reporting
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            done_episodes += 1
            reward = new_rewards[0]
            total_rewards.append(reward)

        if len(total_rewards) >= 100:
            mean_rewards = float(np.mean(total_rewards[-100:]))
            print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d" %
                  (step_idx, reward, mean_rewards, done_episodes))
            #writer.add_scalar("episodes", done_episodes, step_idx)
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
        batch_scales_v = torch.FloatTensor(batch_scales)
        logits_v = net(state_v)
        log_prob_v = F.log_softmax(logits_v, dim=1)

        # for each step, take the value for the action taken in that step
        log_prob_actions_v = batch_scales_v * \
            log_prob_v[range(len(batch_states)), batch_actions_t]

        # notice range(len(batch_states)) for the first index, because

        loss_v = -log_prob_actions_v.mean()
        prob_v = F.softmax(logits_v, dim=1)
        # for exploration in PG we can do with entropy instead of epsilon greedy:
        # entropy: if this is high = uniform and minimum when some actions are 1 and some are 0
        entropy_v = -(prob_v*log_prob_v).sum(dim=1).mean()
        entropy_loss_v = -ENTROPY_BETA * entropy_v
        loss_v = loss_v+entropy_loss_v

        loss_v.backward()
        optimizer.step()

        new_logits_v = net(state_v)
        new_prob_v = F.softmax(new_logits_v, dim=1)
        kl_div_v = - ((new_prob_v/prob_v).log() * prob_v).sum(dim=1).mean()

        #writer.add_scalar("KL", kl_div.item(), step_idx)

        grad_max = 0.0
        grad_means = 0.0
        grad_count = 0

        for p in net.parameters():
            grad_max = max(grad_max, p.grad.abs().max().item())
            grad_means += (p.grad**2).mean().sqrt().item()
            grad_count += 1

        # writer.add_scalar("baseline", baseline, step_idx)
        # writer.add_scalar("entropy", entropy_v.item(), step_idx)
        # writer.add_scalar("batch_scales",
        #                   np.mean(batch_scales),
        #                   step_idx)
        # writer.add_scalar("loss_entropy",
        #                   entropy_loss_v.item(),
        #                   step_idx)
        # writer.add_scalar("loss_policy",
        #                   loss_policy_v.item(),
        #                   step_idx)
        # writer.add_scalar("loss_total",
        #                   loss_v.item(),
        #                   step_idx)
        # writer.add_scalar("grad_l2",
        #                   grad_means / grad_count,
        #                   step_idx)
        # writer.add_scalar("grad_max", grad_max, step_idx)

        batch_states.clear()
        batch_actions.clear()
        batch_scales.clear()

        batch_episodes = 0
        batch_states.clear()
        batch_actions.clear()
        batch_scales.clear()

    # writer.close()
