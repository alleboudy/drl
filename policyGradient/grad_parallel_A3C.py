import os
import gym
import ptan
import numpy as np
import argparse
import collections
from tensorboardX import SummaryWriter
import torch.nn.utils as nn_utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import common


GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
REWARD_STEPS = 4
CLIP_GRAD = 0.1
PROCESSES_COUNT = 4
NUM_ENVS = 8
GRAD_BATCH = 64
TRAIN_BATCH = 2
ENV_NAME = "PongNoFrameskip-v4"
REWARD_BOUND = 18

class AtariA2C(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(AtariA2C, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=(8, 8), stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.policy(conv_out), self.value(conv_out)


def unpack_batch(batch, net, device='cpu', last_val_gamma=GAMMA ** REWARD_STEPS):
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []

    for idx, exp in enumerate(batch):
        states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(np.array(exp.last_state, copy=False))
    # the redundant np.array() is due to the issue 13918
    states_v = torch.FloatTensor(np.array(states, copy=False)).to(device)
    actions_t = torch.LongTensor(actions).to(device)
    rewards_np = np.array(rewards, dtype=np.float32)
    # note that, the V(s) term is (Sum_{i-N} GAMMA^i x r_i) + GAMMA^N V(S_N) -> (bellman q for N steps)
    # and the rewards we got rewards_np are just the first part before the +
    # because the ExperineceSourceFirstLast returns rewards as it already discounted for the subtrajectory
    # so, we are missing the second part, only for the not done steps
    # this is the opposite of before with DQN where we just masked Q'(s_{t+1},a) for done episodes, because we had Q values as nw output
    # here we have advantages instead

    if not_done_idx:
        last_states_v = torch.FloatTensor(
            np.array(last_states, copy=False)).to(device)
        last_vals_v = net(last_states_v)[1]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        last_vals_np *= GAMMA ** REWARD_STEPS
        rewards_np[not_done_idx] += last_vals_np * last_val_gamma

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)

    # NOTE: ref_vals_v is not actually the q or the v, but the TD(N) return
    # so, it is not also the return G(t), but just till N steps
    return states_v, actions_t, ref_vals_v


def make_env():
    return ptan.common.wrappers.wrap_dqn(gym.make(ENV_NAME))


TotalReward = collections.namedtuple('TotalReward',
                                     field_names='reward')

def grads_func(proc_name, net, device, train_queue):
    envs = [make_env() for _ in range(NUM_ENVS)]
    agent = ptan.agent.PolicyAgent(
        lambda x: net(x)[0], device=device, apply_softmax=True)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)
    batch = []
    frame_idx = 0
    writer = SummaryWriter(comment=proc_name)
    with common.RewardTracker(writer, REWARD_BOUND) as tracker:
        with ptan.common.utils.TBMeanTracker(
                writer, 100) as tb_tracker:
            for exp in exp_source:
                frame_idx += 1
                new_rewards = exp_source.pop_total_rewards()
                if new_rewards and tracker.reward(
                        new_rewards[0], frame_idx):
                    break
                batch.append(exp)
                if len(batch) < GRAD_BATCH:
                    continue
                data = unpack_batch(
                    batch, net, device=device,
                    last_val_gamma=GAMMA**REWARD_STEPS)
                states_v, actions_t, vals_ref_v = data
                batch.clear()
                net.zero_grad()
                logits_v, value_v = net(states_v)
                loss_value_v = F.mse_loss(
                    value_v.squeeze(-1), vals_ref_v)
                log_prob_v = F.log_softmax(logits_v, dim=1)
                adv_v = vals_ref_v - value_v.detach()
                log_p_a = log_prob_v[range(GRAD_BATCH), actions_t]
                log_prob_actions_v = adv_v * log_p_a
                loss_policy_v = -log_prob_actions_v.mean()
                prob_v = F.softmax(logits_v, dim=1)
                ent = (prob_v * log_prob_v).sum(dim=1).mean()
                entropy_loss_v = ENTROPY_BETA * ent
                loss_v = entropy_loss_v + loss_value_v + \
                         loss_policy_v
                loss_v.backward()
                tb_tracker.track("advantage", adv_v, frame_idx)
                tb_tracker.track("values", value_v, frame_idx)
                tb_tracker.track("batch_rewards", vals_ref_v,
                                 frame_idx)
                tb_tracker.track("loss_entropy", entropy_loss_v,
                                 frame_idx)
                tb_tracker.track("loss_policy", loss_policy_v,
                                 frame_idx)
                tb_tracker.track("loss_value", loss_value_v,
                                 frame_idx)
                tb_tracker.track("loss_total", loss_v, frame_idx)
                nn_utils.clip_grad_norm_(
                    net.parameters(), CLIP_GRAD)
                grads = [
                    param.grad.data.cpu().numpy()
                    if param.grad is not None else None
                    for param in net.parameters()
                ]
                train_queue.put(grads)
    train_queue.put(None)
    

if __name__ == "__main__":
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=True,
                        help="Name of the run")
    args = parser.parse_args()
    device = "cuda" if args.cuda else "cpu"
    writer = SummaryWriter(comment=f"-a3c-data_pong_{args.name}")

    env = make_env()
    net = AtariA2C(env.observation_space.shape,
                   env.action_space.n).to(device)
    net.share_memory()
    optimizer = optim.Adam(net.parameters(),
                           lr=LEARNING_RATE, eps=1e-3)
    train_queue = mp.Queue(maxsize=PROCESSES_COUNT)
    data_proc_list = []
    for proc_idx in range(PROCESSES_COUNT):
        proc_name = f"-a3c-grad_pong_{args.name}#{proc_idx}"
        p_args = (proc_name, net, device, train_queue)
        data_proc = mp.Process(target=grads_func, args=p_args)
        data_proc.start()
        data_proc_list.append(data_proc)

    batch = []
    step_idx = 0
    grad_buffer = None
    try:
        while True:
            train_entry = train_queue.get()
            if train_entry is None:
                break
            step_idx += 1
            if grad_buffer is None:
                grad_buffer = train_entry
            else:
                for tgt_grad, grad in zip(grad_buffer,
                                          train_entry):
                    tgt_grad += grad
            if step_idx % TRAIN_BATCH == 0:
                for param, grad in zip(net.parameters(),
                                       grad_buffer):
                    param.grad = torch.FloatTensor(grad).to(device)
                nn_utils.clip_grad_norm_(
                    net.parameters(), CLIP_GRAD)
                optimizer.step()
                grad_buffer = None
    finally:
        for p in data_proc_list:
            p.terminate()
            p.join()
