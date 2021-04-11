from tensorboardX import SummaryWriter
import torch.optim as optim
import torch.nn as nn
import torch
import collections
import numpy as np
import time
import argparse
from lib import dqN_model2
from lib import wrappers

import pickle

print("our new code!")


DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.0

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
REPLAY_START_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000

EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 1.0
EPSILON_FINAL = 0.01

Experience = collections.namedtuple('Experience', field_names=[
                                    "state", "action", "reward", "done", "next_states"])


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(
            *[self.buffer[idx] for idx in indices])

        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device="cuda"):
        done_reward = None

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)  # shape-> [1,n_actions]
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward
        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def calc_loss(batch, net, tgt_net, device="cuda"):
    states, actions, rewards, dones, next_states = batch
    with open('batch.pkl', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(batch, f)
        print("saved batch!")
        raise 

    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    state_action_values = net(states_v).gather(
        1, actions_v.unsqueeze(-1)).squeeze(-1)

    next_states_values = tgt_net(next_states_v).max(1)[0]
    next_states_values[done_mask] = 0.0

    next_states_values = next_states_values.detach()

    expected_state_action_values = rewards_v + next_states_values * GAMMA

    return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help=f"Name of the environment, default={DEFAULT_ENV_NAME}")

    parser.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND,
                        help=f"Mean reward boundary fpr stopping the training, default={MEAN_REWARD_BOUND}")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = wrappers.make_env(args.env)
    net = dqN_model2.DQN(env.observation_space.shape,
                         env.action_space.n).to(device)
    tgt_net = dqN_model2.DQN(env.observation_space.shape,
                             env.action_space.n).to(device)

    writer = SummaryWriter(comment="-" + args.env)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None

    while True:

        reward = agent.play_step(net, epsilon, device)
        if len(buffer) < REPLAY_START_SIZE:
            print("\033[F\033[  WWarning! buffering: " +
                  str(round(len(buffer)/REPLAY_START_SIZE*100, 2))+"%")
            continue
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START -
                      frame_idx/EPSILON_DECAY_LAST_FRAME)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx-ts_frame) / (time.time()-ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = np.mean(total_rewards[-100:])
            print("{} done {} games, mean reward {}, eps {}, speed {} f/s"
                  .format(frame_idx, len(total_rewards), round(mean_reward, 3), round(epsilon, 3), round(speed, 3)))
            writer.add_scalar("epsilon", round(epsilon, 3), frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", round(mean_reward, 3), frame_idx)
            writer.add_scalar("reward", round(reward, 3), frame_idx)

            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), args.env + "-best.dat")
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f, model saved" %
                          (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
            if mean_reward > args.reward:
                print("Solved in %d frames!" % frame_idx)
                break

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())
        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()
    writer.close()
