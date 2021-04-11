print("our new code!")
from lib import wrappers
from lib import tf_dqn_model

import argparse
import time
import numpy as np
import collections

import tensorflow as tf


from tensorboardX import SummaryWriter


DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.0

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 32#10000
REPLAY_START_SIZE = 32#10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000


loss_object = tf.keras.losses.MSE

optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)


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

        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.bool), np.array(next_states)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0):
        done_reward = None

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = tf.convert_to_tensor(state_a, tf.float32)
            q_vals_v = net(state_v)  # shape-> [1,n_actions]
            action = tf.math.reduce_max(q_vals_v, dim=1).numpy()[0]
            print(action)

        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward
        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


  
  

@tf.function
def train_step(batch, net, tgt_net):
    with tf.GradientTape() as tape:
        states, actions, rewards, dones, next_states = batch
        states_v = tf.convert_to_tensor(states, tf.float32)
        actions_v = tf.convert_to_tensor(actions, tf.int64)
        rewards_v = tf.convert_to_tensor(rewards, tf.float32)
        done_mask = tf.convert_to_tensor(dones, dtype=tf.bool)
        done_mask = tf.expand_dims(tf.cast(done_mask, dtype=tf.float32), axis=1)
        next_states_v = tf.convert_to_tensor(next_states, tf.float32)
        state_action_values = tf.squeeze(tf.gather_nd(net(states_v),tf.expand_dims(actions_v,1)),-1)

        next_states_values =  tf.math.reduce_max(tgt_net(next_states_v),1,False) #* done_mask


        expected_state_action_values = rewards_v + next_states_values * GAMMA

        loss = loss_object(state_action_values, expected_state_action_values)
        gradients = tape.gradient(loss, net.trainable_variables)
        optimizer.apply_gradients(zip(gradients, net.trainable_variables))
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help=f"Name of the environment, default={DEFAULT_ENV_NAME}")

    parser.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND,
                        help=f"Mean reward boundary fpr stopping the training, default={MEAN_REWARD_BOUND}")
    args = parser.parse_args()

    env = wrappers.make_env(args.env)
    net = tf_dqn_model.DQN(env.observation_space.shape,
                         env.action_space.n)
    tgt_net = tf_dqn_model.DQN(env.observation_space.shape,
                             env.action_space.n)

    writer = SummaryWriter(comment="-" + args.env)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None

    while True:

        reward = agent.play_step(net, epsilon)
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
                net.save_weights(args.env + "-best.dat")
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f, model saved" %
                          (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
            if mean_reward > args.reward:
                print("Solved in %d frames!" % frame_idx)
                break

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.set_weights(net.get_weights())
        batch = buffer.sample(BATCH_SIZE)
        loss_t = train_step(batch, net, tgt_net)
    writer.close()
