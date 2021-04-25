import gym
import ptan
import argparse
import torch
import torch.optim as optim
import ptan.ignite as ptan_ignite
import torch.multiprocessing as mp

from tensorboardX import SummaryWriter
from lib import dqn_model, common
import numpy as np

from ignite.engine import Engine

STATES_TO_EVALUATE = 1000
EVAL_EVERY_FRAME = 100

from collections import namedtuple

BATCH_MUL = 8
EpisodeEnded = namedtuple(
    'EpisodeEnded', field_names=('reward', 'steps', 'epsilon'))
def play_func(params, net, cuda, exp_queue):
    env = gym.make(params.env_name)
    env = ptan.common.wrappers.wrap_dqn(env)
    env.seed(common.SEED)
    device = torch.device("cuda" if cuda else "cpu")
    selector = ptan.actions.EpsilonGreedyActionSelector(
        epsilon=params.epsilon_start)
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=params.gamma)
    for frame_idx, exp in enumerate(exp_source):
        epsilon_tracker.frame(frame_idx/BATCH_MUL)
        exp_queue.put(exp)
        for reward, steps in exp_source.pop_rewards_steps():
            exp_queue.put(EpisodeEnded(reward, steps,
                                       selector.epsilon))

class BatchGenerator:
    def __init__(self,
                 buffer: ptan.experience.ExperienceReplayBuffer,
                 exp_queue,
                 fps_handler: ptan_ignite.EpisodeFPSHandler,
                 initial: int, batch_size: int):
        self.buffer = buffer
        self.exp_queue = exp_queue
        self.fps_handler = fps_handler
        self.initial = initial
        self.batch_size = batch_size
        self._rewards_steps = []
        self.epsilon = None
    def pop_rewards_steps(self):
        res = list(self._rewards_steps)
        self._rewards_steps.clear()
        return res
    def __iter__(self):
        while True:
            while exp_queue.qsize() > 0:
                exp = exp_queue.get()
                if isinstance(exp, EpisodeEnded):
                    self._rewards_steps.append((exp.reward,
                                                exp.steps))
                    self.epsilon = exp.epsilon
                else:
                    self.buffer._add(exp)
                    self.fps_handler.step()
            if len(self.buffer) < self.initial:
                continue
            yield self.buffer.sample(self.batch_size * BATCH_MUL)



if __name__ == "__main__":

    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default='pong', help="Environment to solve")
    parser.add_argument("--cuda", default=True,
                        action="store_true", help="Enable cuda")
    parser.add_argument(
        "--n", default=1,  help="how many steps to unroll from the bellman equation")
    parser.add_argument(
        "--envs", default=3, type=int, help="how many environments to use for generating training examples")
    parser.add_argument("--double", default=False,
                        action="store_true", help="Enable double DQN")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    params = common.HYPERPARAMS[args.env]
    envs = []
    for _ in range(args.envs):
        env = gym.make(params.env_name)
        env = ptan.common.wrappers.wrap_dqn(env)
        env.seed(common.SEED)
        envs.append(env)
    params.batch_size *= args.envs
    

    
    
    
    net = dqn_model.DQN(env.observation_space.shape,
                        env.action_space.n).to(device)
    tgt_net = ptan.agent.TargetNet(net)

    selector = ptan.actions.EpsilonGreedyActionSelector(
        epsilon=params.epsilon_start)

    epsilon_tracker = common.EpsilonTracker(selector, params)

    agent = ptan.agent.DQNAgent(net, selector, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(
        envs, agent, gamma=params.gamma)

    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, buffer_size=params.replay_size)

    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)
    exp_queue = mp.Queue(maxsize=BATCH_MUL*2)
    play_proc = mp.Process(target=play_func, args=(params, net,
                                                   args.cuda,
                                                   exp_queue))
    play_proc.start()
    fps_handler = ptan_ignite.EpisodeFPSHandler()
    batch_generator = BatchGenerator(buffer, exp_queue, fps_handler, params.replay_initial, params.batch_size)
    def process_batch(engine, batch):
        optimizer.zero_grad()
        loss_v = common.calc_loss_dqn(
            batch, net, tgt_net.target_model, gamma=params.gamma, device=device)
        loss_v.backward()
        optimizer.step()
        epsilon_tracker.frame(engine.state.iteration)
        if engine.state.iteration%params.target_net_sync==0:
            print('syncing...')
            tgt_net.sync()
        
        return {'loss':loss_v.item(), "epsilon":batch_generator.epsilon}


    engine = Engine(process_batch)
    ptan_ignite.EndOfEpisodeHandler(batch_generator, bound_avg_reward=17.0).attach(engine)
    fps_handler.attach(engine, manual_step=True)
    common.setup_ignite(engine, params,exp_source, params.run_name)

    #engine.run(common.batch_generator(buffer, params.replay_initial, params.batch_size, 1))
    engine.run(batch_generator)
    play_proc.kill()
    play_proc.join()






