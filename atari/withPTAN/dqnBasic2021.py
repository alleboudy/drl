import gym
import ptan
import argparse
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from lib import dqn_model, common
import numpy as np

from ignite.engine import Engine

STATES_TO_EVALUATE = 1000
EVAL_EVERY_FRAME = 100





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default='pong', help="Environment to solve")
    parser.add_argument("--cuda", default=True,
                        action="store_true", help="Enable cuda")
    parser.add_argument(
        "--n", default=1,  help="how many steps to unroll from the bellman equation")
    parser.add_argument("--double", default=False,
                        action="store_true", help="Enable double DQN")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    params = common.HYPERPARAMS[args.env]
    
    env = gym.make(params.env_name)
    env = ptan.common.wrappers.wrap_dqn(env)
    env.seed(common.SEED)
    net = dqn_model.DQN(env.observation_space.shape,
                        env.action_space.n).to(device)
    tgt_net = ptan.agent.TargetNet(net)

    selector = ptan.actions.EpsilonGreedyActionSelector(
        epsilon=params.epsilon_start)

    epsilon_tracker = common.EpsilonTracker(selector, params)

    agent = ptan.agent.DQNAgent(net, selector, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=params.gamma)

    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, buffer_size=params.replay_size)

    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)

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
        
        return {'loss':loss_v.item(), "epsilon":selector.epsilon}


    engine = Engine(process_batch)
    common.setup_ignite(engine, params,exp_source, params.run_name)

    engine.run(common.batch_generator(buffer, params.replay_initial, params.batch_size))






