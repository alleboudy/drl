#!/usr/bin/env python3
import gym
import ptan
import argparse
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from lib import dqn_model, common,wrappers
import numpy as np
STATES_TO_EVALUATE = 1000
EVAL_EVERY_FRAME = 100


if __name__ == "__main__":
    params = common.HYPERPARAMS['pong']
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True, action="store_true", help="Enable cuda")
    parser.add_argument("--n", default=1,  help="how many steps to unroll from the bellman equation")
    parser.add_argument("--double", default=False, action="store_true", help="Enable double DQN")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    unrolling_steps = int(args.n) if args.n else 1
    print('bellman unrolling steps: '+ str(unrolling_steps))
    double = args.double if args.double else False
    #env = gym.make(params['env_name'])
    #ptan.common.wrappers.wrap_dqn(env)
    env = wrappers.make_env(params['env_name'])
    writer = SummaryWriter(comment="-" + params['run_name'] + "-basic")
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=unrolling_steps)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params['replay_size'])
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])
    frame_idx = 0
    eval_states = None
    with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            frame_idx += 1
            buffer.populate(1)#where all the magic happens!
            # 1- the buffer will ask the experience source to produce a transision s,a,R,s'
            # 2- the experience source will feed the current observation s to the agent
            # 3- the agent will feed the observation to the network, get the q values of the observation and ask the action selector to decide an action
            # 4- the action selector will generate a random value, compare it to epsilon and decide whether to act greedy or randomly decide the action
            # 5- the action decided is passed to the experience source which feeds it to the environment to get the reward r and new state s' and now, s,a,R,s' are passed to the buffer
            # 6- the buffer takes in the s,a,R,s' data in and kicks out an old one to maintain the same size
            epsilon_tracker.frame(frame_idx)
            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    #If the reward tracker returns True, then it's an indication that the mean reward has reached the score boundary and we can stop our training.
                    break
            
            if len(buffer) < params['replay_initial']:# we need to fill the buffer before training so that we have episodes to train on
                continue
            if eval_states is None:
                eval_states = buffer.sample(STATES_TO_EVALUATE)
                eval_states = [np.array(transition.state, copy=False) for transition in eval_states]
                eval_states = np.array(eval_states, copy=False)
            if frame_idx % EVAL_EVERY_FRAME == 0:
                mean_val = common.calc_values_of_states(eval_states, net, device=device)
                writer.add_scalar("values_mean", mean_val, frame_idx)
            # taking a training step!
            optimizer.zero_grad()
            batch = buffer.sample(params['batch_size'])
            loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=params['gamma']**unrolling_steps, double = double, device=device)
            loss_v.backward()
            optimizer.step()
            if frame_idx % params['target_net_sync'] == 0:#when to sync our target network
                tgt_net.sync()