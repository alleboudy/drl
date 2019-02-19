import gym
import collections
from tensorboardX import SummaryWriter


""" 
Main data structures we use:
1- Reward Dictionary key:(state,action) value:immediate reward
2- Transition table key:(state,action) value: Dictionary {key:landingState, val:count<how many times we landed in this state by executing that action in theat starting state>}
3- Value Dictionary key:state value: 
V(s) = max_{a\inA} Q(s,a)
         = max_{a\inA} E_{s'~S}[r(s,a)+gamma*V(s')]
         = max{a\inA} Sum_s'\inS P_a,s->s' * [r_{s,a,s'}+V_s']

1- we play 100 random steps to populate the reward and transition tables
2- perform a value iteration loop over all states to update the value table
3- play several full episodes using the updated value table to compute an average reward for those episodes -> during that we also update the reward and transition tables, unlike cross entropy where we did not update our weights til the end of the full episode
4- if the average reward is > 0.8, stop training.

"""

ENV_NAME = "FrozenLake8x8-v0"
GAMMA = 0.9
TEST_EPISODES = 20


class Agent():
    """ Our agent for the environment frozen lake v0"""
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)
    def play_n_random_steps(self,count):
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done, _ = self.env.step(action)
            self.rewards[(self.state,action,new_state)]  = reward
            self.transits[(self.state,action)][new_state]+=1
            self.state = self.env.reset() if is_done else new_state
        
    # def calc_action_value(self, state, action):#not needed in the Q learning anymore 
    #     """ Given a state and action, this function returns the Q(s,a)
    #     A) Estimating the transition probabilities of the action a from the state s to the possible landing state s'
    #         1- get the transitions (s,a)->{s':count}
    #         2- sum the counts to get the total landings
    #         3- normalize the counts into probabilities
    #     B) Compute the Q(s,a) using the bellman equation
    #     Q(s,a) = E_{s'~S}[r_{s,a,s'}+GMMA*V_s']
    #            = Sum_{s'\inS}P{a,s->s'} *[r_{s,a,s'}+GAMMA *V_s']
    #         so, for each s' landing state we get from the transitions table[(s,a)]
    #         1- compute P_{a,s->s'} *( r_{s,a,s'}+GAMMA V_s')
    #         2- sum the (1)s of all landing states to get the total Q(s,a)

    #     """
    #     target_counts = self.transits[(state,action)]
    #     total = sum(target_counts.values())
    #     action_value = 0.0
    #     for tgt_state, count in target_counts.items():
    #         reward = self.rewards[(state,action,tgt_state)]
    #         action_value +=(count/total) * (reward + GAMMA * self.values[tgt_state])
    #     return action_value

            
    def select_action(self,state):
        """ from a given state, for all possible actions from that state,
         compute the Q value of each action Q(s,a)
          using the calc_action_value and
           select the maximum action to return 
            We select the max, as we already do the exploration separately up in the play_n_random_steps
            a different implementaiton would be to do them using an epsilon -> exploration vs exploitation rate
        ---------------
            in the Q learning we no longer need to calculate the action value here to determine the best, just pick it from the stored Q table

        """
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value=self.values[(state,action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action


    def play_episode(self,env):
        """ For the test episodes, so, it uses a different copy of the environemt so that we don't change the environment used to gather the random data
            uses the select_action function to select the best action to take
            note that here we do update the transition and reward dictionaries although it is just testing!
             """
        total_reward = 0.0
        state = env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, is_done, _ = env.step(action)
            self.rewards[(state,action,new_state)] = reward
            self.transits[(state,action)][new_state] +=1
            total_reward +=reward
            if is_done:
                break
            state = new_state
        return total_reward


    def value_iteration(self):
        """ For each state in our environment, computes the V(s) """
        for state in range(self.env.observation_space.n):# we do so for each state we have in the environment
            for action in range(self.env.action_space.n):
                action_value = 0.0
                target_counts = self.transits[(state,action)]
                total = sum(target_counts.values())
                for tgt_state, count in target_counts.items():
                    reward = self.rewards[(state,action,tgt_state)]
                    best_action  = self.select_action(tgt_state)
                    action_value += (count/total) * (reward + GAMMA * self.values[(tgt_state,best_action)])
                self.values[(state,action)] = action_value

            


            
if __name__=="__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-v-learning")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no +=1
        agent.play_n_random_steps(100)
        agent.value_iteration()
        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward +=agent.play_episode(test_env)
        reward /=TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        if reward >best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward> 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()
    test_env.close()
    agent.env.close()




