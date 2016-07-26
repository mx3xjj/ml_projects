import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import namedtuple

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        # sets self.env = env, state = None, next_waypoint = None, and a default color
        super(LearningAgent, self).__init__(env) 
        self.state = None 
        # override color
        self.color = 'red' 
        # simple route planner to get next_waypoint
        self.planner = RoutePlanner(self.env, self)  
        # TODO: Initialize any additional variables here
        # learning rate
        # The learning rate determines to what extent the newly acquired information will override the old information. 
        # A factor of 0 will make the agent not learn anything, while a factor of 1 would make the agent consider only the most recent information. 
        self.alpha = 0.8
        # discount rate
        # The discount factor gamme determines the importance of future rewards. 
        # A factor of 0 will make the agent "myopic" (or short-sighted) by only considering current rewards, 
        # while a factor approaching 1 will make it strive for a long-term high reward.
        self.gamma = 0.7
        # all possible actions- None, 'forward', 'left', 'right'
        self.pa = ['forward', 'left', 'right', None]
        # Traffic light possibilities
        self.tl = ['green', 'red']
        self.step = 0
        # Q learner dictionary
        self.q_learner = {}
        # Initialize q learner values
        for pa in self.pa:
            for signal in self.tl:
                for action in self.env.valid_actions:
                    self.q_learner[((pa, signal), action)] = 100    
     
    def state_tuple(self, state):
        state_t = namedtuple("State", ["light", "next_waypoint"])
        return state_t(light = state['light'], next_waypoint = self.planner.next_waypoint())
  
    def q_max_action(self, state):
        action = None
        q_max = 0.0
        for a in self.env.valid_actions:
            q = self.q_learner[(state, a)]
            if q > q_max:
                action = a
                q_max = q
        return (q_max, action)

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.state = None        

    def update(self, t):
        action = None
        q_max = 0.0
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        # Define the deadline
        deadline = self.env.get_deadline(self)
        # TODO: Update state
        self.state = (self.next_waypoint, inputs['light'])
        # TODO: Select action according to your policy      
        for i in self.env.valid_actions:
            q = self.q_learner[(self.state, i)]
            if q > q_max:
                action = i
                q_max = q
        # Execute action and get reward
        reward = self.env.act(self, action)
        # TODO: Learn policy based on state, action, reward
        # epsilon = 0.1 
        # if random.uniform(0,1) < epsilon: #Exploration/Exploitation Trade-off
        #     action = random.choice(['forward', 'left', 'right', None])
        # else:
        (q_sub, action_sub) = self.q_max_action((self.next_waypoint, inputs['light']))        
        # Q learning formula
        q = q + self.alpha * (reward + self.gamma * q_sub - q) 
        self.q_learner[(self.state, action)] = q
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

def run():
    """Run the agent for a finite number of trials."""
    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track
    # Now simulate it
    sim = Simulator(e, update_delay=0.00000000001)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit

if __name__ == '__main__':
    run()
