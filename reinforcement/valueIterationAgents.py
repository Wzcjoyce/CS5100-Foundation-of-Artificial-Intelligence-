# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        # iteration counter
        current_iteration = 0

        # when current iteration is less than the given iteration times stay in the while loop otherwise leave the loop
        while current_iteration < self.iterations:
            # setup the temporary action_value dictionary
            actions_value = util.Counter()

            # loop through all possible states
            for state in self.mdp.getStates():

                # set the initial max value as a extremely small negative value
                max = -99999999

                # loop through all possible actions from the given state using the given
                # getPossibleActions function
                for action in self.mdp.getPossibleActions(state):
                    # calculate the q value for the specific state and action pair
                    Q_Value = self.computeQValueFromValues(state, action)

                    # if the q value is larger than the max value, max value will be replaced by this q value
                    if Q_Value > max:
                        max = Q_Value
                        # save this q value into the temporary dictionary
                        actions_value[state] = Q_Value
            # assign the latest temporary dictionary into the self.values
            self.values = actions_value

            # counter plus 1
            current_iteration += 1



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        # initialize the initial q value
        QValue = 0

        # call the given transitionStateAndProbs function to get a tuple of next state and probability
        for next_state, probability in self.mdp.getTransitionStatesAndProbs(state, action):
            # get the reward using getReward function for the state, next_state and action
            reward = self.mdp.getReward(state, action, next_state)

            # get the discount rate
            discount_rate = self.discount

            # calculat the value of times the discount rate
            value_after_discount = discount_rate * self.values[next_state]

            # calculate the sum of q value for all possible transitions
            QValue += probability * (reward + value_after_discount)

        # return the q value
        return QValue

        #util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        # check the terminal state as per the description above
        if self.mdp.isTerminal(state):
            return None

        # set a small negative max value at beginning
        max = -999999999

        # initial variable for holding the selected action
        selected_action = None

        # iterate through all possible actions and calcuate the q value using computeQValueFromValues function
        for action in self.mdp.getPossibleActions(state):
            QValue = self.computeQValueFromValues(state, action)

            # if the q value is larger than the max value, max value will be replaced by this q value
            # selected action will be replaced by the current action
            if QValue > max:
                max = QValue
                selected_action = action

        # if the selecte section is still none, return the last possible action
        if selected_action == None:
            return self.mdp.getPossibleActions(state)[len(self.mdp.getPossibleActions(state))]

        # return the selected action
        return selected_action

        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

