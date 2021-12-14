# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        # setup the values dictionary
        self.values = util.Counter();

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # get and return the value by passing the key into the self.values dictionary
        # the key is a tuple of state and action pair
        return self.values[(state, action)]
        util.raiseNotDefined()


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # initiate a small
        max = -9999999

        # if there is no legal action return 0.0 as per requirement above
        if len(self.getLegalActions(state)) == 0:
            return 0.0

        # iterate through all leagal actions and calculate the q value using getQvalue fucntion
        for legalaction in self.getLegalActions(state):
            Q_value = self.getQValue(state, legalaction)

            # if the q value is larger than the max value, max value will be replaced by this q value
            if Q_value > max:
                max = Q_value

        # return the max value
        return max
        #util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"

        # initiate a small
        max = -9999999

        # initial variable for holding the selected action
        selected_action = None

        # if there is no legal action return None as per requirement above
        if len(self.getLegalActions(state)) == 0:
            return None

        # iterate through all leagal actions and calculate the q value using getQvalue fucntion
        for legalaction in self.getLegalActions(state):
            Q_value = self.getQValue(state, legalaction)

            # if the q value is larger than the max value, max value will be replaced by this q value
            # selected action will be replaced by the current legal action
            if selected_action == None or Q_value > max:
                max = Q_value
                selected_action = legalaction

        # return the selected action
        return selected_action

        #util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"

        # if there is no legal action return None as per requirement above
        if len(legalActions) == 0:
            return None

        # get the action by calling the pre-defined computeActionFromQValues function
        action = self.computeActionFromQValues(state)

        # as per the Hint, using the util.flipCoin(prob) and if true
        # ramdomly pick a action from the legal action list
        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)


        #util.raiseNotDefined()

        # return the action
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # get the q value by calling the pre-defined getQvalue function
        Q_Value = self.getQValue(state, action)

        # get the updated Q value using the formula in the lecture slide and assign it to the self.values dictionary
        self.values[(state, action)] = Q_Value + self.alpha *( reward + self.discount * self.computeValueFromQValues(nextState) - Q_Value )
        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['c'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()

        Q_Value = 0

        # get dictionary of state features by calling given method
        state_features = self.featExtractor.getFeatures(state, action)

        # iterate through the key of state_features dictionary
        for feature_key in state_features.keys():
            # pass the key into state_features and self.weights to get the value
            # and calculate the Q value
            Q_Value += state_features[feature_key] * self.weights[feature_key]

        return Q_Value


    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()

        # get dictionary of state features by calling given method
        state_features = self.featExtractor.getFeatures(state, action)

        weights_dict = self.weights

        # calculat the difference based on the equation given in the assignment page
        difference = (reward + self.discount * self.getValue(nextState)) - self.getQValue(state, action)

        # iterate through the weight dictionary and calculate using the given formula
        for weight_key in weights_dict:
            self.weights[weight_key]  = self.weights[weight_key] + self.alpha * difference * state_features[weight_key]





    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            # nothing added to here
            pass
