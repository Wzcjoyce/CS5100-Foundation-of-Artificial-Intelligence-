# multiAgents.py
# --------------
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
from statistics import mean

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        from util import manhattanDistance

        # convert newFood to a list by using asList() function
        food_locations_list = newFood.asList();

        # set the distance to closest food to very large number
        closest_food_distance = 99999999

        # for loop to find the distance to the closest food
        for each_food in food_locations_list:
            food_pacman_distance = manhattanDistance(each_food, newPos)
            if food_pacman_distance < closest_food_distance:
                closest_food_distance = food_pacman_distance

        # get sucessor's ghost position
        ghosts_location_list = successorGameState.getGhostPositions()

        # set the distance to closest ghost to very large number
        closest_ghost_distance = 999999

        # for loop to find the distance to the ghosts
        for each_ghost_location in ghosts_location_list:
            ghost_pacman_distance = manhattanDistance(each_ghost_location, newPos)
            if ghost_pacman_distance < closest_ghost_distance:
                closest_ghost_distance = ghost_pacman_distance

        # if the distance between successor and the cloest ghost is less than 2 unit return a negative evaluation value
        if closest_ghost_distance < 2:
            return - 999999999

        '''
        if the evaluation score is larger than 0 (2 unit distance away from the cloest ghost), 
        return the current score plus the cloest ghost distance divied by closest food distance
        since the bigger the cloest ghost distance and the lower the cloest food distance the successor have
        the better evaluation score
        '''
        if successorGameState.getScore() > 0:
            if closest_ghost_distance > closest_food_distance:
                return successorGameState.getScore() + closest_ghost_distance/closest_food_distance
            else:
                return successorGameState.getScore()

        # return the score
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # call the minimax_decision function below and save the returned action and return it
        # Since the minimax_decision function handle the max value function at depth 0 for pacman(agentIndex = 0)
        # so that, the parameter passed in are depth = 1  and agentIndex = 1
        chosen_action = self.minimax_decision(gameState, 1, 1)
        return chosen_action
        #util.raiseNotDefined()

    def minimax_decision(self, gameState, cur_depth, agentIndex):
        '''

        :param gameState: current game state
        :param cur_depth: current depth in the tree
        :param agentIndex: agent ID, 0 is pacman, all other number are ghosts
        :return: a best action based on minimax decision
        '''

        # get leagal actions list by calling the provided getLegalActions function
        legal_actions = gameState.getLegalActions(self.index)

        max_score = -999999999

        temp_action_list = []

        # call the min value which is recursively call max_value and min_value and eventually return the action with max evaluation score
        for each_legal_action in legal_actions:
            next_state = gameState.generateSuccessor(self.index, each_legal_action)

            # call min_value function for ghost
            score = self.min_value(next_state, cur_depth, agentIndex)
            if max_score < score:
                max_score = score
                temp_action_list.append(each_legal_action)

        return temp_action_list[-1]


    def max_value(self, gameState, cur_depth, agentIndex):
        '''
        :param gameState: current game state
        :param cur_depth: current depth in the tree
        :param agentIndex: agent ID, should always be 0 in this fucntion as max value is used for pacman
        :return: the max evaluation value
        '''

        # terminal test
        self.terminal_test(gameState)

        # reach the depth limiation, return the evaluation value of current game state
        if cur_depth == self.depth:
            return self.evaluationFunction(gameState)

        # Since this is the max value which is used for pacman which means
        # next will be min value and depth should add 1 for ghost agent
        cur_depth += 1

        # pacman agent should be zero
        pacman_index = self.index

        # get legal actions for pacman
        action_list = gameState.getLegalActions(pacman_index)

        # if there is no legal action, return the evaluation value for the current game state
        if len(action_list) == 0:
            return self.evaluationFunction(gameState)

        max = -9999999

        # iterate through the action list and call the min_value function from ghost agent
        for each_legal_action in action_list:
            temp_value = self.min_value(gameState.generateSuccessor(pacman_index, each_legal_action), cur_depth, pacman_index + 1)
            if temp_value > max:
                max = temp_value

        return max

    def min_value(self, gameState, cur_depth, agentIndex):
        '''
        :param gameState: current game state
        :param cur_depth: current depth in the tree
        :param agentIndex: agent ID, should not be zero as this function is for ghost agent
        :return: the minimum evaluation value
        '''

        # find the index of the last agent by reducing the number of agents by 1
        last_agent_index = gameState.getNumAgents() - 1

        # call the terminal test
        self.terminal_test(gameState)

        # get the legal actions list
        action_list = gameState.getLegalActions(agentIndex)


        min = 9999999

        # if there is no legal actions, return the evaluation value from current game state
        if len(action_list) == 0:
            return self.evaluationFunction(gameState)

        # if the current agent index is not the last agnet index
        # agent index number increase by one and recursively call min_value as the next agent is still ghost agent
        if agentIndex != last_agent_index:
            for each_legal_action in action_list:
                temp_value = self.min_value(gameState.generateSuccessor(agentIndex, each_legal_action), cur_depth, agentIndex + 1)
                if temp_value < min:
                    min = temp_value

        # if the current agent index is the last agnet index
        # call the max value since the next agent will be pacman agent
        else:
            for each_legal_action in action_list:
                temp_value = self.max_value(gameState.generateSuccessor(agentIndex, each_legal_action), cur_depth, agentIndex)
                if temp_value < min:
                    min = temp_value

        return min

    # terminal test which test if the game is end by checking isWin and isLost functions
    def terminal_test(self, gameState):
        '''
        :param gameState: current game state
        :return: boolean if the game has reached the end
        '''
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # set alpha as a negative value and beta as a very large number at beginning
        a = -99999
        b = 99999


        # call the alpha_beta_pruning_decision function below and save the returned action and return it
        # Since the alpha_beta_pruning_decision function handle the max value function at depth 0 for pacman(agentIndex = 0)
        # so that, the parameter passed in are depth = 1  and agentIndex = 1
        chosen_action = self.alpha_beta_pruning_decision(gameState, 1, 1, a, b)
        return chosen_action
        # util.raiseNotDefined()

    def alpha_beta_pruning_decision(self, gameState, cur_depth, agentIndex, a, b):
        '''
        :param gameState: current game state
        :param cur_depth: current depth in the tree
        :param agentIndex: agent ID, should not be zero as this function is for ghost agent
        :param a: alpha value which is the best highest value
        :param b: beta value which is the best lowest value
        :return: the best action returned by alpha beta pruning
        '''

        # get leagal actions list by calling the provided getLegalActions function
        legal_actions = gameState.getLegalActions(self.index)

        max_score = -99999

        temp_action_list = []

        # iterate through the action list and call the min_value function from ghost agent
        for each_legal_action in legal_actions:
            next_state = gameState.generateSuccessor(self.index, each_legal_action)
            score = self.min_value(next_state, cur_depth, agentIndex, a, b)
            if max_score < score:
                max_score = score
                temp_action_list.append(each_legal_action)

            # if the score is larger than beta, we don't need to check the rest of action
            # and we can directly return the action
            if score > b:
                return temp_action_list[-1]

            a = max(a, score)

        return temp_action_list[-1]

    def max_value(self, gameState, cur_depth, agentIndex, a, b):
        '''
        :param gameState: current game state
        :param cur_depth: current depth in the tree
        :param agentIndex: agent ID, should be zero as this function is for pacman agent
        :param a: alpha value which is the best highest value
        :param b: beta value which is the best lowest value
        :return: the minimum evaluation value
        '''

        # terminal test
        self.terminal_test(gameState)

        # reach the depth limiation, return the evaluation value of current game state
        if cur_depth == self.depth:
            return self.evaluationFunction(gameState)

        cur_depth += 1

        pacman_index = self.index

        # get leagal actions list by calling the provided getLegalActions function
        action_list = gameState.getLegalActions(pacman_index)

        # if there is no legal actions, return the evaluation value from current game state
        if len(action_list) == 0:
            return self.evaluationFunction(gameState)

        max = -9999999

        new_a = a

        for each_legal_action in action_list:
            temp_value = self.min_value(gameState.generateSuccessor(pacman_index, each_legal_action), cur_depth,
                                        pacman_index + 1, new_a, b)
            if temp_value > max:
                max = temp_value

            # if the score is larger than beta, we don't need to check the rest of action
            # and we can directly return the max value we have
            if max > b:
                return max

            # updating the alpha value to make sure it is always the max value
            if new_a < max:
                new_a = max

        return max

    def min_value(self, gameState, cur_depth, agentIndex, a, b):
        '''
        :param gameState: current game state
        :param cur_depth: current depth in the tree
        :param agentIndex: agent ID, should not be zero as this function is for ghost agent
        :param a: alpha value which is the best highest value
        :param b: beta value which is the best lowest value
        :return: the minimum evaluation value
        '''

        last_agent_index = gameState.getNumAgents() - 1

        # terminal test
        self.terminal_test(gameState)

        # get leagal actions list by calling the provided getLegalActions function
        action_list = gameState.getLegalActions(agentIndex)
        min = 9999999

        # if there is no legal actions, return the evaluation value from current game state
        if len(action_list) == 0:
            return self.evaluationFunction(gameState)

        new_b = b

        # if it is not the last agent index we should call min value function
        # as the next agent will be ghost agent and agent index will add 1
        if agentIndex != last_agent_index:
            for each_legal_action in action_list:
                temp_value = self.min_value(gameState.generateSuccessor(agentIndex, each_legal_action), cur_depth,
                                            agentIndex + 1, a, new_b)
                if temp_value < min:
                    min = temp_value

                # if the score is smaller than alpha, we don't need to check the rest of action
                # and we can directly return the min value we have
                if min < a:
                    return min

                # updating the beta value to make sure it is always the min value
                if new_b > min:
                    new_b = min

        # if it is the last agent index we should call max value function for ghost
        # since the next agent will be pacman
        else:
            for each_legal_action in action_list:
                temp_value = self.max_value(gameState.generateSuccessor(agentIndex, each_legal_action), cur_depth,
                                            agentIndex, a, new_b)

                if temp_value < min:
                    min = temp_value

                # if the score is smaller than alpha, we don't need to check the rest of action
                # and we can directly return the min value we have
                if min < a:
                    return min

                # updating the beta value to make sure it is always the min value
                if new_b > min:
                    new_b = min

        return min

    # terminal test which test if the game is end by checking isWin and isLost functions
    def terminal_test(self, gameState):
        '''
        :param gameState: current game state
        :return: boolean if the game has reached the end
        '''
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        #util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # call the expectimax_decision function below and save the returned action and return it
        # Since the expectimax_decision function handle the max value function at depth 0 for pacman(agentIndex = 0)
        # so that, the parameter passed in are depth = 1  and agentIndex = 1
        chosen_action = self.expectimax_decision(gameState, 1, 1)
        return chosen_action


    def expectimax_decision(self, gameState, cur_depth, agentIndex):
        '''
        :param gameState: current game state
        :param cur_depth: current depth in the tree
        :param agentIndex: agent ID, should not be zero as this function is for ghost agent
        :return: the action based on expectimax decision
        '''

        # get leagal actions list by calling the provided getLegalActions function
        legal_actions = gameState.getLegalActions(self.index)

        max_score = -999999999

        temp_action_list = []

        for each_legal_action in legal_actions:
            next_state = gameState.generateSuccessor(self.index, each_legal_action)
            score = self.expect_value(next_state, cur_depth, agentIndex)
            if max_score < score:
                max_score = score
                temp_action_list.append(each_legal_action)

        return temp_action_list[-1]

    def max_value(self, gameState, cur_depth, agentIndex):
        '''
        :param gameState: current game state
        :param cur_depth: current depth in the tree
        :param agentIndex: agent ID, should be zero as this function is for pacman agent
        :return: the higest evalucation
        '''

        self.terminal_test(gameState)

        if cur_depth == self.depth:
            return self.evaluationFunction(gameState)

        cur_depth += 1

        pacman_index = self.index

        # get leagal actions list for pacman by calling the provided getLegalActions function
        action_list = gameState.getLegalActions(pacman_index)

        # if there is no legal actions, return the evaluation value from current game state
        if len(action_list) == 0:
            return self.evaluationFunction(gameState)

        max = -9999999

        for each_legal_action in action_list:
            temp_value = self.expect_value(gameState.generateSuccessor(pacman_index, each_legal_action), cur_depth,
                                        pacman_index + 1)
            if temp_value > max:
                max = temp_value

        return max

    def expect_value(self, gameState, cur_depth, agentIndex):
        '''
        :param gameState: current game state
        :param cur_depth: current depth in the tree
        :param agentIndex: agent ID, should be zero as this function is for pacman agent
        :return: the higest evalucation
        '''

        last_agent_index = gameState.getNumAgents() - 1

        # terminal test
        self.terminal_test(gameState)

        # get leagal actions list for ghost by calling the provided getLegalActions function
        action_list = gameState.getLegalActions(agentIndex)
        expect = 9999999

        # if there is no legal actions, return the evaluation value from current game state
        if len(action_list) == 0:
            return self.evaluationFunction(gameState)

        expect_value_list = []

        # if it is not the last agent, means the next agent is ghost
        # we should call expect_value function, add agent index by 1 and also append the expect value into the expect_value_list
        if agentIndex != last_agent_index:
            for each_legal_action in action_list:
                temp_value = self.expect_value(gameState.generateSuccessor(agentIndex, each_legal_action), cur_depth,
                                            agentIndex + 1)
                expect_value_list.append(temp_value)

        # if it is the last agent, means the next agent is pacman
        # we should call the max_value function and append the expect value into the expect_value_list
        else:
            for each_legal_action in action_list:
                temp_value = self.max_value(gameState.generateSuccessor(agentIndex, each_legal_action), cur_depth,
                                            agentIndex)
                expect_value_list.append(temp_value)

        # calculate the average expect valuein the list
        mean_expect_value = mean(expect_value_list)

        return mean_expect_value

    # terminal test which test if the game is end by checking isWin and isLost functions
    def terminal_test(self, gameState):
        '''
        :param gameState: current game state
        :return: boolean if the game has reached the end
        '''
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    # similar approach to questio 1 until line 661
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


    from util import manhattanDistance

    food_locations_list = newFood.asList();
    closest_food_distance = 99999999

    # for loop to find the distance to the closest food
    for each_food in food_locations_list:
        food_pacman_distance = manhattanDistance(each_food, newPos)
        if food_pacman_distance < closest_food_distance:
            closest_food_distance = food_pacman_distance

    ghosts_location_list = currentGameState.getGhostPositions()

    closest_ghost_distance = 999999

    # for loop to find the distance to the nearest ghost
    for each_ghost_location in ghosts_location_list:
        ghost_pacman_distance = manhattanDistance(each_ghost_location, newPos)
        if ghost_pacman_distance < closest_ghost_distance:
            closest_ghost_distance = ghost_pacman_distance

    # if the distance between successor and the cloest ghost is less than 2 unit return a negative evaluation value
    if closest_ghost_distance < 2:
        return - 999999999

    # get the location all capsules
    capsules_location_list = currentGameState.getCapsules()
    closest_capsule_distance = 99999999

    # for loop to find the distance to the nearest capsules
    for each_capsules in capsules_location_list:
        capsules_pacman_distance = manhattanDistance(each_capsules, newPos)
        if capsules_pacman_distance < closest_capsule_distance:
            closest_capsule_distance = capsules_pacman_distance

    # sum the scared time
    total_scared_time = sum(newScaredTimes)


    if currentGameState.getScore() > 0:

        # if there is no scared time, the higher cloest ghost distance, the lower closest food distance and the closest capsule distance
        # the better
        if total_scared_time == 0:
            if closest_ghost_distance > closest_food_distance:
                # add the closest capsule distance to the denominator
                return currentGameState.getScore() + (closest_ghost_distance / (closest_food_distance + closest_capsule_distance))

            else:
                return currentGameState.getScore()

        # if there is scared time, pacman should looking for ghost instead of food
        # so that, the higher the distance to closest food and the lower the distance to the closest ghost
        # the better
        else:
            return currentGameState.getScore() + closest_food_distance / closest_ghost_distance



    return currentGameState.getScore()
    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
