# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()

    # initiate empty list for the visited_list and culmulative_action
    visit_list = []
    culmulative_action = []

    # check if the start state is at goal state
    if problem.isGoalState(problem.getStartState()):
        return culmulative_action

    # initiate Stack for location coordinate
    dfs_stack = util.Stack()
    # push start coordinate into Stack
    dfs_stack.push(problem.getStartState())

    # initiate Stack for action
    action_stack = util.Stack()
    # push start action list(empty) into Stack
    action_stack.push(culmulative_action)

    # pop the start node
    curr_node = dfs_stack.pop()
    culmulative_action = action_stack.pop()

    # leave the while loop until goal is achieved
    while problem.isGoalState(curr_node) == False:

        # make sure to only expand the node that has not been visited yet
        if curr_node not in visit_list:
            visit_list.append(curr_node)
            for each_successor in problem.getSuccessors(curr_node):
                state, action, cost = each_successor
                temp_list = []
                temp_list.append(action)
                temp_action = culmulative_action + temp_list
                dfs_stack.push(state)
                action_stack.push(temp_action)

        if dfs_stack.isEmpty() == True:
            break

        # pop from the associated stack
        curr_node = dfs_stack.pop()
        culmulative_action = action_stack.pop()

    return culmulative_action

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    '''
    # initiate empty list for the visited_list and culmulative_action
    visit_list = []
    culmulative_action = []

    # check if the start state is at goal state
    if problem.isGoalState(problem.getStartState()):
        return culmulative_action

    # initiate queue for location coordinate
    bfs_queue = util.Queue()
    # push the start location coordinate into the queue
    bfs_queue.push(problem.getStartState())

    # initiate queue for action
    action_queue = util.Queue()
    # push the start action list (empty list) into the queue
    action_queue.push(culmulative_action)

    # pop the start node
    curr_node = bfs_queue.pop()
    culmulative_action = action_queue.pop()

    # leave the while loop until goal is achieved
    while problem.isGoalState(curr_node) == False:

        # make sure to only expand the node that has not been visited yet
        if curr_node not in visit_list:
            visit_list.append(curr_node)
            for each_successor in problem.getSuccessors(curr_node):
                state, action, cost = each_successor
                temp_list = []
                temp_list.append(action)
                temp_action = culmulative_action + temp_list
                bfs_queue.push(state)
                action_queue.push(temp_action)

        if bfs_queue.isEmpty() == True:
            break

        # pop from the associated queue
        curr_node = bfs_queue.pop()
        culmulative_action = action_queue.pop()

    return culmulative_action
    '''
    frontier = util.Queue()
    explored = set()

    explored.add(problem.getStartState())

    frontier.push(problem.getStartState())
    ans = {problem.getStartState(): []}
    while not frontier.isEmpty():
        current = frontier.pop()
        if problem.isGoalState(current):
            return ans[current]
        for successor in problem.getSuccessors(current):
            if successor[0] not in explored:
                explored.add(successor[0])
                frontier.push(successor[0])
                ans[successor[0]] = ans[current] + [successor[1]]
    #util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # initiate empty list for the visited_list and culmulative_action
    visit_list = []
    culmulative_action = []

    # check if the start state is at goal state
    if problem.isGoalState(problem.getStartState()):
        return culmulative_action

    # initiate priority queue for location coordinate
    ucs_priority_queue = util.PriorityQueue()
    ucs_priority_queue.push(problem.getStartState(), 0)

    # initiate priority queue for action
    action_priority_queue = util.PriorityQueue()
    action_priority_queue.push(culmulative_action, 0)

    # initiate priority queue for cost
    cost_priority_queue = util.PriorityQueue()
    cost_priority_queue.push(0, 0)

    # pop the start node
    curr_node = ucs_priority_queue.pop()
    culmulative_action = action_priority_queue.pop()
    culmulative_cost = cost_priority_queue.pop()

    # leave the while loop until goal is achieved
    while problem.isGoalState(curr_node) == False:

        # make sure to only expand the node that has not been visited yet
        if curr_node not in visit_list:
            visit_list.append(curr_node)
            for each_successor in problem.getSuccessors(curr_node):
                state, action, cost = each_successor
                temp_list = []
                temp_list.append(action)
                temp_action = culmulative_action + temp_list

                # calculate the true cost g(n)
                new_cost_gn = culmulative_cost + cost

                ucs_priority_queue.push(state, new_cost_gn)
                action_priority_queue.push(temp_action, new_cost_gn)
                cost_priority_queue.push(new_cost_gn, new_cost_gn)

        if ucs_priority_queue.isEmpty() == True:
            break

        # pop from the associated priority queue
        curr_node = ucs_priority_queue.pop()
        culmulative_action = action_priority_queue.pop()
        culmulative_cost = cost_priority_queue.pop()

    return culmulative_action

    #util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # initiate empty list for the visited_list and culmulative_action and initial cost
    visit_list = []
    culmulative_action = []
    culmulative_cost = 0

    # check if the start state is at goal state
    if problem.isGoalState(problem.getStartState()):
        return culmulative_action

    # initiate priority queue for location coordinate
    astar_priority_queue = util.PriorityQueue()
    heuristic_cost = heuristic(problem.getStartState(), problem)
    astar_priority_queue.push(problem.getStartState(), heuristic_cost)

    # initiate priority queue for action
    action_priority_queue = util.PriorityQueue()
    action_priority_queue.push(culmulative_action, heuristic_cost)

    # initiate priority queue for cost
    cost_priority_queue = util.PriorityQueue()
    cost_priority_queue.push(culmulative_cost, heuristic_cost)

    # pop the start node
    curr_node = astar_priority_queue.pop()
    culmulative_action = action_priority_queue.pop()
    culmulative_cost = cost_priority_queue.pop()

    # leave the while loop until goal is achieved
    while problem.isGoalState(curr_node) == False:

        # make sure to only expand the node that has not been visited yet
        if curr_node not in visit_list:
            visit_list.append(curr_node)
            for each_successor in problem.getSuccessors(curr_node):
                state, action, cost = each_successor
                temp_list = []
                temp_list.append(action)
                temp_action = culmulative_action + temp_list

                # Calculate the true cost g(n)
                new_cost_gn = culmulative_cost + cost

                # using the heuristic function to calculate the heuristic cost h(n)
                heuristic_cost_hn = heuristic(state, problem)

                # Calculate the A* cast f(n) = g(n) + h(n)
                overall_cost_fn = new_cost_gn + heuristic_cost_hn

                # push the expanded state into the associated prioity queue
                astar_priority_queue.push(state, overall_cost_fn)
                action_priority_queue.push(temp_action, overall_cost_fn)
                cost_priority_queue.push(new_cost_gn, overall_cost_fn)

        if astar_priority_queue.isEmpty() == True:
            break

        # pop from the associated priority queue
        curr_node = astar_priority_queue.pop()
        culmulative_action = action_priority_queue.pop()
        culmulative_cost = cost_priority_queue.pop()

    return culmulative_action
    #util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
