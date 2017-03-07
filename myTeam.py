# myTeam.py
# ---------
# Team DotDevourers
# Henry Pan, hepan@ucsc.edu
# Viraj Patel, vispatel@ucsc.edu
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import random, time, util
from util import nearestPoint
import distanceCalculator
from game import Directions
import game


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='AggressiveAgent', second='AggressiveAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########
class BaseAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """
    def __init__(self):
        self.index = 0
        self.treeDepth = int(2)

    def scoreEvaluationFunction(self, currentGameState):
        pacPosition = currentGameState.getPacmanPosition()
        ghostStates = currentGameState.getGhostStates()
        foodGrid = currentGameState.getFood()
        foodList = foodGrid.asList()
        capsulesList = currentGameState.getCapsules()

        scoreGhost = 0

        distanceFood = map(lambda x: (1.0 / util.manhattanDistance(x, pacPosition)), foodList) + [0]
        distanceCapsule = map(lambda x: (1.0 / util.manhattanDistance(x, pacPosition)), capsulesList) + [0]

        distanceFood = filter(lambda x: x not in distanceCapsule, distanceFood) + [0]

        for ghostState in ghostStates:
            distanceGhost = util.manhattanDistance(pacPosition, ghostState.getPosition())
            if distanceGhost == 0:
                break
            newScaredTimes = ghostState.scaredTimer

            if newScaredTimes > 0:
                scoreGhost += (1.0 / distanceGhost)
            else:
                scoreGhost -= (1.0 / distanceGhost)
        scoreFood = max(distanceFood)
        scoreCapsule = max(distanceCapsule)
        scoreTotal = currentGameState.getScore() + scoreFood + scoreGhost + scoreCapsule
        return scoreTotal

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        depth = self.treeDepth
        scores = []
        actions = gameState.getLegalActions(self.index)
        actions = filter(lambda x: x != 'Stop', actions)

        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            scores.append(self.minValue(successor, 1, depth))
        bestScore = max(scores)
        bestIndex = [index for index in range(len(scores)) if scores[index] == bestScore]
        choseIndex = random.choice(bestIndex)

        return actions[choseIndex]

    def maxValue(self, state, depth):
        if depth == 0 or state.isWin() or state.isLose():
            return self.scoreEvaluationFunction(state)
        actions = state.getLegatActions(0)
        actions = filter(lambda x: x != 'Stop', actions)
        value = -1000000
        for action in actions:
            successor = state.generateSuccessor(0, action)
            value = max(value, self.minValue(successor, 1, depth))
        return value

    def minValue(self, state, agent, depth):
        if depth == 0 or state.isWin() or state.isLose():
            return self.scoreEvaluationFunction(state)
        actions = state.getLegalActions(agent)
        actions = filter(lambda x: x != 'Stop', actions)
        values = []
        for action in actions:
            successor = state.generateSuccessor(agent, action)
            if agent == (state.getNumAgent() - 1):
                newdepth = depth - 1
                values.append(self.maxValue(successor, newdepth))
            else:
                newagent = agent + 1
                values.append(self.minValue(successor, newagent, depth))
        finalValue = sum(values) / len(actions)
        return finalValue


    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}


class AggressiveAgent(BaseAgent):
    """ This Agent prioritizes food and attempts to stay away from all other agents including teammate """
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)

    def evaluate(self, gameState, action):
        successorGameState = gameState.generateSuccessor(self.index, action)
        new_position = successorGameState.getPosition(self.index)
        team_mate = filter(lambda x: x != self.index, self.getTeam(successorGameState))
        opponents = self.getOpponents(successorGameState)

        food_list = self.getFood(successorGameState).asList()
        distFood = map(lambda x: (1.0 / self.getMazeDistance(x, new_position)), food_list) + [0]
        scoreFood = max(distFood)
        scoreTotal = successorGameState.getScore() + scoreFood
        return scoreTotal

    def chooseAction(self, gameState):
        """ Calculate action to get closest food which increases priority
            as less food exists and attempts to maintain a safe distance from known
            ghosts and team agent """
        actions = gameState.getLegalActions(self.index)
        actions = filter(lambda x: x != 'Stop', actions)
        scores = [self.evaluate(gameState, action) for action in actions]
        bestScore = max(scores)
        best_index = [index for index in range(len(scores)) if scores[index] == bestScore]
        return actions[random.choice(best_index)]


class DummyAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at baselineTeam.py for more details about how to
    create an agent as this is the bare minimum.
    """

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """

        '''
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py.
        '''
        CaptureAgent.registerInitialState(self, gameState)

        '''
        Your initialization code goes here, if you need any.
        '''

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        actions = gameState.getLegalActions(self.index)

        '''
        You should change this in your own agent.
        '''

        return random.choice(actions)
