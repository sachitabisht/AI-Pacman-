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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        newGhost = successorGameState.getGhostPositions() #useful info for ghost locations
        newScore = successorGameState.getScore() #useful info for current scores
        posOfFood = float(9999999) #set high value for comparison
        for i in newFood.asList(): #iterate through food to find best location of food for pacman
            posOfFood = min(posOfFood, manhattanDistance(newPos, i))
        for i in newGhost: #iterate through ghost locations to see if any are too close to Pacman
            if (manhattanDistance(newPos, i) <= 1):
                return -1
        return newScore + 10.0/posOfFood

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        return self.maxValue(gameState)[1]
    def maxValue(self, gameState, depth=0, agentIndex=0):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), Directions.STOP
        lowerBound = float(-99999999)
        bestAction = Directions.STOP
        for a in gameState.getLegalActions(agentIndex):
            v = self.minValue(gameState.generateSuccessor(agentIndex, a), depth, agentIndex + 1)[0]
            if v > lowerBound:
                lowerBound = v
                bestAction = a
        return lowerBound, bestAction
    def minValue(self, gameState, depth=0, agentIndex=1):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), Directions.STOP
        upperBound = float(99999999)
        bestAction = Directions.STOP
        for a in gameState.getLegalActions(agentIndex):
            if agentIndex == gameState.getNumAgents() - 1:
                v = self.maxValue(gameState.generateSuccessor(agentIndex, a), depth + 1, 0)[0]
            else:
                v = self.minValue(gameState.generateSuccessor(agentIndex, a), depth, agentIndex + 1)[0]
            if v < upperBound:
                upperBound = v
                bestAction = a
        return upperBound, bestAction
    #util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.maxValue(gameState)[1]
    def maxValue(self, gameState, depth=0, agentIndex=0, alpha=float(-99999999), beta=float(99999999)):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), Directions.STOP
        lowerBound = float(-99999999)
        bestAction = Directions.STOP
        for a in gameState.getLegalActions(agentIndex):
            v = self.minValue(gameState.generateSuccessor(agentIndex, a), depth, agentIndex + 1, alpha, beta)[0]
            if v > lowerBound:
                lowerBound = v
                bestAction = a
            if v > beta:
                return v, a
            alpha = max(alpha, lowerBound)
        return lowerBound, bestAction
    def minValue(self, gameState, depth=0, agentIndex=1, alpha=float(-99999999), beta=float(99999999)):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), Directions.STOP
        upperBound = float(99999999)
        bestAction = Directions.STOP
        for a in gameState.getLegalActions(agentIndex):
            if agentIndex == gameState.getNumAgents() - 1:
                v = self.maxValue(gameState.generateSuccessor(agentIndex, a), depth + 1, 0, alpha, beta)[0]
            else:
                v = self.minValue(gameState.generateSuccessor(agentIndex, a), depth, agentIndex + 1, alpha, beta)[0]
            if v < upperBound:
                upperBound = v
                bestAction = a
            if v < alpha:
                return v, a
            beta = min(beta, upperBound)
        return upperBound, bestAction
    #util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.maxValue(gameState)[1]
    def maxValue(self, gameState, depth=0, agentIndex=0):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), Directions.STOP
        lowerBound = float(-99999999)
        bestAction = Directions.STOP
        for a in gameState.getLegalActions(agentIndex):
            v = self.minValue(gameState.generateSuccessor(agentIndex, a), depth, agentIndex + 1)
            if v > lowerBound:
                lowerBound = v
                bestAction = a
        return lowerBound, bestAction
    def minValue(self, gameState, depth=0, agentIndex=1):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        upperBound = float(99999999)
        bestAction = Directions.STOP
        expected = 0
        for a in gameState.getLegalActions(agentIndex):
            if agentIndex == gameState.getNumAgents() - 1:
                v = self.maxValue(gameState.generateSuccessor(agentIndex, a), depth + 1, 0)[0]
            else:
                v = self.minValue(gameState.generateSuccessor(agentIndex, a), depth, agentIndex + 1)
            expected = expected + v
        return expected / len(gameState.getLegalActions(agentIndex))
    #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    newScore = currentGameState.getScore() #useful info for current scores
    newCapsules = currentGameState.getCapsules()
    posOfFood = float(9999999) #set high value for comparison
    highScore = 0
    if currentGameState.isWin(): #winning state points
        return float(99999999)
    elif currentGameState.isLose(): #losing state points
        return float(-99999999)
    for food in newFood.asList(): #iterate through food to find best pos
        posOfFood = min(posOfFood, manhattanDistance(newPos, food))
    for ghost in newGhostStates: #iterate through ghost states to find scared ghosts
        if ghost.scaredTimer:
            highScore += 100
    highScore += random.random() * 10 #randomizing
    return newScore + 10.0/posOfFood + highScore

    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
