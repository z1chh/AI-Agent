from __future__ import annotations

import numpy as np
import math
import sys

from agents.monte_carlo.GameState import State # agents.monte_carlo.

class Node():
    '''
    Represents a node in the MCT.
    
    4 big steps:
        - Selection
        - Expansion (called from within selection)
        - Simulation
        - Backpropagation
    
    Some steps contain helper functions to split the code into smaller tasks.
    '''
    
    
    # Just to distinguish a Node from another (for toString method)
    NUM_NODES = 0 # The total number of Nodes created
    
    
    # Constructor
    def __init__(self, state: State, parent: Node = None, our_turn: bool = True):
        # for toString()
        Node.NUM_NODES += 1
        self._node_num = Node.NUM_NODES
        
        # Explicit params
        self._state: State = state
        self._parent: Node = parent
        self._our_turn: bool = our_turn
        
        # Add this child to the parent, if there is one
        if parent is not None:
            parent._children.append(self)
        
        # Initialize other attributes
        self._children: list[Node] = []
        self._unexplored_states: list[State] = self.unexplored_states()
        self._number_of_visits: int = 0
        self._wins: float = 0.0
    
    
    
    '''
    Selection - Apply tree policy
        Iterate through every child, and compute their UCT value.
        Then, recursively call this method on that child.
        If the current node contains unexplored children, return one of these children randomly.
        If the current node is a leaf, call expansion.
        
        To be called on the root.
        Returns the leaf node to expand.
    '''
    def selection(self, scaling_constant = math.sqrt(2)) -> Node:
        node = self
        while len(node._children) > 0 or len(node._unexplored_states) > 0:
            # Return a random unexplored state if there are any
            if len(node._unexplored_states) > 0:
                return node.expansion()
            
            # Otherwise, apply UCT
            else:
                values = []
                # Compute value for every child
                for child in node._children:
                    exploitation = float(child._wins) / child._number_of_visits
                    exploration = scaling_constant * math.sqrt(math.log(node._number_of_visits) / child._number_of_visits)
                    values.append(exploitation + exploration)
                    
                # Pick best value
                index = 0
                best = values[0]
                for i in range(len(values)):
                    value = values[i]
                    if value > best:
                        best = value
                        index = i
                
                # Update current node
                node = node._children[index]
        
        # Return the leaf node
        return node
    
    
    
    '''
    Expansion - Initialize every children
        Compute BFS to find every reachable cell.
        For each cell visited, check where can a wall be placed.
        Each cell + wall placement combination corresponds to a new State.
        Create a new child for every possible State.
        
        To be called on the leaf node to expand.
        Returns the same node (which is now expanded but unexplored).
    '''
    def expansion(self) -> Node:
        return Node(self._unexplored_states.pop(), self, not self._our_turn)
    
    
    
    '''
    Simulation - Apply default policy
        Choose one random new leaf.
        Simulate one run of the game randomly until it ends.
        Check the results of the game.
        
        To be called on the leaf node that got expanded.
        Returns the result of the simulation.
    '''
    def simulation(self) -> int:
        # Get current state
        current_state = self._state
        
        # While it is not an end-game state, simulate randomly
        turn = self._our_turn
        while not current_state.is_end_game_state():
            move = current_state.get_random_move(turn)
            current_state = current_state.apply_move(move, turn)
            turn = not turn
        
        # Return the score
        return current_state._game_score
    #    # Find random leaf to run on
    #    state = self.rollout_policy()
    #    
    #    # Run a simulation
    #    return self.rollout(state)
    
    
    # Returns one random new state from a list of unexplored states
    #def rollout_policy(self) -> State:
    #    # Check if the current leaf is an end-game state
    #    if len(self._unexplored_states) == 0:
    #        # If yes, return results directly
    #        return self._state
        
    #    return self._unexplored_states[np.random.randint(len(self._unexplored_states))]
    
    
    # Returns the random simulation's score and whether it was won or not.
    # +1 for a win, -1 for a loss, 0 for a tie.
    #def rollout(self, state: State) -> int:
    #    # Get current state
    #    current_state = state
    #    
    #    # While it is not an end-game state, simulate randomly
    #    turn = self._our_turn
    #    while not current_state.is_end_game_state():
    #        move = current_state.get_random_move(turn)
    #        current_state = current_state.apply_move(move, turn)
    #        turn = not turn
    #    
    #    # Return the score
    #    return current_state._game_score
    
    
    
    '''
    Backpropagation
        Apply the results of the random simulation to the leaf.
        Propagate the results back to the root.
        Set every Node on the path as visited (increment).
        
        To be called on the leaf that performed the simulation.
        Returns nothing.
    '''
    def backpropagate(self, score: int) -> None:
        # Increment number of visits
        self._number_of_visits += 1
        
        # Increment number of wins if result is Win (or 0.3 for draw)
        if score == 1:
            self._wins += 1.0
        elif score == 0:
            self._wins += 0.3
        
        # Update its parent, if there is one
        if self._parent is not None:
            self._parent.backpropagate(score)
    
    
    
    # Other functions:
    # # Gets ALL neighbouring states of the current game state.
    # To be used in the constructor ONCE only!
    def unexplored_states(self) -> list[State]:
        return self._state.get_neighbouring_states(self._our_turn)
    
    
    # Computes the Node's win ratio
    def win_ratio(self) -> float:
        return float(self._wins) / self._number_of_visits

    
    # Extract move
    def extract_move(self, node: Node) -> tuple(tuple(int, int), int):
        return self._state.extract_move(node._state)
    
    
    # toString()
    def __repr__(self) -> str:
        return "Node%d:\n\tChildren:\t%d\n\tVisits:\t\t%d\n\tWins:\t\t%d\n" %(self._node_num, len(self._children), self._number_of_visits, self._wins)



if __name__ == '__main__':
    n1 = Node(None, None, True)        #              n1
    n2 = Node(None, n1, False)         #             /  \
    n3 = Node(None, n1, False)         #            /    \
    n4 = Node(None, n2, True)          #           n2    n3
    n5 = Node(None, n2, True)          #          /  \
    print(n1)                          #         /    \
    print(n2)                          #        n4    n5
    print(n3)
    print(n4)
    print(n5)
    print("n1 has %d children." %(len(n1._children)))
    
    # Test to see if UCT functions work
    print("\n\n\nTest for UCT function")
    n6 = Node(None, None, True)        #              n6
    n7 = Node(None, n6, False)         #             /  \
    n8 = Node(None, n6, False)         #            /    \
    n6._number_of_visits = 10          #           n7    n8
    n6._wins = 6
    n7._number_of_visits = 7
    n7._wins = 5
    n8._number_of_visits = 2
    n8._wins = 0
    print(n6)
    print(n7)
    print(n8)
    n = n6.selection()
    print("UCT for n6: choose %s" %(n))
    print("Done")