from __future__ import annotations

import time
import math

from agents.monte_carlo.MCTNode import Node # agents.monte_carlo.
from agents.monte_carlo.GameState import State # agents.monte_carlo.

class Tree():
    '''
    Represents a MCT.
    
    Runs simulations for time_to_run (default value 1.95sec).
    UCT uses scaling_constant (default value math.sqrt(2)).
    '''
    
    
    # Constructor
    def __init__(self, root: Node):
        # Initialize the root as the current game state
        self._root: Node = root
    
    
    # Computes best move
    def execute(self, time_to_run: float = 1.95, scaling_constant: float = math.sqrt(2)) -> tuple(tuple(int, int), int):
        # Run MCTS for n sec
        t_end = time.time() + time_to_run
        while time.time() < t_end:
            # Selection phase (expansion phase also computed)
            node = self._root.selection(scaling_constant)
        
            # Simulation phase
            result = node.simulation()
        
            # Backpropagation phase
            node.backpropagate(result)
        # Return best move to make
        best_node = self.get_best_child()
        return self.extract_move(best_node)
    
    
    # Computes best child of the root
    def get_best_child(self) -> Node:
        node = self._root._children[0]
        ratio = node.win_ratio()
        for n in self._root._children:
            r = n.win_ratio()
            if r > ratio:
                node = n
                ratio = r
        return node
    
    
    # Extract move
    def extract_move(self, node: Node) -> tuple(tuple(int, int), int):
        return self._root.extract_move(node)
    
