# Name    : HU, Zi Chen
# GitHub  : https://github.com/z1chh

# My AI Agent
from __future__ import annotations

from agents.agent import Agent
from store import register_agent
import sys

from copy import deepcopy
import numpy as np
import math

from agents.monte_carlo.MCTNode import Node # agents.
from agents.monte_carlo.MCTree import Tree # agents.
from agents.monte_carlo.GameState import State # agents.



@register_agent("ai_agent")
class AIAgent(Agent):
    """
    AI agent.
    Uses MCTS to compute best move.
    """

    
    def __init__(self):
        super(AIAgent, self).__init__()
        self.name = "AI Agent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }


    # Compute the next step
    def step(self, chess_board, my_pos, adv_pos, max_step):
        # Create MCT
        state = State(chess_board, my_pos, adv_pos, max_step)
        node = Node(state, None, True)
        mct = Tree(node)
        
        return mct.execute(1.95, math.sqrt(2))
    
        # Dead code
        #same_time = True
        
        # MCTS with time = 1.95 everytime.
        #if(same_time):
        #    return mct.execute(1.95, math.sqrt(2))
        
        # MCTS with time constraint reduced if board is bigger.
        #size = len(chess_board)
        # Perform search and return results
        #if size < 7:
        #    return mct.execute(1.94, math.sqrt(2))
        #elif size == 7:
        #    return mct.execute(1.88, math.sqrt(2))
        #elif size == 8:
        #    return mct.execute(1.82, math.sqrt(2))
        #elif size == 9:
        #    return mct.execute(1.74, math.sqrt(2))
        #elif size == 10:
        #    return mct.execute(1.68, math.sqrt(2))
        #elif size == 11:
        #    return mct.execute(1.66, math.sqrt(2))
        #else:
        #    return mct.execute(1.64, math.sqrt(2))
    