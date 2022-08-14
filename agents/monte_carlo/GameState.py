from __future__ import annotations

import numpy as np
from copy import deepcopy
import random

class State:
    '''
    Represents the state of a MCT node.
    
    Methods:
        - __init__(self, chess_board, my_pos, adv_pos, max_step):
              Constructor to store the current state of a Colosseum game
        
        - is_valid_move(self, move: tuple(tuple(int, int), int), our_turn: bool) -> bool:
              Checks if the move can be applied to the current state
        
        - check_row(self, my_pos: tuple(int, int), steps: int, our_turn: bool) -> list[State]:
              Computes a list of reachable state in the current col
        
        - def get_neighbouring_states(self, our_turn: bool) -> list[State]:
              Computes reachable states, given the current state and max_step
        
        - is_end_game_state(self) -> bool:
              Checks whether the current state represents a finished game, and updates the score if yes
        
        - get_random_move(self, our_turn: bool) -> tuple(tuple(int, int), int):
              Computes one random move from the current state
        
        - apply_move(self, move: tuple(tuple(int, int), int), our_turn: bool) -> State:
              Apply the given move to the current state, and returns the resulting state
        
        - extract_move(self, state: State) -> tuple(tuple(int, int), int):
              Computes the move to make to reach the second state from the current one
    '''


    # Global variable (class attribute)
    MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))
    
    
    # Constructor
    def __init__(self, chess_board, my_pos, adv_pos, max_step):
        self._chess_board = chess_board
        self._board_size: int = len(self._chess_board)
        self._my_pos: tuple(int, int) = my_pos
        self._adv_pos: tuple(int, int) = adv_pos
        self._max_step: int = max_step
        self._game_score: int = -1
    
    
    # Check if the given move is valid
    def is_valid_move(self, move: tuple(tuple(int, int), int), our_turn: bool) -> bool:
        # Initialize variables
        start_pos = deepcopy(self._my_pos)
        if not our_turn:
            start_pos = deepcopy(self._adv_pos)
        start_pos = np.asarray(start_pos)
        end_pos, barrier_dir = move
        end_pos = np.asarray(end_pos)
        
        ori_my_pos = deepcopy(self._my_pos)
        ori_my_pos = np.asarray(ori_my_pos)
        ori_adv_pos = deepcopy(self._adv_pos)
        ori_adv_pos = np.asarray(ori_adv_pos)
        
        # Endpoint already has barrier or is boarder
        r, c = end_pos
        if self._chess_board[r, c, barrier_dir]:
            return False
        if np.array_equal(start_pos, end_pos):
            return True

        # Get position of the adversary
        adv_pos = ori_adv_pos if our_turn else ori_my_pos

        # BFS
        state_queue = [(start_pos, 0)]
        visited = {tuple(start_pos)}
        is_reached = False
        while state_queue and not is_reached:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            if cur_step == self._max_step:
                break
            for dir, move in enumerate(State.MOVES):
                if self._chess_board[r, c, dir]:
                    continue

                next_pos = cur_pos + move
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                if np.array_equal(next_pos, end_pos):
                    is_reached = True
                    break
                
                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))

        return is_reached
    
    
    # Returns all reachable states by staying in the current row
    def check_col(self, my_pos: tuple(int, int), steps: int, our_turn: bool) -> list[State]:
        cur_pos = deepcopy(my_pos)
        x, y = cur_pos
        neighbours = []
        
        # Check current pos
        for i in range(4):
            move = ((x, y), i)
            if self.is_valid_move(move, our_turn):
                    neighbours.append(self.apply_move(move, our_turn))
        
        # Check up
        for _ in range(steps):
            x -= 1
            if x < 0:
                break
            for i in range(4):
                move = ((x, y), i)
                if self.is_valid_move(move, our_turn):
                    neighbours.append(self.apply_move(move, our_turn))
        
        # Check down
        x, y = cur_pos
        for _ in range(steps):
            x += 1
            if x >= self._board_size:
                break
            for i in range(4):
                move = ((x, y), i)
                if self.is_valid_move(move, our_turn):
                    neighbours.append(self.apply_move(move, our_turn))
        
        # Return list of reachable States
        return neighbours
    
    
    # Returns all reachable states
    def get_neighbouring_states(self, our_turn: bool) -> list[State]:
        if self.is_end_game_state():
            return []
        
        neighbours = []
        ori_pos = deepcopy(self._my_pos)
        
        # Check current row
        states = self.check_col(self._my_pos, self._max_step, our_turn)
        for state in states:
            neighbours.append(state)
        
        # Check cols on the left
        x, y = ori_pos
        for i in range(1, self._max_step + 1):
            y += 1
            if y >= self._board_size:
                break
            states = self.check_col((x, y), self._max_step - i, our_turn)
            for state in states:
                neighbours.append(state)
        
        # Check cols on the right
        x, y = ori_pos
        for i in range(1, self._max_step + 1):
            y -= 1
            if y < 0:
                break
            states = self.check_col((x, y), self._max_step - i, our_turn)
            for state in states:
                neighbours.append(state)
        
        random.shuffle(neighbours)
        return neighbours
    
    
    # Check if this state is an Game End state, and sets the score if it is
    def is_end_game_state(self) -> bool:
        # Union-Find
        father = dict()
        for r in range(self._board_size):
            for c in range(self._board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(self._board_size):
            for c in range(self._board_size):
                for dir, move in enumerate(
                    State.MOVES[1:3]
                ):  # Only check down and right
                    if self._chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(self._board_size):
            for c in range(self._board_size):
                find((r, c))
        p0_r = find(self._my_pos)
        p1_r = find(self._adv_pos)
        if p0_r == p1_r:
            return False
        else:
            p0_score = list(father.values()).count(p0_r)
            p1_score = list(father.values()).count(p1_r)
            if p0_score > p1_score:
                self._game_score = 1
            elif p0_score == p1_score:
                self._game_score = 0
            else:
                self._game_score = -1
            return True

    
    # Computes a random move given the current state
    def get_random_move(self, our_turn: bool) -> tuple(tuple(int, int), int):
        # Initialize variables
        if our_turn:
            my_pos = self._my_pos
            adv_pos = self._adv_pos
            ori_pos = deepcopy(my_pos)
        else:
            my_pos = self._adv_pos
            adv_pos = self._my_pos
            ori_pos = deepcopy(my_pos)
        steps = np.random.randint(0, self._max_step + 1)

        # Random Walk
        for _ in range(steps):
            r, c = my_pos
            dir = np.random.randint(0, 4)
            m_r, m_c = State.MOVES[dir]
            my_pos = (r + m_r, c + m_c)

            # Special Case enclosed by Adversary
            k = 0
            while self._chess_board[r, c, dir] or my_pos == adv_pos:
                k += 1
                if k > 300:
                    break
                dir = np.random.randint(0, 4)
                m_r, m_c = State.MOVES[dir]
                my_pos = (r + m_r, c + m_c)

            if k > 300:
                my_pos = ori_pos
                break

        # Put Barrier
        dir = np.random.randint(0, 4)
        r, c = my_pos
        while self._chess_board[r, c, dir]:
            dir = np.random.randint(0, 4)

        return my_pos, dir
    
    
    # Applies the move to the current state, and returns a new state.
    # Note that we assume the move is valid.
    def apply_move(self, move: tuple(tuple(int, int), int), our_turn: bool) -> State:
        new_pos, dir = move
        adv_pos = deepcopy(self._adv_pos)
        if not our_turn:
            adv_pos = deepcopy(self._my_pos)
        
        # Update the chess_board with the new wall.
        new_board = deepcopy(self._chess_board)
        x, y = new_pos
        new_board[x][y][dir] = True
        
        # Update opposite wall
        if dir == 0 and x != 0:
            new_board[x - 1][y][2] = True
        if dir == 1 and y != self._board_size - 1:
            new_board[x][y + 1][3] = True
        if dir == 2 and x != self._board_size - 1:
            new_board[x + 1][y][0] = True
        if dir == 3 and y != 0:
            new_board[x][y - 1][1] = True
        
        # Return the new state
        if not our_turn:
            return State(new_board, adv_pos, new_pos, self._max_step)
        return State(new_board, new_pos, adv_pos, self._max_step)

    
    # Compute the move that was played to reach the input state
    def extract_move(self, state: State) -> tuple(tuple(int, int), int):
        # Get the updated position
        my_pos = state._my_pos
        x, y = my_pos
        
        # Find the wall that was placed
        ori_board = self._chess_board
        new_board = state._chess_board
        dir = 0
        
        for i in range(4):
            if ori_board[x][y][i] != new_board[x][y][i]:
                dir = i
                break
        
        # Return the move that was made
        return (my_pos, dir)

##################################################
#                                                #
#    Number of possible outcomes for 2 steps:    #
#                                                #
#        +-----+-----+-----+-----+-----+         #
#        |     |     |  x2 |     |     |         #
#        +-----+-----+-----+-----+-----+         #
#        |     |  x2 |  x1 |  x2 |     |         #
#        +-----+-----+-----+-----+-----+         #
#        |  x2 |  x1 |x0/x2|  x1 |  x2 |         #
#        +-----+-----+-----+-----+-----+         #
#        |     |  x2 |  x1 |  x2 |     |         #
#        +-----+-----+-----+-----+-----+         #
#        |     |     |  x2 |     |     |         #
#        +-----+-----+-----+-----+-----+         #
#                                                #
#   13 positions x 4 walls/position = 52 sols    #
#                                                #
#                                                #
#   Other steps (maximum):                       #
#   (not taking into account exising walls)      #
#   (nor opponent)                               #
#                                                #
#   3 steps:  25 positions - 100 possible sols   #
#   4 steps:  42 positions - 168 possible sols   #
#   5 steps:  62 positions - 248 possible sols   #
#   6 steps:  86 positions - 344 possible sols   #
#                                                #
##################################################