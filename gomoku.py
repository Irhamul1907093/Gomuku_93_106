from asyncio import PriorityQueue
import pygame
from Minimax import MinimaxNode
from checkers.constants import *
import random
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from copy import deepcopy
from math import inf as infinity


class Gomoku:
    def __init__(self):
        self.board = [[None for _ in range(COLS)] for _ in range(ROWS)]
        self.current_player = 'human'
        self.moves = []

    def draw_board(self, win):
        win.fill(WHITE)
        # vertical_offset = (HEIGHT - (ROWS * SQUARE_SIZE)) // 2
        for row in range(ROWS):
            for col in range(COLS):

                rect = pygame.Rect(BOARD_START_POS_X + col * CELL_SIZE, BOARD_START_POS_Y + row * CELL_SIZE, CELL_SIZE,
                                   CELL_SIZE)
                pygame.draw.rect(win, BLACK, rect, 1)

                # pygame.draw.rect(win, BLACK, (col * SQUARE_SIZE, row * SQUARE_SIZE + vertical_offset, SQUARE_SIZE, SQUARE_SIZE), 1)
                if self.board[row][col] == 'human':
                    self.draw_cross(win, row, col)
                    # print(f"{row,col}=>{self.board[row][col]}")
                elif self.board[row][col] == 'AI':
                    self.draw_circle(win, row, col)

                font = pygame.font.Font(None, 72)
                text_surface = font.render(f"Current Turn : {self.current_player}", True, BLACK)
                text_rect = text_surface.get_rect(center=(400, 20))

                win.blit(text_surface, text_rect)

                # win.blit()

    def draw_cross(self, win, row, col):
        x = BOARD_START_POS_X + col * CELL_SIZE
        y = BOARD_START_POS_Y + row * CELL_SIZE
        pygame.draw.line(win, GREEN, (x, y), (x + CELL_SIZE, y + CELL_SIZE), 2)
        pygame.draw.line(win, GREEN, (x + CELL_SIZE, y), (x, y + CELL_SIZE), 2)

    def draw_circle(self, win, row, col):
        x = (BOARD_START_POS_X + col * CELL_SIZE + (CELL_SIZE // 2))
        y = (BOARD_START_POS_Y + row * CELL_SIZE + (CELL_SIZE // 2))

        pygame.draw.circle(win, RED, (x, y), CELL_SIZE // 2 - 2, 2)

    def make_move(self, row, col):
        if self.board[row][col] is None:
            self.board[row][col] = self.current_player
            
            self.current_player = 'AI' if self.current_player == 'human' else 'human'
            return True
        return False

    def heuristic(self, board_state):
        # Implement a heuristic function that evaluates the board state
        # For example, count the potential winning lines for the AI
        return self.evaluate(board_state)[self.current_player]

    def generate_successors(self, board_state):
        # Generate all possible moves and return successor states
        possible_moves = self.generate_possible_moves(board_state, expansion_range=1)
        successors = []
        for move in possible_moves:
            new_board = self.make_move(board_state, move, self.current_player)
            successors.append((new_board, move))
        return successors

    def is_goal_state(self, board_state):
        # Check if the board state is a winning state
        return self.game_over(board_state)

    def a_star_search(self):
        start_state = self.board
        frontier = PriorityQueue()
        frontier.put((0, start_state, None))
        came_from = {}
        cost_so_far = {}
        came_from[start_state] = None
        cost_so_far[start_state] = 0

        while not frontier.empty():
            _, current_state, move = frontier.get()

            if self.is_goal_state(current_state):
                return move

            for next_state, next_move in self.generate_successors(current_state):
                new_cost = cost_so_far[current_state] + 1  # Assuming uniform cost for simplicity
                if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                    cost_so_far[next_state] = new_cost
                    priority = new_cost + self.heuristic(next_state)
                    frontier.put((priority, next_state, next_move))
                    came_from[next_state] = current_state

        # Fallback if no goal state is found
        return None

    def ai_move(self):
        """
        First Checking if the game has just started
        i.e. there is not enough move so that AI can choose a move
        intelligently.
        """
        if len(self.moves) <= 3:
            row, col = self.random_move(1)
            return row, col

        """
        If there is enough moves,(>3)
        then check if there is any checkmate
        moves that AI can take
        """
        chkmate = self.checkmate()

        if chkmate:
            print("AI checkmate move")
            return chkmate

        """
        If there is no checkmate moves as well, 
        then check if there is any high impact move
        """
        print("Checking for high-impact move...")
        opponent = 'AI' if self.current_player == 'human' else 'human'
        if ENABLE_HIGH_IMPACT_MOVE:
            opponent_high_impact_move, opponent_high_impact_score = self.high_impact_move(opponent)
            com_high_impact_move, com_high_impact_score = self.high_impact_move(self.current_player)

            # Fuzzy Logic Implementation
            # Define fuzzy variables
            threat_level = ctrl.Antecedent(np.arange(0, 11, 1), 'threat_level')
            opportunity_level = ctrl.Antecedent(np.arange(0, 11, 1), 'opportunity_level')
            move_priority = ctrl.Consequent(np.arange(0, 11, 1), 'move_priority')

            # Define membership functions
            threat_level['low'] = fuzz.trimf(threat_level.universe, [0, 0, 5])
            threat_level['medium'] = fuzz.trimf(threat_level.universe, [0, 5, 10])
            threat_level['high'] = fuzz.trimf(threat_level.universe, [5, 10, 10])

            opportunity_level['low'] = fuzz.trimf(opportunity_level.universe, [0, 0, 5])
            opportunity_level['medium'] = fuzz.trimf(opportunity_level.universe, [0, 5, 10])
            opportunity_level['high'] = fuzz.trimf(opportunity_level.universe, [5, 10, 10])

            move_priority['low'] = fuzz.trimf(move_priority.universe, [0, 0, 5])
            move_priority['medium'] = fuzz.trimf(move_priority.universe, [0, 5, 10])
            move_priority['high'] = fuzz.trimf(move_priority.universe, [5, 10, 10])

            # Define fuzzy rules
            rule1 = ctrl.Rule(threat_level['high'] | opportunity_level['high'], move_priority['high'])
            rule2 = ctrl.Rule(threat_level['medium'] | opportunity_level['medium'], move_priority['medium'])
            rule3 = ctrl.Rule(threat_level['low'] & opportunity_level['low'], move_priority['low'])

            # Create control system
            move_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
            move_sim = ctrl.ControlSystemSimulation(move_ctrl)

            # Evaluate high-impact moves using fuzzy logic
            move_sim.input['threat_level'] = opponent_high_impact_score
            move_sim.input['opportunity_level'] = com_high_impact_score
            move_sim.compute()
            high_impact_priority = move_sim.output['move_priority']

            if high_impact_priority > 5:
                if opponent_high_impact_move and opponent_high_impact_score > com_high_impact_score:
                    print("AI may lose as HUMAN has a high-impact move.")
                    print("AI has taken this move (a defensive move).")
                    return opponent_high_impact_move
                if com_high_impact_move and com_high_impact_score >= opponent_high_impact_score:
                    print("AI may win it has a high-impact move.")
                    print("AI has taken this move (an offensive move).")
                    return com_high_impact_move

            print("No high impact moves")

        """
        If there is no high impact move as well, 
        then check if there is any combo move
        """
        print("Checking for combo moves...")

        opponent_combo_move = self.combo_move(self.board, opponent)
        com_combo_move = self.combo_move(self.board, self.current_player)

        if com_combo_move:
            print("AI has a combo move. Take it!")
            return com_combo_move

        if opponent_combo_move:
            print("HUMAN has a combo move. Block it!")
            return opponent_combo_move

        """
        If there is no combo move as well, 
        AI will use alpha_beta_pruning algorithm to 
        find the best move
        """
        print("There is no combo move.")
        print("---------------------------------")
        print("Will use the Alpha-Beta pruning algorithm. Calculating...")

        root_node = MinimaxNode(self.board, self.moves[-1::1], self.current_player, None)
        MAX_TREE_DEPTH_LEVEL = 2  # The maximum depth the AI will search
        score = self.alpha_beta(root_node, MAX_TREE_DEPTH_LEVEL, -infinity, +infinity, True)
        print("Completed calculation with depth =", MAX_TREE_DEPTH_LEVEL, ".")

        if root_node.planing_next_move is None:
            print("No valid move found with Alpha-Beta pruning.")
            print("Switching to A* search algorithm.")
            best_move = self.a_star_search()
            if best_move:
                print("A* search found a move:", best_move)
                return best_move
            else:
                print("A* search did not find a valid move. Using random move.")
                return self.random_move(1)

        return root_node.planing_next_move

    def combo_move(self, board, current_turn):

        # combo move
        # is a combo which could create
        # a one-end-blocked-four and a unblocked three 
        # or n blocked-four (n>=2)

        # get moves that could create
        # one-end-blocked-four

        blocked_four_patterns = []
        blocked_four_pattern_length = 5
        matched_blocked_four_pattern_move_direction_list = []
        move_direction_dictionary = dict()

        # add element(s) to blocked_four_patterns
        if current_turn == 'AI':
            for pattern in AI_5_PATTERNS:
                if pattern.count('AI') == 4:
                    blocked_four_patterns.append(pattern)
        elif current_turn == 'human':
            for pattern in human_5_PATTERNS:
                if pattern.count('human') == 4:
                    blocked_four_patterns.append(pattern)

        # scan for blocked-four

        possible_moves = self.generate_possible_moves(board, 2)

        for p_m_move in possible_moves:
            move_direction_set = set()
            matched_direction_count = 0

            direction_pattern_tuples = self.get_direction_pattern_tuples(board, p_m_move, 4, current_turn)

            if len(direction_pattern_tuples) > 0:
                for tuple in direction_pattern_tuples:
                    direction, pattern = tuple
                    for i in range(0, len(pattern) - blocked_four_pattern_length + 1):
                        checking_pattern = [
                            pattern[i],
                            pattern[i + 1],
                            pattern[i + 2],
                            pattern[i + 3],
                            pattern[i + 4],
                        ]
                        has_pattern_in_this_direction = False
                        for blocked_four_pattern in blocked_four_patterns:
                            if checking_pattern == blocked_four_pattern:
                                has_pattern_in_this_direction = True
                                move_direction_dictionary[p_m_move] = (direction, checking_pattern)

                        if has_pattern_in_this_direction:
                            matched_blocked_four_pattern_move_direction_list.append((direction, p_m_move))
                            if (direction, p_m_move) not in move_direction_set:
                                move_direction_set.add((direction, p_m_move))
                                matched_direction_count += 1
                                # this means that move can create at least 2 blocked fours -> a combo move
                                if matched_direction_count > 1:
                                    return p_m_move

                                    # for each move that could create one-end-blocked-four,
        # we scan if there is any unblocked-three created by that move
        if len(matched_blocked_four_pattern_move_direction_list) >= 1:

            move_pos_in_pattern = 4
            # scan for unblocked-three
            for p_m_move in matched_blocked_four_pattern_move_direction_list:
                blocked_four_direction, blocked_four_move = p_m_move
                direction_pattern_tuples = self.get_direction_pattern_tuples(board, blocked_four_move,
                                                                             move_pos_in_pattern, current_turn)

                if len(direction_pattern_tuples) > 0:
                    for tuple in direction_pattern_tuples:
                        direction, pattern = tuple  # len(pattern) = 7
                        # make sure that current_turn is counted in pattern
                        if pattern[move_pos_in_pattern] == current_turn:  # center pattern must be the current move
                            M = current_turn
                            E = None
                            opponent = 'AI' if self.current_player == 'human' else 'human'
                            check_left_pattern = pattern[1:5].count(current_turn) >= 3 and pattern[0:5].count(
                                opponent) == 0
                            check_right_pattern = pattern[4:].count(current_turn) >= 3 and pattern[4:].count(
                                opponent) == 0
                            check_center_pattern = (
                                    pattern[2:7] == [E, M, M, M, E]
                                    or pattern[1:7] == [E, M, E, M, M, E]
                                    or pattern[2:-1] == [E, M, M, E, M, E]
                            )

                            has_unblocked_three = check_left_pattern or check_right_pattern or check_center_pattern
                            if has_unblocked_three and direction != blocked_four_direction:
                                return blocked_four_move

        return None

    def get_direction_pattern_tuples(self, board, move, streak, current_turn):
        """
            It takes a board, a move, 
            a streak, and the current turn, 
            and returns a list of lists of the
            pieces in the directions of the move
        
            :param board: the current board (will not be changed after running this function)
            :param move: the move that is being evaluated
            :param streak: the number of pieces in a row needed to win
            :param current_turn: the current player's turn
            :return: A list of lists of patterns.
        """

        if not self.is_valid_move(move, board):
            return []
        # streak = number of unblocked pieces
        move_r, move_c = move
        # r ~ x
        # c ~ y
        direction_patterns = []
        # horizontal
        pattern = []
        for i in range(-streak, streak + 1):
            if i == 0:
                temp_move = move
                pattern.append(current_turn)
            else:
                temp_move = (move_r + i, move_c)
                if self.is_valid_move(temp_move, board):
                    temp_move_r, temp_move_c = temp_move
                    pattern.append(board[temp_move_r][temp_move_c])
        if len(pattern) > streak + 2:
            direction_patterns.append(('H', pattern))

        # vertical
        pattern = []
        for i in range(-streak, streak + 1):
            if i == 0:
                temp_move = move
                pattern.append(current_turn)
            else:
                temp_move = (move_r, move_c + i)
                if self.is_valid_move(temp_move, board):
                    temp_move_r, temp_move_c = temp_move
                    pattern.append(board[temp_move_r][temp_move_c])
        if len(pattern) > streak + 2:
            direction_patterns.append(('V', pattern))

        # diagonals
        # lower-left to upper-right
        pattern = []
        for i in range(-streak, streak + 1):
            if i == 0:
                temp_move = move
                pattern.append(current_turn)
            else:
                temp_move = (move_r + i, move_c + i)
                if self.is_valid_move(temp_move, board):
                    temp_move_r, temp_move_c = temp_move
                    pattern.append(board[temp_move_r][temp_move_c])
        if len(pattern) > streak + 2:
            direction_patterns.append(('D1', pattern))
        # upper-left to lower-right
        pattern = []
        for i in range(-streak, streak + 1):
            if i == 0:
                temp_move = move
                pattern.append(current_turn)
            else:
                temp_move = (move_r - i, move_c + i)
                if self.is_valid_move(temp_move, board):
                    temp_move_r, temp_move_c = temp_move
                    pattern.append(board[temp_move_r][temp_move_c])
        if len(pattern) > streak + 2:
            direction_patterns.append(('D2', pattern))

        return direction_patterns