from copy import deepcopy
from checkers.constants import *
# from gomoku import Gomoku

class MinimaxNode:
    def __init__(self, board, last_move, current_turn, planing_next_move) -> None:
        self.board = deepcopy(board)
        self.last_move = last_move
        self.planing_next_move = planing_next_move
        self.current_turn = current_turn

    def generate_child_nodes(self):
        """
        It takes a board state, and returns 
        a list of all possible moves that 
        can be made from that board state
        :return: A list of MinimaxNode objects.
        """
        possible_moves = self.generate_possible_moves(self.board, 2)
        child_nodes = []
        opponent = 'human' if self.current_turn == 'AI' else 'AI'
        for move in possible_moves:
            child_node = MinimaxNode(self.board, move, opponent, None)
            child_node.board[move[0]][move[1]] = self.current_turn

            child_nodes.append(child_node)
        
        return child_nodes
    