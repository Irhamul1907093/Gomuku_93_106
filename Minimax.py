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
    
    def generate_possible_moves(self,board,expansion_range):
        """
        returns 
        all the possible moves that are
        not empty and have a neighbor
        """
        
        possible_moves = []
        for r in range(0, ROWS):
            for c in range(0, COLS):
                temp_move = board[r][c]
                if(temp_move != None): 
                    #if the move we are intentended to do is already occupied , 
                    #we will skip that square of the board
                    # (i.e it is not possible to move to that square(invalid) as it is already occupied)
                    continue
                
                neighbor = None
                # valid = 0
                
                #after checking if the square is occupied or not
                #now we will check if there is any occupied neighbour of that square
                #if there is occupied neighbour we will append the square to possible moves.
                
                #This logic implies that the AI will never choose a desserted square from the board. 
                # desserted square = the square which has no neighbour occupied square
                
                
                for i in range(-expansion_range,expansion_range+1):
                    for j in range(-expansion_range,expansion_range+1):
                        neighbor_r = i+r
                        neighbor_c = j+c
                        
                        if(0 <= neighbor_r < ROWS and 0 <= neighbor_c < COLS):
                            
                            neighbor = board[neighbor_r][neighbor_c]
                            
                        if(neighbor != None):
                            break
                
                
                if neighbor == None:
                    continue
                
                possible_moves.append((r, c))
        
        return possible_moves
    