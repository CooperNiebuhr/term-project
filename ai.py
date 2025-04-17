import random
import chess

def select_random_move(board):
    legal_moves = list(board.legal_moves)
    if legal_moves:
        return random.choice(legal_moves)
    return None
