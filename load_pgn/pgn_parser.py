import os
import chess.pgn

def parse_pgns(folder_path):
    """
    Iterate through all PGN files in a folder and yield (FEN, played_move_uci).
    """
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith('.pgn'):
            continue
        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'r') as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                board = game.board()
                for move in game.mainline_moves():
                    fen = board.fen()
                    move_uci = move.uci()
                    board.push(move)
                    yield fen, move_uci
