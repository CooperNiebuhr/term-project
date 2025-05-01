""" 
****** README ******

This script reads a PGN string from standard input, parses it, and prints the FEN strings and UCI moves for each move in the game.
You must have a reuslt tag at the end of the PGN string (e.g., "1-0" or "0-1", or *).
"""




import sys, io, chess.pgn

def load_moves_from_pgn_string(pgn_text):
    pgn_io = io.StringIO(pgn_text)
    game = chess.pgn.read_game(pgn_io)
    if game is None:
        raise ValueError("No game found in PGN input—check your formatting.")
    board = game.board()
    moves = []
    for move in game.mainline_moves():
        fen_before = board.fen()
        moves.append((fen_before, move.uci()))
        board.push(move)
    return moves

def main():
    

    rating_input = input("Enter your current rating (e.g. 1500): ").strip()
    try:
        player_rating = int(rating_input)
    except ValueError:
        print(f"Warning: '{rating_input}' is not a valid integer. Defaulting rating to 0.")
        player_rating = 1000
    print(f"Player rating recorded as: {player_rating}\n")

    print("Paste your PGN, then on a new line type EOF and press Enter:")
    lines = []
    while True:
        line = input()
        if line.strip() == 'EOF':
            break
        lines.append(line)
    pgn_text = '\n'.join(lines)

    # parse & print
    try:
        moves = load_moves_from_pgn_string(pgn_text)
    except ValueError as e:
        print("Error:", e)
        return

    print(f"\nParsed {len(moves)} moves!\n")


if __name__ == "__main__":
    main()
