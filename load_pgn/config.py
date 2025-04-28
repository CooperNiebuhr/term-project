

# load_pgn/config.py

import os

# Base directory for this package
HERE = os.path.dirname(__file__)

# Folder where users drop their .pgn files
PGN_FOLDER = "pgns"
# Stockfish binary (install via `brew install stockfish` or point to full path)
ENGINE_PATH = "stockfish"

# Search depth (you can bump this up if you have the time)
ENGINE_DEPTH = 15

# Where to write your CSV of blunders (one level up: term-project/outputs/)
OUTPUT_CSV = os.path.join(HERE, '..', 'outputs', 'blunder_results.csv')

# (Later) your pre-made puzzle database CSV
PUZZLE_DB_CSV = os.path.join(HERE, '..', 'puzzles', 'puzzles.csv')
