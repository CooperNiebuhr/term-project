import torch
from torch.utils.data import Dataset
import pandas as pd
import chess

def fen_to_tensor(fen: str) -> torch.FloatTensor:
    """
    Turn a FEN into a (12×8×8) binary tensor.
    Channels: [P,N,B,R,Q,K, p,n,b,r,q,k].
    """
    board = chess.Board(fen)
    tensor = torch.zeros(12, 8, 8, dtype=torch.float32)
    mapping = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    for sq, piece in board.piece_map().items():
        idx = mapping[piece.symbol()]
        row = 7 - (sq // 8)
        col = sq % 8
        tensor[idx, row, col] = 1.0
    return tensor

class TacticDataset(Dataset):
    """
    Dataset for tactics puzzles. Expects a CSV with columns:
      - FEN            (position before the motif move)
      - simple_label   (the motif tag, e.g. 'fork', 'pin', etc.)
    """
    def __init__(self, csv_path: str, label_map: dict):
        # 1) Load and normalize column names to lowercase
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.lower()

        # 2) Rename the label column to 'motif' for consistency
        df = df.rename(columns={'simple_label': 'motif'})

        # 3) Extract the lists of FENs and integer labels
        self.fens   = df['fen'].tolist()
        self.labels = [label_map[m] for m in df['motif']]

    def __len__(self):
        return len(self.fens)

    def __getitem__(self, idx):
        # Returns (board_tensor, label_int)
        fen   = self.fens[idx]
        label = self.labels[idx]
        return fen_to_tensor(fen), label
