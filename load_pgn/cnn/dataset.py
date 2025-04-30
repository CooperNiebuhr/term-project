# load_pgn/cnn/dataset.py

import torch
from torch.utils.data import Dataset
import pandas as pd
import chess
import random

def fen_to_tensor(fen: str, last_move_uci: str = None) -> torch.FloatTensor:
    """
    22-channel tensor:
      - 12 piece planes
      - 1 side-to-move
      - 2 last-move src/dst masks
      - 4 castling rights
      - 1 en passant possibility
      - 1 move generation features (attack/defense count)
      - 1 check indicator
    """
    board = chess.Board(fen)
    tensor = torch.zeros(22, 8, 8, dtype=torch.float32)

    # 12 piece planes
    mapping = {
        'P':0,'N':1,'B':2,'R':3,'Q':4,'K':5,
        'p':6,'n':7,'b':8,'r':9,'q':10,'k':11
    }
    for sq, piece in board.piece_map().items():
        idx = mapping[piece.symbol()]
        row = 7 - (sq // 8)
        col = sq % 8
        tensor[idx, row, col] = 1.0

    # side-to-move plane
    tensor[12, :, :] = float(board.turn)

    # last-move masks
    if last_move_uci:
        uci = last_move_uci[:4]
        src_sq, dst_sq = uci[:2], uci[2:4]
        src_idx = chess.parse_square(src_sq)
        dst_idx = chess.parse_square(dst_sq)
        src_row, src_col = 7 - (src_idx // 8), src_idx % 8
        dst_row, dst_col = 7 - (dst_idx // 8), dst_idx % 8
        tensor[13, src_row, src_col] = 1.0
        tensor[14, dst_row, dst_col] = 1.0
    
    # Add castling rights (4 planes)
    tensor[15, :, :] = float(board.has_kingside_castling_rights(chess.WHITE))
    tensor[16, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))
    tensor[17, :, :] = float(board.has_kingside_castling_rights(chess.BLACK))
    tensor[18, :, :] = float(board.has_queenside_castling_rights(chess.BLACK))
    
    # Add en passant possibility
    if board.ep_square:
        ep_row = 7 - (board.ep_square // 8)
        ep_col = board.ep_square % 8
        tensor[19, ep_row, ep_col] = 1.0
    
    # Add attack/defense counts
    for sq in range(64):
        row, col = 7 - (sq // 8), sq % 8
        attackers = len(board.attackers(board.turn, sq))
        defenders = len(board.attackers(not board.turn, sq))
        tensor[20, row, col] = min(1.0, (attackers - defenders) / 3.0 + 0.5)  # normalize to [0,1]
    
    # Check indicator
    tensor[21, :, :] = float(board.is_check())
    
    return tensor

class TacticDataset(Dataset):
    """
    Expects a DataFrame or CSV with columns:
      - FEN
      - Moves (space-separated UCI moves; first is the motif move)
      - simple_label
    """
    def __init__(self, data, label_map: dict, augment: bool=False):
        # data: either path string or pandas.DataFrame
        if isinstance(data, str):
            df = pd.read_csv(data)
        else:
            df = data.copy()
        df.columns = df.columns.str.lower()
        df = df.rename(columns={'simple_label':'motif'})

        self.fens    = df['fen'].tolist()
        self.moves   = df['moves'].tolist()
        self.labels  = [label_map[m] for m in df['motif']]
        self.augment = augment

    def __len__(self):
        return len(self.fens)

    def __getitem__(self, idx):
        fen  = self.fens[idx]
        move = self.moves[idx].split()[0]
        x    = fen_to_tensor(fen, last_move_uci=move)

        if self.augment:
            # Horizontal flip (more chess-legitimate)
            if random.random() < 0.5:
                x = torch.flip(x, dims=[2])
                
            # Color inversion (swap white/black pieces and update turn)
            if random.random() < 0.3:
                # Swap white and black piece planes
                white_pieces = x[0:6].clone()
                black_pieces = x[6:12].clone()
                x[0:6], x[6:12] = black_pieces, white_pieces
                # Flip side to move
                x[12] = 1.0 - x[12]
                # Update castling rights
                castle_w, castle_b = x[15:17].clone(), x[17:19].clone()
                x[15:17], x[17:19] = castle_b, castle_w
                
            # Subtle noise for robustness (controlled magnitude)
            if random.random() < 0.2:
                noise = torch.randn_like(x) * 0.02
                x = torch.clamp(x + noise, 0, 1)
                
            # Tactical feature emphasis (randomly boost certain features)
            if random.random() < 0.3:
                # Emphasize attacks/defenses or check
                feature_idx = random.choice([20, 21])
                x[feature_idx] = torch.clamp(x[feature_idx] * random.uniform(1.0, 1.5), 0, 1)

        return x, self.labels[idx]
