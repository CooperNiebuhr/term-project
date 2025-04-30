import os
import torch
import pandas as pd
from load_pgn.config import OUTPUT_CSV
from cnn.model   import TacticCNN
from cnn.dataset import fen_to_tensor

# Must match LABELS in train.py
LABELS = ['fork','pin','back_rank_mate','capture','none']
label_map = {i:lbl for i,lbl in enumerate(LABELS)}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TacticCNN(len(LABELS)).to(device)
    model.load_state_dict(torch.load('models/tactic_cnn.pth', map_location=device))
    model.eval()

    df = pd.read_csv(OUTPUT_CSV)  # your raw blunders
    motifs = []
    for fen in df['fen']:
        tensor = fen_to_tensor(fen).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
        pred = logits.argmax(dim=1).item()
        motifs.append(label_map[pred])

    df['motif_cnn'] = motifs
    out_path = os.path.join(os.path.dirname(OUTPUT_CSV), 'errors_with_motif_cnn.csv')
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} classified errors to {out_path}")

if __name__ == '__main__':
    main()
