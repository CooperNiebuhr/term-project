# load_pgn/cnn/train.py

import os
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from load_pgn.config      import PUZZLE_DB_CSV
from load_pgn.cnn.dataset import TacticDataset
from load_pgn.cnn.model   import TacticCNN

def main():
    # 1) Read the CSV just to discover all unique motifs
    df = pd.read_csv(PUZZLE_DB_CSV)
    # Normalize as in your dataset loader
    df.columns = df.columns.str.lower()
    df = df.rename(columns={'simple_label': 'motif'})

    # 2) Build your label list and map
    LABELS = sorted(df['motif'].unique())
    label_map = {lbl: idx for idx, lbl in enumerate(LABELS)}
    print(f"Detected labels ({len(LABELS)}): {LABELS}")

    # Hyperparameters
    batch_size = 64
    epochs     = 10
    lr         = 1e-3
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 3) Create the dataset and split
    dataset = TacticDataset(PUZZLE_DB_CSV, label_map)
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    # 4) Model, loss, optimizer
    model     = TacticCNN(len(LABELS)).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 5) Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        for boards, labels in train_loader:
            boards, labels = boards.to(device), labels.to(device)
            preds = model(boards)
            loss  = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for boards, labels in val_loader:
                boards, labels = boards.to(device), labels.to(device)
                pred = model(boards).argmax(dim=1)
                correct += (pred == labels).sum().item()
                total   += labels.size(0)
        acc = correct / total
        print(f"Epoch {epoch}/{epochs} — val_acc: {acc:.3f}")

    # 6) Save the trained model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/tactic_cnn.pth')
    print("Model saved to models/tactic_cnn.pth")

if __name__ == '__main__':
    main()
