import torch.nn as nn

class TacticCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(12, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(128 * 2 * 2, 256)
        self.fc2   = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (batch, 12, 8, 8)
        x = nn.functional.relu(self.conv1(x))   # → (batch,32,8,8)
        x = self.pool(nn.functional.relu(self.conv2(x)))  # → (batch,64,4,4)
        x = self.pool(nn.functional.relu(self.conv3(x)))  # → (batch,128,2,2)
        x = x.view(x.size(0), -1)               # → (batch,128*2*2)
        x = nn.functional.relu(self.fc1(x))    # → (batch,256)
        return self.fc2(x)                     # → (batch,num_classes)
