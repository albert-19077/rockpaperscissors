import torch
import torch.nn as nn
import torch.nn.functional as F
class RockPaperScissorsCNN(nn.Module):
    """
    User's exact RockPaperScissorsCNN architecture.
    Accepts 224x224 RGB image input.
    """
    def __init__(self, num_classes=3):
        super(RockPaperScissorsCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2) # Divides dimensions by 2: 112 x 112 after pool1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # 56 x 56 after pool2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 28 x 28 after pool3
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

