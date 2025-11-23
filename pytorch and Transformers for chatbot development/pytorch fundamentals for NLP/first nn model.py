import torch
import torch.nn as nn

class MNISTNET(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(784, 500)
    self.fc2 = nn.Linear(500, 10)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.fc1(x)
    x = self.fc2(x)
    x = self.relu(x)
    return x
model = MNISTNET()
dummy = torch.randn(1, 784)
print(model(dummy))
