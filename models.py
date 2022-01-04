import torch.nn as nn

class BaseLineCNN(nn.Module):
  def __init__(self, num_channels=1, number_of_classes=16):
    super(BaseLineCNN, self).__init__()

    self.conv1 = nn.Sequential(
        nn.Conv2d(num_channels, 128, 2),
        nn.BatchNorm2d(128),
        nn.ReLU(),
    )

    
    self.conv2 = nn.Sequential(
        nn.Conv2d(128, 64, 2),
        nn.BatchNorm2d(64),
        nn.ReLU(),

    )

    self.conv3 = nn.Sequential(
        nn.Conv2d(64, 32, 2),
        nn.BatchNorm2d(32),
        nn.ReLU(),
    )

    self.linear = nn.Linear(32, number_of_classes)

  def __call__(self, x):
    x = self.conv1(x)
    x = torch.nn.functional.max_pool2d(x, (2,2))
    x = self.conv2(x)
    x = torch.nn.functional.max_pool2d(x, (2,2))
    x = self.conv3(x)
    x = torch.nn.functional.max_pool2d(x, (2,2))
    x = x.mean((2, 3))
    x = self.linear(x)
    #x = torch.sigmoid(x)
    return x
