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
  
class VGG16(nn.module):
  def __init__(self, num_channels=1, number_of_classes=16):
    super(VGG16, self).__init__()
    
    self.conv1 = nn.Sequential(
      nn.Conv2d(num_channels,64,kernel_size=3),
      nn.Conv2d(64,64,3),
      nn.Relu(),
      nn.MaxPool2d(kernel_size=3,stride=2),
      nn.Relu(),
    )
    
    self.conv3 = nn.Sequential(
      nn.Conv2d(64,128,kernel_size=3),
      nn.Conv2d(128,128,3),
      nn.Relu(),
      nn.MaxPool2d(kernel_size=3,stride=2),
      nn.Relu(),
    )
    
    self.conv5 = nn.Sequential(
      nn.Conv2d(256,256,kernel_size=3),
      nn.Conv2d(256,256,3),
      nn.Relu(),
      nn.MaxPool2d(kernel_size=3,stride=2),
      nn.Relu(),
    )
    
    self.conv7 = nn.Sequential(
      nn.Conv2d(256,512,kernel_size=3),
      nn.Conv2d(512,512,3),
      nn.Conv2d(512,512,3),
      nn.Relu(),
      nn.MaxPool2d(kernel_size=3,stride=2),
      nn.Relu(),
    )
    
    self.conv10 = nn.Sequential(
      nn.Conv2d(512,512,kernel_size=3),
      nn.Conv2d(512,512,3),
      nn.Conv2d(512,512,3),
      nn.Relu(),
      nn.MaxPool2d(kernel_size=3,stride=2),
      nn.Relu(),
    )
    
    #self.flatten = Flatten()
    
    self.linear_relu_stack = nn.Sequential(
      nn.Linear(SHAPE,4096),
      nn.Relu(),
      nn.Linear(4096,4096),
      nn.Relu(),
      nn.Linear(4096,4096),
      nn.Relu(),
      nn.Linear(4096,16),
      nn.Softmax(dim=1)
    )
  
  def __call__(self, x):
    x = self.conv1(x)
    x = self.conv3(x)
    x = self.conv5(x)
    x = self.conv7(x)
    x = self.conv10(x)
    x = x.mean((2, 3))
    x = self.linear_relu_stack(x)
    return x
