import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class CnnNetVoice(nn.Module):
    def __init__(self):
        super(CnnNetVoice, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (3,2), padding=(1,1))
        self.conv2 = nn.Conv2d(16, 32, (3,2))
        self.conv3 = nn.Conv2d(32, 64, (3,2))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2944, 1)
        # self.simgmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        # x = self.simgmoid(x)
        return x

# cnnNetVoice = CnnNetVoice()
# print(cnnNetVoice)
# input = torch.ones(1,1,50,2)
# output = cnnNetVoice(input)
# print(output)
#
# writer = SummaryWriter("./log")
# writer.add_graph(cnnNetVoice,input)
# writer.close()