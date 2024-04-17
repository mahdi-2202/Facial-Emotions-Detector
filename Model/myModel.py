import torch
import torch.nn as nn
from torch.nn import functional as F




class EmotionsDetector(nn.Module):
    def __init__(self):
        super(EmotionsDetector, self).__init__()

        # 1st Convolutional layer
        self.conv1_1 = nn.Conv2d(3, 48, kernel_size=(3, 3), padding=1)
        self.bn1_1 = nn.BatchNorm2d(48)
        self.conv1_2 = nn.Conv2d(48, 48, kernel_size=(3, 3), padding=1)
        self.bn1_2 = nn.BatchNorm2d(48)
        self.conv1_3 = nn.Conv2d(48, 48, kernel_size=(3, 3), padding=1)
        self.bn1_3 = nn.BatchNorm2d(48)


        # 2nd Convolutional layer
        self.conv2_1 = nn.Conv2d(48, 64, kernel_size=(3, 3), padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.conv2_3 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        self.bn2_3 = nn.BatchNorm2d(64)


        # 3rd Convolutional layer
        self.conv3_1 = nn.Conv2d(64, 80, kernel_size=(3, 3), padding=1)
        self.bn3_1 = nn.BatchNorm2d(80)
        self.conv3_2 = nn.Conv2d(80, 80, kernel_size=(3, 3), padding=1)
        self.bn3_2 = nn.BatchNorm2d(80)
        self.conv3_3 = nn.Conv2d(80, 80, kernel_size=(3, 3), padding=1)
        self.bn3_3 = nn.BatchNorm2d(80)


        # 4th Convolutional layer
        self.conv4_1 = nn.Conv2d(80, 126, kernel_size=(3, 3), padding=1)
        self.bn4_1 = nn.BatchNorm2d(126)
        self.conv4_2 = nn.Conv2d(126, 126, kernel_size=(3, 3), padding=1)
        self.bn4_2 = nn.BatchNorm2d(126)
        self.conv4_3 = nn.Conv2d(126, 126, kernel_size=(3, 3), padding=1)
        self.bn4_3 = nn.BatchNorm2d(126)
        

        # pool layers
        self.pool1 = nn.MaxPool2d(3, 1)
        self.pool2 = nn.MaxPool2d(3, 2)
        self.pool3 = nn.MaxPool2d(3, 3)
        self.gapool = nn.AdaptiveAvgPool2d(output_size=(1, 1))


        # Fully connected layers
        self.fc = nn.Linear(126, 6)


    def forward(self, x):

        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = F.relu(self.bn1_3(self.conv1_3(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = F.relu(self.bn2_3(self.conv2_3(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = F.relu(self.bn3_3(self.conv3_3(x)))
        x = self.pool3(x)

        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = F.relu(self.bn4_3(self.conv4_3(x)))
        x = self.gapool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x
