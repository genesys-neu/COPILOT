import numpy as np
import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import PolynomialFeatures

# CNN based model for LiDAR modality - Infocom version
class LidarNet(nn.Module):
    def __init__(self, input_dim, output_dim, fusion='ultimate'):
        super(LidarNet, self).__init__()
        dropProb1 = 0.3
        dropProb2 = 0.2
        channel = 32
        self.conv1 = nn.Conv2d(input_dim, channel, kernel_size=(3, 3), padding='same')
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=(3, 3), padding='same')
        self.pool1 = nn.MaxPool2d((2,2))
        self.pool2 = nn.MaxPool2d((1, 2))

        self.hidden_new1 = nn.Linear(800, 512)
        self.hidden_new2 = nn.Linear(512, 512)
        self.hidden_new3 = nn.Linear(512, 512)
        self.hidden3 = nn.Linear(512, 256)

        self.out = nn.Linear(256, output_dim)  # 128
        #######################
        self.drop1 = nn.Dropout(dropProb1)
        self.drop2 = nn.Dropout(dropProb2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.fusion = fusion

    def forward(self, x):
        # FOR CNN BASED IMPLEMENTATION
        # x = F.pad(x, (0, 0, 2, 1))
        a = x = self.relu(self.conv1(x))
        # x = F.pad(x, (0, 0, 2, 1))
        x = self.relu(self.conv2(x))
        # x = F.pad(x, (0, 0, 2, 1))
        x = self.relu(self.conv2(x))
        # print("Shapes: ", x.shape, a.shape)
        x = torch.add(x, a)
        x = self.pool1(x)
        b = x = self.drop1(x)

        x = self.relu(self.conv2(x))
        x = self.relu(self.conv2(x))
        x = torch.add(x, b)
        x = self.pool1(x)
        c = x = self.drop1(x)

        x = self.relu(self.conv2(x))
        x = self.relu(self.conv2(x))
        x = torch.add(x, c)
        
        x = x.reshape(x.size(0), -1)

        x = self.relu(self.hidden_new1(x))
        x = self.drop2(x)
        x = self.relu(self.hidden_new2(x))
        x = self.drop2(x)
        x = self.relu(self.hidden_new3(x))
        x = self.drop2(x)

        x = self.relu(self.hidden3(x))
        x = self.drop2(x)

        x = self.out(x) 
        
        return x