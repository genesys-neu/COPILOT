import numpy as np
import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import PolynomialFeatures
import time

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
        self.hidden1 = nn.Linear(128, 128)
        self.hidden2 = nn.Linear(128, 256)
        self.hidden_new1 = nn.Linear(512, 512)
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

    def forward(self, x1, x2, x3):
        # FOR CNN BASED IMPLEMENTATION
        # x = F.pad(x, (0, 0, 2, 1))
        a = x1 = self.relu(self.conv1(x1))
        b = x2 = self.relu(self.conv1(x2))
        c = x3 = self.relu(self.conv1(x3))
        # x = F.pad(x, (0, 0, 2, 1))
        x1, x2, x3 = self.relu(self.conv2(x1)), self.relu(self.conv2(x2)), self.relu(self.conv2(x3))
        # x = F.pad(x, (0, 0, 2, 1))
        x1, x2, x3 = self.relu(self.conv2(x1)), self.relu(self.conv2(x2)), self.relu(self.conv2(x3))
        # print("Shapes: ", x.shape, a.shape)
        x1, x2, x3 = torch.add(x1, a), torch.add(x2, b), torch.add(x3, c)
        x1, x2, x3 = self.pool1(x1), self.pool1(x2), self.pool1(x3)

        d = x1 = self.drop1(x1)
        e = x2 = self.drop1(x2)
        f = x3 = self.drop1(x3)

        x1, x2, x3 = self.relu(self.conv2(x1)), self.relu(self.conv2(x2)), self.relu(self.conv2(x3))
        x1, x2, x3 = self.relu(self.conv2(x1)), self.relu(self.conv2(x2)), self.relu(self.conv2(x3))
        x1, x2, x3 = torch.add(x1, d), torch.add(x2, e), torch.add(x3, f)
        x1, x2, x3 = self.pool1(x1), self.pool1(x2), self.pool1(x3)

        g = x1 = self.drop1(x1)
        h = x2 = self.drop1(x2)
        i = x3 = self.drop1(x3)

        x1, x2, x3 = self.relu(self.conv2(x1)), self.relu(self.conv2(x2)), self.relu(self.conv2(x3))
        x1, x2, x3 = self.relu(self.conv2(x1)), self.relu(self.conv2(x2)), self.relu(self.conv2(x3))

        x1, x2, x3 = torch.add(x1, g), torch.add(x2, h), torch.add(x3, i)
        x1, x2, x3 = self.pool1(x1), self.pool1(x2), self.pool1(x3)

        #t1 = (time.perf_counter_ns() - start)  / 3

        x1, x2, x3 = x1.reshape(x1.size(0), -1), x2.reshape(x2.size(0), -1), x3.reshape(x3.size(0), -1)

        x1, x2, x3 = self.relu(self.hidden1(x1)), self.relu(self.hidden1(x2)), self.relu(self.hidden1(x3))
        x1, x2, x3 = self.relu(self.hidden2(x1)), self.relu(self.hidden2(x2)), self.relu(self.hidden2(x3))

        x1, x2, x3 = x1.view(x1.shape[0], 1, x1.shape[1]), x2.view(x2.shape[0], 1, x2.shape[1]), x3.view(x3.shape[0], 1, x3.shape[1])

        attention1 = torch.bmm(x1, x2.permute(0,2,1))
        attention2 = torch.bmm(x1, x3.permute(0,2,1))

        #t2 = (time.perf_counter_ns() - t1) / 2

        scores = torch.cat([attention1, attention2], dim = 1)
        weight = F.softmax(scores, dim = 1)

        #t3 = time.perf_counter_ns() - t2

        cav1 = torch.bmm(x2.permute(0,2,1), weight[:,0,None]) 
        cav2 = torch.bmm(x3.permute(0,2,1), weight[:,1,None])

        fuse = torch.cat([x1.permute(0,2,1) , (cav1 + cav2)], dim = 2)
        #fuse = torch.cat([x1.squeeze, cav1.T, cav2.T], dim = 1)
        #t4 = time.perf_counter_ns() - t3

        fuse = fuse.reshape(fuse.size(0), -1)
        
        x = self.relu(self.hidden_new1(fuse))
        x = self.drop2(x)
        x = self.relu(self.hidden_new2(x))
        x = self.drop2(x)
        x = self.relu(self.hidden_new3(x))
        x = self.drop2(x)

        x = self.relu(self.hidden3(x))
        x = self.drop2(x)

        x = self.out(x) 
        #t5 = time.perf_counter_ns() - t4

        return x