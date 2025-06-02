import numpy as np
import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import PolynomialFeatures
from torch.utils.data import DataLoader, random_split, TensorDataset
from tqdm import tqdm
from numpy import load
from sklearn.preprocessing import OneHotEncoder
from glob import glob
from Model.LiDAR_Context_Fusion import LidarNet
import pandas as pd
from torchvision.io import read_image


def open_npz(path,key):
    data = load(path)
    return data[key] 

def custom_label(y, strategy='one_hot'):
    output = np.empty([y.shape[0], 3])
    for i in range(0,y.shape[0]):
        output[i,:] = 0
        output[i, y[i] - 1] = 1
    return output

folders = glob("Quantized Maps 2\*", recursive = True)

GT = []
first = True
for folder in folders:
    paths = glob(folder + "\*", recursive = True)

    for path in paths:
        if first == True:
            open_file = open_npz(path, 'lidar')
            data1 = open_file[0 : int(open_file.shape[0] / 3) : 1]
            data2 = open_file[int(open_file.shape[0] / 3) : int(open_file.shape[0] / 3) * 2 : 1]
            data3 = open_file[int(open_file.shape[0] / 3) * 2 : int(open_file.shape[0] / 3) * 3: 1] 

            data1 = np.expand_dims(data1, axis = 1)
            data2 = np.expand_dims(data2, axis = 1)
            data3 = np.expand_dims(data3, axis = 1)

            data = np.concatenate([data1, data2, data3], axis = 1)  

            if 'End' in path:
                GT.extend([3] * data.shape[0])
            elif 'Mid' in path:
                GT.extend([2] * data.shape[0])
            elif 'Front' in path:
                GT.extend([1] * data.shape[0])
     
            first = False

        else:
            open_file = open_npz(path, 'lidar')

            data1 = open_file[0 : int(open_file.shape[0] / 3) : 1]
            data2 = open_file[int(open_file.shape[0] / 3) : int(open_file.shape[0] / 3) * 2 : 1]
            data3 = open_file[int(open_file.shape[0] / 3) * 2 : int(open_file.shape[0] / 3) * 3: 1] 

            data1 = np.expand_dims(data1, axis = 1)
            data2 = np.expand_dims(data2, axis = 1)
            data3 = np.expand_dims(data3, axis = 1)

            data_curr = np.concatenate([data1, data2, data3], axis = 1) 
            data = np.concatenate((data, data_curr),axis = 0)

            if 'End' in path:
                GT.extend([3] * data_curr.shape[0])
            elif 'Mid' in path:
                GT.extend([2] * data_curr.shape[0])
            elif 'Front' in path:
                GT.extend([1] * data_curr.shape[0])
        
print(data.shape)
labels = np.array(GT)

np.random.seed(42)

randperm = np.random.permutation(data.shape[0])

train_data = data[randperm[:int(0.75*len(randperm))]]
validation_data = data[randperm[int(0.75*len(randperm)) : int(0.9*len(randperm))]]

train_label = labels[randperm[:int(0.75*len(randperm))]]
val_label = labels[randperm[int(0.75*len(randperm)) : int(0.9*len(randperm))]]

y_train = custom_label(train_label)
y_test = custom_label(val_label)
print(y_train.shape)

# Get cpu or gpu device for training.
device = "cpu"
print(f"Using {device} device")

X_train = torch.from_numpy(train_data).to(device)
X_validation = torch.from_numpy(validation_data).to(device)

Y_train = torch.from_numpy(y_train).to(device)
Y_validation = torch.from_numpy(y_test).to(device)

print(X_train.shape)
#one_hot = torch.nn.functional.one_hot(Y_train.to(torch.int64), 4)
#print(one_hot)

train_dataset = TensorDataset(X_train.float(), Y_train)
train_loader = DataLoader(train_dataset, batch_size= 4)

val_dataset = TensorDataset(X_validation.float(), Y_validation)
val_loader = DataLoader(val_dataset, batch_size= 4)

learning_rate = 0.0001
train_CNN = False
shuffle = True
pin_memory = True
num_workers = 1

model = LidarNet(4, 3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training Function
def train(model, criterion, optimizer, train_loader, train_loss):
    size = len(train_loader.dataset)
    running_loss = 0.0
    model.train()
    for features, labels in train_loader:
        # zero the parameter gradients
        features = features.permute(0, 1, 4, 2, 3)
        cav1 = features[:,0,:,:,:]
        cav2 = features[:,1,:,:,:]
        cav3 = features[:,2,:,:,:]

        #print(features.shape)
        optimizer.zero_grad()

        ypred = model(cav1, cav2, cav3)

        loss = criterion(ypred, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print("Training Loss : ", running_loss / size)
    train_loss.append(running_loss / size)

# Validation Function
def validate(model, criterion, test_loader, val_loss, acc):
    size = len(test_loader.dataset)
    model.eval()
    test_loss, accuracy = 0, 0
    with torch.no_grad():
        for features, labels in test_loader:

            features = features.permute(0, 1, 4, 2, 3)
            cav1 = features[:,0,:,:,:]
            cav2 = features[:,1,:,:,:]
            cav3 = features[:,2,:,:,:]

            pred = model(cav1, cav2, cav3)
            test_loss += criterion(pred, labels).item()
            output = torch.log_softmax(pred, dim = 1)
            #print(torch.tensor(torch.sum(output.argmax(1) == labels.argmax(1)).item()))
            #accuracy += (output.argmax(1) == labels).type(torch.float).sum().item()
            accuracy += torch.tensor(torch.sum(output.argmax(1) == labels.argmax(1)).item())

        print("Validation Loss : ", test_loss / size)
        val_loss.append(test_loss / size)
        print("Accuracy : ", accuracy / size)
        acc.append((accuracy / size).item())

Epochs = 20

train_loss = []
val_loss = []
accuracy = []
for epoch in range(Epochs):
    #model.load_state_dict(torch.load("Model/NN.pth"))
    train(model, criterion, optimizer, train_loader, train_loss)
    torch.save(model.state_dict(), "ML/Model/NN_Concat3.pth")
    validate(model, criterion, val_loader, val_loss, accuracy)

print(accuracy)

test_data = data[randperm[int(0.9 * len(randperm)) :: ]]
test_label = labels[randperm[int(0.9 * len(randperm)) :: ]]

y_test = custom_label(test_label)

device = "cpu"
print(f"Using {device} device")

X_test = torch.from_numpy(test_data).to(device)
Y_test = torch.from_numpy(y_test).to(device)

print(X_test.shape)
print(Y_test.shape)

test_dataset = TensorDataset(X_test.float(), Y_test)
test_loader = DataLoader(test_dataset, batch_size = 1)

model.load_state_dict(torch.load("ML/Model/NN_Concat3.pth"))

test_loss = []
accuracy = []
validate(model, criterion, test_loader, test_loss, accuracy)
