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
from Model.LiDAR_NN import LidarNet

def open_npz(path,key):
    data = load(path)
    return data[key] 

def custom_label(y, strategy='one_hot'):
    output = np.empty([y.shape[0], 3])
    for i in range(0,y.shape[0]):
        output[i,:] = 0
        output[i, y[i] - 1] = 1
    return output

def dataset(set = "Train"):
    if set == 'Train':
        folders = glob("Category wise Testing\LOS\*", recursive = True)

    elif set == 'Test':
        folders = glob("Quantized Maps 2\*", recursive = True)

    GT = []
    first = True
    for folder in folders:
        if set != 'Test' or 'NLOS' in folder:
            print(folder)
            paths = glob(folder + "\*", recursive = True)

            for path in paths:
                open_file = open_npz(path, 'lidar')
                np.random.seed(42)
                np.random.shuffle(open_file)
                print(open_file.shape)
                if 'End' in path:
                    data1 = open_file[0 : int(open_file.shape[0] / 3) : 1]

                elif 'Mid' in path:
                    data1 = open_file[int(open_file.shape[0] / 3) : int(open_file.shape[0] / 3) * 2 : 1] 

                if 'Front' in path:
                    data1 = open_file[int(open_file.shape[0] / 3) * 2 : int(open_file.shape[0] / 3) * 3: 1] 

                if first == True: 
                    first = False
                    data = data1
                    if 'End' in path:
                        GT.extend([3] * data1.shape[0])
                    elif 'Mid' in path:
                        GT.extend([2] * data1.shape[0])
                    elif 'Front' in path:
                        GT.extend([1] * data1.shape[0])

                else:
                    data = np.concatenate((data, data1),axis = 0)

                    if 'End' in path:
                        GT.extend([3] * data1.shape[0])
                    elif 'Mid' in path:
                        GT.extend([2] * data1.shape[0])
                    elif 'Front' in path:
                        GT.extend([1] * data1.shape[0])

    return data, GT      
        
data, GT = dataset("Train")
labels = np.array(GT)

np.random.seed(42)

randperm = np.random.permutation(data.shape[0])

train_data = data[randperm[ : int(0.8 * len(randperm))]]
validation_data = data[randperm[int(0.8 * len(randperm)) ::]]

train_label = labels[randperm[ : int(0.8 * len(randperm))]]
val_label = labels[randperm[int(0.8 * len(randperm)) ::]]

y_train = custom_label(train_label)
y_test = custom_label(val_label)
print(y_train.shape)


# Get cpu or gpu device for training.
device = "cuda"
print(f"Using {device} device")

X_train = torch.from_numpy(train_data).to(device)
X_validation = torch.from_numpy(validation_data).to(device)
Y_train = torch.from_numpy(y_train).to(device)
Y_validation = torch.from_numpy(y_test).to(device)

print(Y_train)
#one_hot = torch.nn.functional.one_hot(Y_train.to(torch.int64), 4)
#print(one_hot)

train_dataset = TensorDataset(X_train.float(), Y_train)
train_loader = DataLoader(train_dataset, batch_size= 4)

val_dataset = TensorDataset(X_validation.float(), Y_validation)
val_loader = DataLoader(val_dataset, batch_size= 4)

num_epochs = 20
learning_rate = 0.0001
train_CNN = False
shuffle = True
pin_memory = True
num_workers = 1

model = LidarNet(4, 3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training Function
def train(model, criterion, optimizer, train_loader):
    size = len(train_loader.dataset)
    running_loss = 0.0
    model.train()
    for features, labels in train_loader:
        # zero the parameter gradients
        optimizer.zero_grad()

        ypred = model(features.permute(0,3,1,2))

        loss = criterion(ypred, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print("Training Loss : ", running_loss / size)

# Validation Function
def validate(model, criterion, test_loader):
    size = len(test_loader.dataset)
    model.eval()
    test_loss, accuracy = 0, 0
    with torch.no_grad():
        for features, labels in test_loader:
            pred = model(features.permute(0,3,1,2))
            test_loss += criterion(pred, labels).item()
            output = torch.log_softmax(pred, dim = 1)
            #print(torch.tensor(torch.sum(output.argmax(1) == labels.argmax(1)).item()))
            #accuracy += (output.argmax(1) == labels).type(torch.float).sum().item()
            accuracy += torch.tensor(torch.sum(output.argmax(1) == labels.argmax(1)).item())

        print("Validation Loss : ", test_loss / size)
        print("Accuracy : ", accuracy / size)

Epochs = 20

for epoch in range(Epochs):
    #model.load_state_dict(torch.load("Model/NN.pth"))
    train(model, criterion, optimizer, train_loader)
    torch.save(model.state_dict(), "ML/Model/NN.pth")
    validate(model, criterion, val_loader)


test_data, test_GT = dataset('Test')
test_label = np.array(test_GT)
model.load_state_dict(torch.load("ML/Model/NN.pth"))

randperm = np.random.permutation(test_data.shape[0])

test_data = test_data[randperm[ : int(0.2 * len(randperm))]]
test_label = test_label[randperm[ : int(0.2 * len(randperm))]]

y_test = custom_label(test_label)

device = "cuda"
print(f"Using {device} device")

X_test = torch.from_numpy(test_data).to(device)
Y_test = torch.from_numpy(y_test).to(device)

print(X_test.shape)
print(Y_test.shape)

test_dataset = TensorDataset(X_test.float(), Y_test)
test_loader = DataLoader(test_dataset, batch_size= 4)

model.load_state_dict(torch.load("ML/Model/NN.pth"))

test_loss = []
accuracy = []
validate(model, criterion, test_loader)

