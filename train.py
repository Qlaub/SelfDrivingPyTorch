import os
import torch
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Lambda
# from torchvision.transforms import ToTensor, Lambda
# from torchvision import datasets, transforms
from torch import nn
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
import time

# start timer
start_time = time.time()


# define dataset class
class DrivingDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image.double(), label  # added .double() to match input and weight types


# specify path to csv file and image directory
annotations_training = 'C:/Users/Games/Desktop/og_nn/train_data/data.txt'
img_training = 'C:/Users/Games/Desktop/og_nn/train_data'

annotations_testing = 'C:/Users/Games/Desktop/og_nn/test_data/data.txt'
img_testing = 'C:/Users/Games/Desktop/og_nn/test_data'

# paths for debugging, be sure to comment out above paths if using
# annotations_training = 'C:/Users/Games/Desktop/og_nn/debug_data/debug_data.txt'
# img_training = 'C:/Users/Games/Desktop/og_nn/debug_data'

# annotations_testing = 'C:/Users/Games/Desktop/og_nn/debug_data/debug_data.txt'
# img_testing = 'C:/Users/Games/Desktop/og_nn/debug_data'

# instantiate datasets
training_dataset = DrivingDataset(annotations_file=annotations_training, img_dir=img_training,
                                  transform=Lambda(lambda x: (x / 255.0)),
                                  target_transform=Lambda(lambda x: ((x + 540) / 1080.0)))
# 540 is the assumed degrees of rotation for one direction of the steering wheel in most cars

testing_dataset = DrivingDataset(annotations_file=annotations_testing, img_dir=img_testing,
                                 transform=Lambda(lambda x: (x / 255.0)),
                                 target_transform=Lambda(lambda x: ((x + 540) / 1080.0)))

# create dataloaders
train_dataloader = DataLoader(training_dataset, batch_size=32, shuffle=True)

test_dataloader = DataLoader(testing_dataset, batch_size=32, shuffle=True)


# define my neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(94976, 10)
        self.fc2 = nn.Linear(10, 1)
        self.double()  # added to match data types between inputs and weights

    def forward(self, x):
        x = x.view(x.size(0), 3, 455, 256)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


""" return x.squeeze(-1) had been in place prior to avoid - UserWarning: Using a target size (torch.Size([32])) that is 
different to the input size (torch.Size([32, 1])). This will likely lead to incorrect results due to broadcasting. 
Please ensure they have the same size. return F.mse_loss(input, target, reduction=self.reduction) """


# use gpu if available, cpu if not
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print('Using {} device'.format(device))

model = NeuralNetwork().to(device)
print(model)

# my hyperparameters
learning_rate = 1e-4
batch_size = 32
epochs = 6

# loss function
loss_fn = nn.MSELoss()

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# training and testing loops
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.cuda(), y.cuda()  # loads data into GPU
        y = y.unsqueeze(-1)  # added
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.cuda(), y.cuda()  # loads data into GPU
            y = y.unsqueeze(-1)  # added
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# run our training and testing loops for the number of epochs specified in our hyperparameters
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

# save model and weights
torch.save(model.state_dict(), 'model_weights.pth')
torch.save(NeuralNetwork(), 'model.pth')

print("--- {0:.2f} seconds ---".format((time.time() - start_time)))
