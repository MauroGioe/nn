import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable

from sklearn.model_selection import train_test_split
torch.manual_seed(0)
np.random.seed(0)

df = pd.read_csv("./input/train.csv")

y = df['label'].values
X = df.drop(['label'],axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)




batch_size = 32



torch_X_train = torch.from_numpy(X_train).type(torch.LongTensor)
torch_y_train = torch.from_numpy(y_train).type(torch.LongTensor)

# create feature and target tensor for test set.
torch_X_test = torch.from_numpy(X_test).type(torch.LongTensor)
torch_y_test = torch.from_numpy(y_test).type(torch.LongTensor)


torch_X_train = torch_X_train.view(-1, 1,28,28).float()
torch_X_test = torch_X_test.view(-1,1,28,28).float()

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(torch_X_train,torch_y_train)
test = torch.utils.data.TensorDataset(torch_X_test,torch_y_test)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)


def fit(model, train_loader, epochs = 5):
    optimizer = torch.optim.Adam(model.parameters())  # ,lr=0.001, betas=(0.9,0.999))
    error = nn.CrossEntropyLoss()
    EPOCHS = epochs
    model.train()
    for epoch in range(EPOCHS):
        correct = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            var_X_batch = Variable(X_batch).float()
            var_y_batch = Variable(y_batch)
            optimizer.zero_grad()
            output = model(var_X_batch)
            loss = error(output, var_y_batch)
            loss.backward()
            optimizer.step()

            # Total correct predictions
            predicted = torch.max(output.data, 1)[1]
            correct += (predicted == var_y_batch).sum()
            # print(correct)
            if batch_idx % 50 == 0:
                print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
                    epoch, batch_idx * len(X_batch), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                    loss.data, float(correct * 100) / float(batch_size * (batch_idx + 1))))


def evaluate(model):
    correct = 0
    for test_imgs, test_labels in test_loader:
        #print(test_imgs.shape)
        test_imgs = Variable(test_imgs).float()
        output = model(test_imgs)
        predicted = torch.max(output,1)[1]
        correct += (predicted == test_labels).sum()
    print("Test accuracy:{:.3f}% ".format( float(correct) / (len(test_loader)*batch_size)))


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(3 * 3 * 64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #print(f"SHAPE conv1: {x.shape}")
        x = F.relu(F.max_pool2d(self.conv2(x), 2))#maxpooling divides last two dimensions by half
        #print(f"SHAPE conv2: {x.shape}")
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.dropout(x, p=0.2, training=self.training)
        #print(f"SHAPE conv3: {x.shape}")
        x = x.view(-1, 3 * 3 * 64) #flattening the tensor (batch size, output of previous layer)
        #print(f"SHAPE flattten: {x.shape}")
        #x = x.flatten()
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


cnn = CNN()
print(cnn)

fit(cnn,train_loader)

evaluate(cnn)

#torch.save(cnn.state_dict(), "./pytorch_model_weights")
cnn.load_state_dict(torch.load("./pytorch_model_weights"))
