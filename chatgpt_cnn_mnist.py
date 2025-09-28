import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
torch.manual_seed(123)
np.random.seed(123)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


df = pd.read_csv("./input/train.csv")

y = df['label'].values
X = df.drop(['label'],axis=1).values

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.10)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.10)

X_train=X_train / 255.0
X_test=X_test / 255.0
X_valid=X_valid / 255.0

def data_augmentation_fun(X_train:np.ndarray)->torch.Tensor:
    data_augmentation = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    #transforms.RandomRotation((15, 345)),
    transforms.RandomCrop([28, 28]),
    transforms.ToTensor()
    ])

    list_tensor = []
    for array in X_train:
        list_tensor.append(data_augmentation(array.reshape((28,28))))

    X_train = torch.stack(list_tensor)
    X_train = X_train.to(device)
    return X_train


#torch_X_train = data_augmentation_fun(X_train)
torch_X_train = torch.from_numpy(X_train).type(torch.FloatTensor).to(device)
torch_y_train = torch.from_numpy(y_train).type(torch.LongTensor).to(device)

# create feature and target tensor for test set.
torch_X_test = torch.from_numpy(X_test).type(torch.FloatTensor).to(device)
torch_y_test = torch.from_numpy(y_test).type(torch.LongTensor).to(device)


torch_X_valid = torch.from_numpy(X_valid).type(torch.FloatTensor).to(device)
torch_y_valid = torch.from_numpy(y_valid).type(torch.LongTensor).to(device)



torch_X_train = torch_X_train.view(-1, 1,28,28).float()
torch_X_test = torch_X_test.view(-1,1,28,28).float()
torch_X_valid = torch_X_valid.view(-1,1,28,28).float()

from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    def __init__(self, X, y=None, is_test=False, transforms=None):
        self.X = X
        self.y = y
        self.is_test = is_test
        self.transforms = transforms

    def __len__(self):
        return len(self.X)


    def __getitem__(self, index):
        image = self.X[index].reshape(28,28)

        if self.transforms:
            image = self.transforms(image)
        if self.is_test:
            return image
        else:
            return image.to(device), torch.from_numpy(np.array(self.y[index])).type(torch.LongTensor).to(device)

data_augmentation = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    #transforms.RandomRotation((15, 345)),
    transforms.RandomCrop([28, 28]),
    transforms.ToTensor(),
    ])
#train = MNISTDataset(X_train, y_train,  is_test=False, transforms=data_augmentation)
train = torch.utils.data.TensorDataset(torch_X_train,torch_y_train)
test = torch.utils.data.TensorDataset(torch_X_test,torch_y_test)
valid = torch.utils.data.TensorDataset(torch_X_valid,torch_y_valid)

batch_size = 64

# Dati di esempio: usa il tuo dataset e dataloaders
# train_dataset e val_dataset sono oggetti di tipo Dataset di PyTorch
train_loader = DataLoader(train, batch_size = batch_size, shuffle=False)
test_loader = DataLoader(test, batch_size = batch_size, shuffle=False)
val_loader = DataLoader(valid, batch_size = batch_size, shuffle=False)

# Funzione di allenamento
def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()  # Modalità allenamento
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        optimizer.zero_grad()  # Reset dei gradienti

        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Calcola la perdita
        loss.backward()  # Calcola i gradienti
        optimizer.step()  # Aggiorna i pesi

        running_loss += loss.item()
        predicted = torch.max(outputs.data, 1)[1]
        total += labels.size(0)
        correct += (predicted == labels).sum()

    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total * 100
    return avg_loss, accuracy


# Funzione di validazione
def validate(model, dataloader, criterion):
    model.eval()  # Modalità validazione
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Non calcolare i gradienti durante la validazione
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total * 100
    return avg_loss, accuracy

def evaluate(model, test_loader):
    correct = 0
    total = 0
    for test_imgs, test_labels in test_loader:
        #print(test_imgs.shape)
        output = model(test_imgs)
        predicted = torch.max(output,1)[1]
        total += test_labels.size(0)
        correct += (predicted == test_labels).sum()
    accuracy = correct / total * 100
    print("Test accuracy:{:.3f}% ".format(accuracy))




class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience  # Numero di epoche senza miglioramento
        self.min_delta = min_delta  # Minimo miglioramento richiesto per essere considerato come miglioramento
        self.counter = 0  # Contatore di epoche senza miglioramento
        self.best_loss = None  # La migliore perdita di validazione vista finora
        self.early_stop = False  # Flag per fermare l'allenamento

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), "./lstm_pytorch_model_weights")
            print("Model saved")
        else :
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


# Definisci il modello (ad esempio, un semplice CNN)


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


early_stopping = EarlyStopping(patience=3, min_delta=0.001)
# Inizializza il modello, la loss function e l'ottimizzatore
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
# Numero di epoche di allenamento
num_epochs = 50

def training(num_epochs, model, train_loader, val_loader, criterion, optimizer):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Fase di allenamento
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        # Fase di validazione
        val_loss, val_accuracy = validate(model, val_loader, criterion)
        print(f"Validation Loss: {val_loss:.5f}, Validation Accuracy: {val_accuracy:.2f}%")
        writer.add_scalar("Loss/validation", val_loss, epoch)
        writer.add_scalar("Accuracy/validation", val_accuracy, epoch)

        scheduler.step(val_loss)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

training(num_epochs, model, train_loader, val_loader, criterion, optimizer)
writer.flush()
writer.close()
#torch.save(model.state_dict(), "./gpt_pytorch_model_weights")
model.load_state_dict(torch.load("./gpt_pytorch_model_weights"))

evaluate(model, test_loader)
#99.167
def predict(model, input_loader):
    with torch.no_grad():
        class_predictions = []
        for x,y in input_loader:
            pred = model(x)
            class_predictions.append( torch.max(pred, 1)[1].cpu().numpy())
        class_predictions = np.concatenate(class_predictions)
        return class_predictions

result = predict(model, test_loader)
len(result)
