import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
#import matplotlib
#matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

torch.manual_seed(123)
np.random.seed(123)
train = pd.read_csv("./input/AEP_hourly_train.csv")
test = pd.read_csv("./input/AEP_hourly_test.csv")
target= "AEP_MW"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def normalize_data(df, scaler = None, test=True):
    if test != True:
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(df[target].values.reshape(-1,1))
    else:
        normalized_data = scaler.transform(df[target].values.reshape(-1,1))
    df[target] = normalized_data
    return df, scaler

def load_data(data, seq_len, multi_steps):
    X = []
    y = []
    if multi_steps == 1:
        multi_steps_len = multi_steps-1
    else:
        multi_steps_len = multi_steps
    for i in range(seq_len, len(data)-multi_steps_len):
        X.append(data.iloc[i-seq_len : i])
        y.append(data.iloc[i:i+multi_steps])

    X = np.array(X)
    X = np.reshape(X, (X.shape[0], seq_len, 1))
    y = np.array(y)
    y = np.reshape(y, (y.shape[0], multi_steps))
    return X, y


def get_validation_data(x_train, y_train, split):
    split_idx = int(x_train.shape[0] * split)
    x_valid = x_train[split_idx:, :]
    x_train = x_train[:split_idx, :]
    y_valid = y_train[split_idx:]
    y_train = y_train[:split_idx]
    return x_train, x_valid, y_train, y_valid




def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100




train_norm, scaler = normalize_data(train, test = False)
test_norm, _ = normalize_data(test, scaler)
train_norm = train_norm[target]
test_norm = test_norm[target]

#set it to 1 to have only 1 prediction per sequence
multi_steps = 1
seq_len = 24
batch_size = 32
X_train, y_train = load_data(train_norm, seq_len = 24, multi_steps = multi_steps)
X_test, y_test = load_data(test_norm, seq_len = 24, multi_steps = multi_steps)
X_train, X_valid, y_train, y_valid = get_validation_data (X_train, y_train, 0.80)

torch_X_train = torch.from_numpy(X_train).type(torch.float).to(device)
torch_y_train = torch.from_numpy(y_train).type(torch.float).to(device)
torch_y_train =torch_y_train.view(-1,multi_steps,1)


torch_X_test = torch.from_numpy(X_test).type(torch.float).to(device)
torch_y_test = torch.from_numpy(y_test).type(torch.float).to(device)
torch_y_test =torch_y_test.view(-1,multi_steps,1)

torch_X_valid = torch.from_numpy(X_valid).type(torch.float).to(device)
torch_y_valid = torch.from_numpy(y_valid).type(torch.float).to(device)
torch_y_valid =torch_y_valid.view(-1,multi_steps,1)

train = torch.utils.data.TensorDataset(torch_X_train,torch_y_train)
test = torch.utils.data.TensorDataset(torch_X_test,torch_y_test)
valid = torch.utils.data.TensorDataset(torch_X_valid,torch_y_valid)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)
val_loader = torch.utils.data.DataLoader(valid, batch_size = batch_size, shuffle=False)


# Funzione di allenamento
def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()  # Modalità allenamento
    running_loss = 0.0

    for x, y in dataloader:
        optimizer.zero_grad()  # Reset dei gradienti

        y_pred = model(x)  # Forward pass
        y_pred = y_pred.view(-1,multi_steps,1)
        loss = criterion(y_pred, y)  # Calcola la perdita
        loss.backward()  # Calcola i gradienti
        optimizer.step()  # Aggiorna i pesi
        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss


# Funzione di validazione
def validate(model, dataloader, criterion):
    model.eval()  # Modalità validazione
    running_loss = 0.0

    with torch.no_grad():  # Non calcolare i gradienti durante la validazione
        for x, y in dataloader:
            y_pred = model(x)  # Forward pass
            y_pred = y_pred.view(-1, multi_steps, 1)
            loss = criterion(y_pred, y)

            running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss

def evaluate(model, test_loader):
    running_rmse = 0
    for x_test, y_test in test_loader:
        y_pred = model(x_test)
        y_test_original = scaler.inverse_transform(y_test.view(-1,multi_steps).cpu().detach().numpy())
        y_pred_original_scale = scaler.inverse_transform(y_pred.cpu().detach().numpy())
        diff = (y_pred_original_scale - y_test_original)
        squared_diff = diff**2
        test_rmse = np.sqrt( np.mean(squared_diff))
        running_rmse += test_rmse
    rmse = running_rmse / len(test_loader)
    print("RMSE:{:.3f} ".format(rmse))



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



class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        #input size = num features
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=48, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=48, hidden_size=48, num_layers=1, batch_first=True)
        self.linear = nn.Linear(48, multi_steps)
    def forward(self, x):
        x, _ = self.lstm1(x)
        x = F.tanh(x)
        x = F.dropout(x, p=0.15, training=self.training)
        x, _ = self.lstm2(x)
        x = F.tanh(x)
        x = F.dropout(x, p=0.15, training=self.training)
        x, _ = self.lstm2(x)
        x = F.tanh(x)
        x = F.dropout(x, p=0.15, training=self.training)
        x = x[:, -1, :]
        x = self.linear(x)
        return x



early_stopping = EarlyStopping(patience=3, min_delta=0.0001)
# Inizializza il modello, la loss function e l'ottimizzatore
model = LSTM().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
# Numero di epoche di allenamento
num_epochs = 50

def training(num_epochs, model, train_loader, val_loader, criterion, optimizer):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Fase di allenamento
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        print(f"Train Loss: {train_loss:.4f}")
        # Fase di validazione
        val_loss = validate(model, val_loader, criterion)
        print(f"Validation Loss: {val_loss:.5f}")

        scheduler.step(val_loss)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

training(num_epochs, model, train_loader, val_loader, criterion, optimizer)

evaluate(model, test_loader)
#RMSE 256
#same value as tensorflow


def predict(model, input_loader):
    with torch.no_grad():
        preddictions = []
        for x,y in input_loader:
            temp_pred = model(x)
            preddictions.append(temp_pred.cpu().numpy())
        preddictions = np.concatenate(preddictions)
        return preddictions

y_pred = predict(model, test_loader)
