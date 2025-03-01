from utils_torch import evaluate, EarlyStopping, training
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
df = pd.read_csv('./input/Iris.csv')


df['Species'] = df['Species'].map({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2})

df.drop(['Id'],axis=1,inplace=True)

X = df.drop(["Species"],axis=1).values
y = df["Species"].values


X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_valid = scaler.fit_transform(X_valid)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
X_valid = torch.FloatTensor(X_valid)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)
y_valid = torch.LongTensor(y_valid)


train = torch.utils.data.TensorDataset(X_train,y_train)
test = torch.utils.data.TensorDataset(X_test,y_test)
valid = torch.utils.data.TensorDataset(X_valid,y_valid)

batch_size = 5

# Dati di esempio: usa il tuo dataset e dataloaders
# train_dataset e val_dataset sono oggetti di tipo Dataset di PyTorch
train_loader = DataLoader(train, batch_size = batch_size, shuffle=False)
test_loader = DataLoader(test, batch_size = batch_size, shuffle=False)
val_loader = DataLoader(valid, batch_size = batch_size, shuffle=False)

class NeuralNetworkClassificationModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNetworkClassificationModel, self).__init__()
        self.input_layer = nn.Linear(input_dim, 128)
        self.hidden_layer1 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.input_layer(x))
        out = self.relu(self.hidden_layer1(out))
        out = self.output_layer(out)
        return out


input_dim  = 4
output_dim = 3
model = NeuralNetworkClassificationModel(input_dim,output_dim)

early_stopping = EarlyStopping("./pytorch_iris_model_weights",patience=3, min_delta=0.001)
# Inizializza il modello, la loss function e l'ottimizzatore
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
num_epochs = 30
training(num_epochs, model, train_loader, val_loader, criterion, optimizer,scheduler,early_stopping, writer)
writer.flush()
writer.close()
model.load_state_dict(torch.load("./pytorch_iris_model_weights"))

evaluate(model, test_loader)