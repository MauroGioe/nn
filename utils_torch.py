import torch

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
    def __init__(self, path, patience=5, min_delta=0):
        self.patience = patience  # Numero di epoche senza miglioramento
        self.min_delta = min_delta  # Minimo miglioramento richiesto per essere considerato come miglioramento
        self.counter = 0  # Contatore di epoche senza miglioramento
        self.best_loss = None  # La migliore perdita di validazione vista finora
        self.early_stop = False  # Flag per fermare l'allenamento
        self.path = path
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            print("Model saved")


def training(num_epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, early_stopping, writer):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Fase di allenamento
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        # Fase di validazione
        val_loss, val_accuracy = validate(model, val_loader, criterion)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        writer.add_scalar("Loss/validation", val_loss, epoch)
        writer.add_scalar("Accuracy/validation", val_accuracy, epoch)

        scheduler.step(val_loss)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
