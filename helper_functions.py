from torch import nn
from torch import optim
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import classification_report


def train(model, train_loader, val_loader, save_dir, num_epochs, learning_rate, early_stopping_patience=None, criterion = nn.CrossEntropyLoss(), optimizer="adam"):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    model.to(device)
    
    if optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd": 
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)

    train_loss_list = []
    train_accuracy_list = []
    val_loss_list = []
    val_accuracy_list = []

    best_val_acc = 0
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        model.train()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_acc += torch.sum(preds == labels.data)

        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)

        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_acc.item())

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_acc.item())

        print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        scheduler.step(val_acc)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            early_stopping_counter = 0
            torch.save(model, save_dir)
        else:
            early_stopping_counter += 1

        if early_stopping_patience is not None and early_stopping_counter >= early_stopping_patience:
            print(f'Validation accuracy did not improve for {early_stopping_patience} epochs. Stopping training...')
            break
    return train_loss_list, train_accuracy_list, val_loss_list, val_accuracy_list        


def evaluate(model, validation_loader, criterion, device):
    validation_loss = 0.0
    validation_acc = 0.0
    model.eval()
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            validation_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            validation_acc += torch.sum(preds == labels.data)

        validation_loss /= len(validation_loader.dataset)
        validation_acc /= len(validation_loader.dataset)

    return validation_loss, validation_acc

def get_predicted_images(supervised, rotation, preturbation, validation_loader, classes):
    supervised.eval()
    rotation.eval()
    preturbation.eval()

    images_list = []
    labels_list = []
    with torch.no_grad():
        for inputs, labels in validation_loader:
            if len(images_list) < 8:
                inputs, labels = inputs, labels
                #supervised outputs
                outputs_supervised = supervised(inputs)
                _, preds_supervised = torch.max(outputs_supervised, 1)
                #rotation outputs
                outputs_rotation = rotation(inputs)
                _, preds_rotation = torch.max(outputs_rotation, 1)
                #preturbation outputs
                outputs_preturbation = preturbation(inputs)
                _, preds_preturbation = torch.max(outputs_preturbation, 1)

                for i in range(len(inputs)):
                    if (preds_supervised[i].item() == labels[i].item()) & (preds_rotation[i].item() == labels[i].item()) & (preds_preturbation[i].item() == labels[i].item()) & (len(images_list) < 8) & (classes[labels[i].item()] not in labels_list):
                        images_list.append(inputs[i].cpu())
                        labels_list.append(classes[labels[i].item()])
            else: 
                return images_list, labels_list      


def plot_confusion(model, validation_loader,  device, speaker_labels, title):
        
        # Calculate Confusion Matrix
        model.eval()
        predictions = []
        labels_all = []
        with torch.no_grad():
            for inputs, labels in tqdm(validation_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                for pred in preds:
                    predictions.append(pred.item())
                for label in labels:
                    labels_all.append(label.item())
        cmap = plt.cm.Blues
        cm = confusion_matrix(labels_all, predictions)

        # Instantiate Plot Variables
        cmap = plt.cm.Blues # Color map for confusion matrix
        title = title # Plot title
        ticks = np.arange(len(speaker_labels))
        fmt = 'd' # Data format
        thresh = cm.max()/2. # Treshold
        
        # Plot Confusion Matrix
        plt.figure(figsize=(15, 10))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        plt.xticks(ticks, speaker_labels, rotation=45)
        plt.yticks(ticks, speaker_labels)
        
        for (i, j) in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                     color='white' if cm[i, j] > thresh else 'black')
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

def plot_loss(train_results, validation_results):
    color = (np.random.random(),np.random.random(),np.random.random())
    plt.plot(np.arange(len(train_results)), train_results, color=color, label='Train Loss')

    color = (np.random.random(),np.random.random(),np.random.random())
    plt.plot(np.arange(len(validation_results)), validation_results, color=color, label='Validation Loss')
    plt.legend()
    plt.title('Train and Validation Loss')
    plt.show()

def plot_accuracy(train_results, validation_results):
    color = (np.random.random(),np.random.random(),np.random.random())
    plt.plot(np.arange(len(train_results)), train_results, color=color, label='Train Accuracy')

    color = (np.random.random(),np.random.random(),np.random.random())
    plt.plot(np.arange(len(validation_results)), validation_results, color=color, label='Validation Accuracy')
    plt.legend()
    plt.title('Train and Validation Accuracy')
    plt.show()

def complex_classifier():
    classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 640),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(640, 320),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(320, 15)
        )
    return classifier

def simple_classifier():
    classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 15)
        )
    return classifier
