import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Task 1: Setup & Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

images, labels = next(iter(train_loader))
print("Batch shape:", images.shape)

# Task 2: Define a Custom Neural Network
class MyNeuralNet(nn.Module):
    def __init__(self):
        super(MyNeuralNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Experiment A: Add Dropout
class MyNeuralNetDropout(nn.Module):
    def __init__(self):
        super(MyNeuralNetDropout, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_and_evaluate(model, optimizer, criterion, train_loader, test_loader, num_epochs=5):
    model = model.to(device)
    train_losses = []
    train_accuracies = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}")
    # Evaluate on test set
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.4f}")
    return train_losses, train_accuracies, test_acc, all_preds, all_labels

# Task 3: Training Loop from Scratch (Adam)
print("\nTraining MyNeuralNet with Adam optimizer:")
model = MyNeuralNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
adam_losses, adam_accs, adam_test_acc, adam_preds, adam_labels = train_and_evaluate(
    model, optimizer, criterion, train_loader, test_loader, num_epochs=5
)

# Task 5A: Dropout Experiment
print("\nTraining MyNeuralNetDropout with Adam optimizer:")
model_dropout = MyNeuralNetDropout()
optimizer_dropout = optim.Adam(model_dropout.parameters(), lr=0.001)
dropout_losses, dropout_accs, dropout_test_acc, dropout_preds, dropout_labels = train_and_evaluate(
    model_dropout, optimizer_dropout, criterion, train_loader, test_loader, num_epochs=5
)

# Task 5B: SGD Optimizer Experiment
print("\nTraining MyNeuralNet with SGD optimizer:")
model_sgd = MyNeuralNet()
optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.01, momentum=0.9)
sgd_losses, sgd_accs, sgd_test_acc, sgd_preds, sgd_labels = train_and_evaluate(
    model_sgd, optimizer_sgd, criterion, train_loader, test_loader, num_epochs=5
)

# Task 4: Evaluation and Visualization (for Adam run)
plt.figure()
plt.plot(range(1, 6), adam_losses, label='Adam')
plt.plot(range(1, 6), dropout_losses, label='Dropout')
plt.plot(range(1, 6), sgd_losses, label='SGD')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss vs Epochs')
plt.legend()
plt.show()

# Confusion matrix for Adam run
cm = confusion_matrix(adam_labels, adam_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.classes)
disp.plot(xticks_rotation='vertical')
plt.title('Confusion Matrix (Adam)')
plt.show()

# Show 10 misclassified images (Adam run)
misclassified = []
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        for img, pred, label in zip(images, preds, labels):
            if pred != label:
                misclassified.append((img.cpu(), pred.cpu(), label.cpu()))
            if len(misclassified) >= 10:
                break
        if len(misclassified) >= 10:
            break

plt.figure(figsize=(10, 4))
for i, (img, pred, label) in enumerate(misclassified):
    plt.subplot(2, 5, i+1)
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(f"P:{pred.item()} A:{label.item()}")
    plt.axis('off')
plt.suptitle("10 Misclassified Images (Adam)")
plt.show()

# Task 5: Accuracy and loss comparison
print("\n--- Accuracy and Loss Comparison ---")
print(f"Adam:      Final Train Acc: {adam_accs[-1]:.4f}, Test Acc: {adam_test_acc:.4f}")
print(f"Dropout:   Final Train Acc: {dropout_accs[-1]:.4f}, Test Acc: {dropout_test_acc:.4f}")
print(f"SGD:       Final Train Acc: {sgd_accs[-1]:.4f}, Test Acc: {sgd_test_acc:.4f}")

print("\n--- Observations ---")
print("Dropout reduces overfitting, so test accuracy may improve or remain stable even if train accuracy drops.")
print("SGD optimizer converges slower than Adam, but may generalize better in some cases.")
