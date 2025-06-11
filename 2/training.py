import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import random

from resnet34_manual import ResNet34

EPOCHS = 50
PATIENCE = 8
BATCH_SIZE = 32 #если на цп обучать, то лучше 16
LR = 1e-4

torch.manual_seed(42)
torch.cuda.manual_seed(42)
random.seed(42)

train_augmentations = transforms.Compose([
    transforms.Resize((224, 224)), #если на цп то лучше 128x128
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
torch.set_num_threads(6)
full_dataset = ImageFolder('train', transform=train_augmentations)

train_len = int(0.7 * len(full_dataset)) #Добавил разделение 70/30 (70 - обучение, 30 - валидация)
val_len = len(full_dataset) - train_len
train_set, val_set = random_split(full_dataset, [train_len, val_len])
val_set.dataset.transform = val_transform

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Без cuda не вариант, на цп в среднем минут 25
model = ResNet34(num_classes=2).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

#Обучение
train_loss_history = []
val_accuracy_history = []
best_val_acc = 0.0
epochs_no_improve = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_loss_history.append(avg_loss)
    print(f"\nEpoch {epoch + 1}/{EPOCHS} - Train Loss: {avg_loss:.4f}")

    #Валидация
    model.eval()
    correct = 0
    total = 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            total += targets.size(0)
            correct += (preds == targets).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    val_acc = correct / total * 100
    val_accuracy_history.append(val_acc)
    scheduler.step(val_acc)

    print(f"Accuracy: {val_acc:.2f}% | Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_model.pth")
        print(f"Новый лучший результат: (Accuracy: {best_val_acc:.2f}%)")
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epoch(s)")

    if epochs_no_improve >= PATIENCE: #ранняя остановка (если улучшений не было за n эпох)
        break

model.load_state_dict(torch.load("best_model.pth"))
print(classification_report(all_targets, all_preds, target_names=full_dataset.classes))
print("Confusion Matrix:")
print(confusion_matrix(all_targets, all_preds))

torch.save(model.state_dict(), "final_model.pth")
print("Модель сохранена как 'final_model.pth'")
epochs_range = range(1, len(train_loss_history) + 1)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_loss_history, label='Train Loss')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(epochs_range, val_accuracy_history, label='Validation Accuracy', color='green')
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.tight_layout()
plt.show()
