import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ✅ Normalized Transform (mean/std from ImageNet for stability)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

print("Train Transform: ", train_transform)
print("Test Transform: ", test_transform)

class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['Glioma', 'Meningioma', 'Notumer', 'Pituitary']
        self.image_paths, self.labels = [], []

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# ✅ Dataset Split (Train/Val/Test)
full_train = BrainTumorDataset('./final_Data_YN_with_categ', transform=train_transform)
test_dataset = BrainTumorDataset('./testing', transform=test_transform)

train_size = int(0.9 * len(full_train))
val_size = len(full_train) - train_size
train_dataset, val_dataset = random_split(full_train, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model
class CNNModel(nn.Module):
    # def __init__(self):
    #     super(CNNModel, self).__init__()
    #     self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
    #     self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
    #     self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
    #     self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    #     self.fc1 = nn.Linear(64 * 28 * 28, 512)
    #     self.fc2 = nn.Linear(512, 4)  # 4 classes
    #
    # def forward(self, x):
    #     x = self.pool(nn.functional.relu(self.conv1(x)))
    #     x = self.pool(nn.functional.relu(self.conv2(x)))
    #     x = self.pool(nn.functional.relu(self.conv3(x)))
    #     x = x.view(-1, 64 * 28 * 28)  # Flatten
    #     x = nn.functional.relu(self.fc1(x))
    #     x = self.fc2(x)
    #     return x

    def __init__(self):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            # convo layer1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # convo layer2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # convo layer3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # convo layer4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # ✅ Global Adaptive Average Pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 4)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

save_path = './model/trained_model.pth'

# Ensure the directory exists before saving
os.makedirs(os.path.dirname(save_path), exist_ok=True)


# Train function
def train_model(model, criterion, optimizer, scheduler, num_epochs=20, save_path='./model/improved_trained_model.pth'):
    best_val_acc = 0
    for epoch in range(num_epochs):
        model.train()
        total, correct, running_loss = 0, 0, 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_acc = correct / total
        val_acc = evaluate_model(model, val_loader)
        scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {running_loss/len(train_dataset):.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

    print(f"Training Complete ✅ | Best Val Accuracy: {best_val_acc:.4f}")


def evaluate_model(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return correct / total


def test_model(model):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("\nTest Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Glioma', 'Meningioma', 'Notumer', 'Pituitary']))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Glioma', 'Meningioma', 'Notumer', 'Pituitary'],
                yticklabels=['Glioma', 'Meningioma', 'Notumer', 'Pituitary'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    print(f"Weighted Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}")

    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    print(f"Macro-averaged Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}")


train_model(model, criterion, optimizer, scheduler, num_epochs=20)
model.load_state_dict(torch.load(save_path))
test_model(model)

