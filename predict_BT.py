import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import random

# -----------------------------
# 1. Load Your Custom CNN Model
# -----------------------------

# Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            # convo layer1
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # convo layer2
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # convo layer3
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # convo layer4
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # convo layer5
            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # ✅ Global Adaptive Average Pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)



# Load model
model = CNNModel()
model.load_state_dict(torch.load("./model_good/improved_trained_model_5convo_25Epoch_98%Accuracy.pth", map_location=torch.device("cuda")))
model.eval()

print("✅ Model Loaded Successfully!")


# -----------------------------
# 2. Image Preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Tumor labels (CHANGE according to your dataset)
class_names = ["Glioma", "Meningioma", "Notumor", "Pituitary"]


# -----------------------------
# 3. Predict Function
# -----------------------------
def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    label = class_names[predicted.item()]
    return label


# -----------------------------
# 4. Pick 2–3 Random Images and Predict
# -----------------------------
IMAGE_FOLDER = "./testing/Meningioma"   # <-- CHANGE to your MRI folder

# get image paths
all_images = [os.path.join(IMAGE_FOLDER, f)
              for f in os.listdir(IMAGE_FOLDER)
              if f.lower().endswith((".jpg", ".png", ".jpeg"))]

if len(all_images) == 0:
    print(" No images found in the folder:", IMAGE_FOLDER)
    exit()

selected_images = random.sample(all_images, min(3, len(all_images)))  # pick 2-3 images

print("\n Selected Images:", selected_images)
print("\n Brain Tumor Classification Results\n")

for img_path in selected_images:
    prediction = predict_image(img_path)
    print(f"📌 Image: {os.path.basename(img_path)} --> Predicted Tumor: **{prediction}**")

print("\n Classification Completed!")
