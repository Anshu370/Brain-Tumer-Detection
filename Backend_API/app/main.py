from fastapi import FastAPI, UploadFile, File, Form
from typing import List
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from .model import CNNModel
from .gradcam import GradCAM
from .utils import overlay_heatmap, to_base64

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ['Glioma', 'Meningioma', 'Notumer', 'Pituitary']

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load model
model = CNNModel().to(device)
model.load_state_dict(torch.load('model/improved_trained_model_5convo_25Epoch_98%Accuracy.pth', weights_only=True))
model.eval()

# Grad-CAM
target_layer = model.features[-3]
grad_cam = GradCAM(model, target_layer)

@app.post("/test")
async def test(files: List[UploadFile] = File(...)):
    return {"msg": "ok"}

@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    images = []
    raw_images = []

    for file in files:
        img = Image.open(file.file).convert("RGB")
        raw_images.append(img)

        img_t = transform(img)
        images.append(img_t)

    batch = torch.stack(images).to(device)

    with torch.no_grad():
        outputs = model(batch)
        probs = F.softmax(outputs, dim=1)

    avg_probs = torch.mean(probs, dim=0)
    final_class = torch.argmax(avg_probs).item()

    heatmaps = []
    slice_results = []

    for i in range(len(images)):
        input_tensor = batch[i].unsqueeze(0)

        output = model(input_tensor)
        pred_class = torch.argmax(output).item()

        cam = grad_cam.generate(input_tensor, pred_class)
        overlay = overlay_heatmap(raw_images[i], cam)
        heatmaps.append(to_base64(overlay))

        slice_results.append({
            "slice": i,
            "prediction": classes[pred_class],
            "confidence": float(probs[i][pred_class].item())
        })

    return {
        "final_prediction": classes[final_class],
        "confidence": float(avg_probs[final_class].item()),
        "slice_results": slice_results,
        "heatmaps": heatmaps
    }