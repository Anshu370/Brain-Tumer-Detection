from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from typing import List
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import uuid
from huggingface_hub import hf_hub_download

from .model import CNNModel
from .gradcam import GradCAM
from .utils import overlay_heatmap

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ['Glioma', 'Meningioma', 'Notumer', 'Pituitary']

# Static folder setup

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


# Transform

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# Load model
MODEL_PATH = hf_hub_download(
    repo_id="Anshu370/brain-tumor-detection",
    filename="improved_trained_model_5convo_25Epoch_98%Accuracy.pth",
    token=os.getenv("HF_TOKEN")
)

model = CNNModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
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

    # Unique folder for this request

    request_id = str(uuid.uuid4())
    request_dir = os.path.join(OUTPUT_DIR, request_id)
    os.makedirs(request_dir, exist_ok=True)


    # Load images

    for file in files:
        img = Image.open(file.file).convert("RGB")
        raw_images.append(img)

        img_t = transform(img)
        images.append(img_t)

    batch = torch.stack(images).to(device)


    # Batch prediction

    with torch.no_grad():
        outputs = model(batch)
        probs = F.softmax(outputs, dim=1)

    avg_probs = torch.mean(probs, dim=0)
    final_class = torch.argmax(avg_probs).item()

    heatmap_urls = []
    slice_results = []


    # Per-slice Grad-CAM

    for i in range(len(images)):
        input_tensor = batch[i].unsqueeze(0)

        # Grad-CAM needs gradients → no torch.no_grad()
        output = model(input_tensor)
        pred_class = torch.argmax(output).item()

        cam = grad_cam.generate(input_tensor, pred_class)
        overlay = overlay_heatmap(raw_images[i], cam)

        # Save heatmap image
        file_name = f"heatmap_{i}.png"
        file_path = os.path.join(request_dir, file_name)
        Image.fromarray(overlay).save(file_path)

        # Create URL
        url = f"http://localhost:8000/outputs/{request_id}/{file_name}"
        heatmap_urls.append(url)

        slice_results.append({
            "slice": i,
            "prediction": classes[pred_class],
            "confidence": float(probs[i][pred_class].item())
        })

    return {
        "final_prediction": classes[final_class],
        "confidence": float(avg_probs[final_class].item()),
        "slice_results": slice_results,
        "heatmaps": heatmap_urls
    }