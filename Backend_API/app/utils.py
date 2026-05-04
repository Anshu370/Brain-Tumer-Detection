import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

def overlay_heatmap(img, cam):
    img = np.array(img)

    # Resize CAM to match original image size
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    # Ensure both are same type
    if heatmap.shape != img.shape:
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    return overlay

def to_base64(img_array):
    pil_img = Image.fromarray(img_array)
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()