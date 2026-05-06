import cv2
import numpy as np
from PIL import Image


def overlay_heatmap(img, cam, alpha=0.4):
    """
    img: PIL Image
    cam: numpy array (H, W) normalized [0,1]
    alpha: heatmap intensity
    """

    # Convert PIL → numpy
    img = np.array(img)

    # Ensure uint8
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    # Resize CAM to match image
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))

    # Convert CAM to heatmap
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )

    # Convert image RGB → BGR (OpenCV format)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 🔥 Ensure same shape (fix your previous crash)
    if heatmap.shape != img_bgr.shape:
        heatmap = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))

    # Overlay
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap, alpha, 0)

    # Convert back to RGB
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    return overlay