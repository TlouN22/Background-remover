# ================= infer.py =================
# U²-Net background removal
# HARD background, SOFT edges, HOLE removal via color

import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from u2net_model import U2NET

# ---------------- CONFIG ----------------
IMAGE_PATH = r"C:\Users\IC\Desktop\Background remover\test.png"
OUTPUT_PATH = r"C:\Users\IC\Desktop\Background remover\cutout.png"
MODEL_PATH = r"C:\Users\IC\Desktop\Background remover\u2net\u2net.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------

print("🚀 infer.py started")
print("Device:", DEVICE)

# ---------- LOAD MODEL ----------
net = U2NET(3, 1)
net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
net.to(DEVICE)
net.eval()
print("✅ Model loaded")

# ---------- PREPROCESS ----------
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

img = Image.open(IMAGE_PATH).convert("RGB")
orig_w, orig_h = img.size
x = transform(img).unsqueeze(0).to(DEVICE)

# ---------- INFERENCE ----------
with torch.no_grad():
    d1, *_ = net(x)
    pred = torch.sigmoid(d1)[0, 0].cpu().numpy()

print("✅ Prediction done")

# ---------- POSTPROCESS ----------
alpha = cv2.resize(pred, (orig_w, orig_h))
alpha = np.clip(alpha, 0, 1)

# ---------------------------------------------------
# STEP 1: HARD FOREGROUND MASK FROM MODEL
# ---------------------------------------------------
fg_mask = (alpha > 0.6).astype(np.uint8) * 255

# ---------------------------------------------------
# STEP 2: REMOVE INNER HOLES USING CONTOURS
# ---------------------------------------------------
contours, hierarchy = cv2.findContours(
    fg_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
)

clean_mask = np.zeros_like(fg_mask)

if hierarchy is not None:
    hierarchy = hierarchy[0]
    for i, cnt in enumerate(contours):
        # parent == -1 → outer contour only
        if hierarchy[i][3] == -1:
            cv2.drawContours(clean_mask, [cnt], -1, 255, thickness=-1)

# ---------------------------------------------------
# STEP 3: COLOR-BASED BACKGROUND PUNCH (FOR HOLES)
# ---------------------------------------------------
rgb = cv2.cvtColor(cv2.imread(IMAGE_PATH), cv2.COLOR_BGR2RGB)

# Sample background color from image corners
bg_pixels = np.vstack([
    rgb[0:10, 0:10].reshape(-1, 3),
    rgb[-10:, 0:10].reshape(-1, 3),
    rgb[0:10, -10:].reshape(-1, 3),
    rgb[-10:, -10:].reshape(-1, 3),
])

bg_color = np.median(bg_pixels, axis=0)

# Compute color distance
color_dist = np.linalg.norm(
    rgb.astype(np.int16) - bg_color.astype(np.int16),
    axis=2
)

COLOR_THRESH = 15  # 10–20 safe range

# Punch background-colored pixels inside object
hole_pixels = (color_dist < COLOR_THRESH) & (clean_mask == 255)
clean_mask[hole_pixels] = 0

# ---------------------------------------------------
# STEP 4: SOFT EDGE (SAFE – NO GREY BG)
# ---------------------------------------------------
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
eroded = cv2.erode(clean_mask, kernel)
edge = clean_mask - eroded
edge = cv2.GaussianBlur(edge, (5, 5), 0)

alpha_final = np.maximum(eroded, edge)

# ---------------------------------------------------
# STEP 5: COMPOSE RGBA
# ---------------------------------------------------
rgba = np.dstack((rgb, alpha_final))

cv2.imwrite(OUTPUT_PATH, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

print("✅ Background removed (holes preserved, no grey haze)")
print(OUTPUT_PATH)
print("🏁 infer.py finished")
