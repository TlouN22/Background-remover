import io
import cv2
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from u2net_model import U2NET

app = FastAPI()

# ✅ CORS fix for local UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow requests from any origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "u2net/u2net.pth"

# ---------- LOAD MODEL ----------
net = U2NET(3, 1)
net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
net.to(DEVICE)
net.eval()

transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@app.post("/remove-bg")
async def remove_bg(file: UploadFile = File(...)):
    img = Image.open(file.file).convert("RGB")
    orig_w, orig_h = img.size

    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        d1, *_ = net(x)
        pred = torch.sigmoid(d1)[0, 0].cpu().numpy()

    alpha = cv2.resize(pred, (orig_w, orig_h))
    alpha = np.clip(alpha, 0, 1)

    fg_mask = (alpha > 0.6).astype(np.uint8) * 255

    contours, hierarchy = cv2.findContours(
        fg_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    clean_mask = np.zeros_like(fg_mask)

    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for i, cnt in enumerate(contours):
            if hierarchy[i][3] == -1:
                cv2.drawContours(clean_mask, [cnt], -1, 255, -1)

    rgb = np.array(img)

    bg_pixels = np.vstack([
        rgb[0:10, 0:10].reshape(-1, 3),
        rgb[-10:, 0:10].reshape(-1, 3),
        rgb[0:10, -10:].reshape(-1, 3),
        rgb[-10:, -10:].reshape(-1, 3),
    ])

    bg_color = np.median(bg_pixels, axis=0)

    color_dist = np.linalg.norm(
        rgb.astype(np.int16) - bg_color.astype(np.int16),
        axis=2
    )

    hole_pixels = (color_dist < 15) & (clean_mask == 255)
    clean_mask[hole_pixels] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(clean_mask, kernel)
    edge = clean_mask - eroded
    edge = cv2.GaussianBlur(edge, (5, 5), 0)

    alpha_final = np.maximum(eroded, edge)

    rgba = np.dstack((rgb, alpha_final))

    _, png = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

    return StreamingResponse(
        io.BytesIO(png.tobytes()),
        media_type="image/png"
    )
