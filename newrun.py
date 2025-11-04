import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from torchvision import transforms

# --------------------
# Model config — must match training
# --------------------
NUM_CLASSES = 6
IMG_SIZE = 128               # use 224 if you trained a ResNet at 224
WEIGHTS_PATH = "finger_counter_torch_best.pth"

# --------------------
# Model (simple CNN version)
# If you trained with ResNet-18, replace this whole class with your ResNet wrapper.
# --------------------
class FingerCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1   = nn.Linear(128 * 16 * 16, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2   = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

# --------------------
# Preprocess (must mirror test-time transforms in train.py)
# --------------------
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# --------------------
# Hand detection + cropping (OpenCV only)
# --------------------
def detect_hand_bbox_bgr(frame_bgr):
    """
    Heuristic skin detection in YCrCb.
    Returns (x, y, w, h) or None if not found.
    """
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    mask = cv2.medianBlur(mask, 5)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    if w * h < 500:
        return None
    return (x, y, w, h)

def crop_with_padding(frame_bgr, bbox, pad_ratio=0.25):
    H, W = frame_bgr.shape[:2]
    x, y, w, h = bbox
    pad_w, pad_h = int(w * pad_ratio), int(h * pad_ratio)
    x0 = max(0, x - pad_w)
    y0 = max(0, y - pad_h)
    x1 = min(W, x + w + pad_w)
    y1 = min(H, y + h + pad_h)
    return frame_bgr[y0:y1, x0:x1], (x0, y0, x1, y1)

def draw_label(frame, label, conf):
    text = f"Fingers: {label}  ({conf*100:.1f}%)"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 2, cv2.LINE_AA)

# --------------------
# Main
# --------------------
def main(weights_path=WEIGHTS_PATH, cam_index=0, use_cuda=True, smooth=5):
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    model = FingerCNN().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    # simple temporal smoothing buffer
    pred_hist = deque(maxlen=max(1, smooth))

    with torch.no_grad():
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            # 1) Find hand and crop ROI (with padding)
            bbox = detect_hand_bbox_bgr(frame_bgr)
            if bbox is not None:
                crop_bgr, (x0, y0, x1, y1) = crop_with_padding(frame_bgr, bbox, pad_ratio=0.25)
                cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (0, 255, 0), 2)  # draw for user
            else:
                # fallback: square center crop
                H, W = frame_bgr.shape[:2]
                side = min(H, W)
                x0 = (W - side) // 2
                y0 = (H - side) // 2
                crop_bgr = frame_bgr[y0:y0+side, x0:x0+side]

            # 2) Preprocess and infer
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            inp = preprocess(crop_rgb).unsqueeze(0).to(device)  # [1,3,IMG,IMG]

            logits = model(inp)
            probs = torch.softmax(logits, dim=1)[0]
            pred  = int(torch.argmax(probs).item())
            conf  = float(probs[pred].item())

            # 3) Temporal smoothing to reduce flicker
            pred_hist.append(pred)
            smoothed_pred = max(set(pred_hist), key=pred_hist.count)

            # 4) Draw label on the original frame
            draw_label(frame_bgr, smoothed_pred, conf)
            cv2.imshow("Finger Counter (press q to quit)", frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Try cam_index=1/2 if you have multiple cameras
    main()
