import argparse, collections, math
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms, models

# -------------------- config --------------------

IMG_SIZE = 224
NUM_CLASSES = 6

# Match training normalization (ImageNet mean/std)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Smoothing / UI
SMOOTH_WIN = 11
EMA_ALPHA  = 0.35
MIN_CONF   = 0.55
PAD_RATIO  = 0.40
MIN_AREA   = 4000

# Make the ROI look like the dataset:
# - white hand on black, centered, square, optional deskew
DESKEW = True               # rotate mask so the principal axis is vertical-ish
SHOW_PROC = True            # show the processed ROI preview
BORDER_RATIO = 0.10         # extra black border around centered mask

LABELS = [str(i) for i in range(NUM_CLASSES)]  # "0".."5"

# -------------------- model --------------------

def build_model(num_classes=6):
    m = models.resnet18()
    m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    return m

# "Classifier" preprocess (expects an image already in dataset style)
to_tensor_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# -------------------- optional ROI helpers --------------------

def try_mediapipe():
    try:
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return hands, mp
    except Exception:
        return None, None

def roi_from_mediapipe(frame_bgr, hands_mp):
    hands, mp = hands_mp
    if hands is None:
        return None
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = hands.process(img_rgb)
    if not res.multi_hand_landmarks:
        return None

    h, w, _ = frame_bgr.shape
    lm = res.multi_hand_landmarks[0]
    xs = [int(min(max(p.x * w, 0), w-1)) for p in lm.landmark]
    ys = [int(min(max(p.y * h, 0), h-1)) for p in lm.landmark]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh  = (x2 - x1), (y2 - y1)
    padw, padh = int(bw * PAD_RATIO), int(bh * PAD_RATIO)
    x1 = max(0, int(cx - (bw//2 + padw)))
    x2 = min(w, int(cx + (bw//2 + padw)))
    y1 = max(0, int(cy - (bh//2 + padh)))
    y2 = min(h, int(cy + (bh//2 + padh)))
    return (x1, y1, x2, y2)

def roi_from_skin(frame_bgr):
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    lower = np.array((0, 133, 77), dtype=np.uint8)
    upper = np.array((255, 173, 127), dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < MIN_AREA:
        return None
    x, y, w, h = cv2.boundingRect(cnt)

    cx, cy = x + w/2, y + h/2
    padw, padh = int(w * PAD_RATIO), int(h * PAD_RATIO)
    x1 = max(0, int(cx - (w//2 + padw)))
    x2 = min(frame_bgr.shape[1], int(cx + (w//2 + padw)))
    y1 = max(0, int(cy - (h//2 + padh)))
    y2 = min(frame_bgr.shape[0], int(cy + (h//2 + padh)))
    return (x1, y1, x2, y2)

# -------------------- dataset-style preprocessor --------------------

def to_dataset_style(roi_bgr):
    """
    Convert ROI to the training set look:
      - white hand (255) on black (0) background
      - keep only the largest contour
      - (optional) deskew to upright
      - center on square black canvas with a small border
      - output: HxW (IMG_SIZE) 3-channel grayscale-duplicated image
    """
    # grayscale + light blur (robust threshold)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # Otsu threshold to binary
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Decide polarity: prefer white hand on black
    # If background is mostly white, invert.
    if np.mean(mask) > 127:
        mask = cv2.bitwise_not(mask)

    # Morph cleanup
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    # Largest contour only
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        # fall back to black canvas
        blank = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        return blank

    cnt = max(cnts, key=cv2.contourArea)
    solid = np.zeros_like(mask)
    cv2.drawContours(solid, [cnt], -1, 255, thickness=cv2.FILLED)

    # Optional deskew by minimum-area rectangle angle
    if DESKEW and cv2.contourArea(cnt) > 0:
        rect = cv2.minAreaRect(cnt)
        angle = rect[-1]
        # OpenCV returns angle in [-90,0); map to a small rotation
        if angle < -45:
            angle = angle + 90
        # rotate the filled mask
        h, w = solid.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        solid = cv2.warpAffine(solid, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

    # Crop tight box around the (possibly rotated) contour again
    cnts2, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt2 = max(cnts2, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(cnt2)

    # Square canvas, centered with a border
    side = int(max(w, h) * (1 + BORDER_RATIO*2))
    side = max(side, 1)
    canvas = np.zeros((side, side), dtype=np.uint8)
    # center the hand on the square
    x0 = (side - w)//2
    y0 = (side - h)//2
    canvas[y0:y0+h, x0:x0+w] = solid[y:y+h, x:x+w]

    # Final resize to model size and expand to 3 channels (white hand)
    canvas = cv2.resize(canvas, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    # Ensure white hand on black
    canvas3 = cv2.merge([canvas, canvas, canvas])  # 3-channel grayscale

    return canvas3

# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to best checkpoint (.pth)")
    ap.add_argument("--camera", type=int, default=0, help="Webcam index")
    args = ap.parse_args()

    # Load model/weights
    model = build_model(NUM_CLASSES).to(DEVICE)
    ckpt = torch.load(args.weights, map_location=DEVICE)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    # Try MediaPipe first (falls back to skin mask)
    hands_mp = try_mediapipe()

    cap = cv2.VideoCapture(args.camera)
    assert cap.isOpened(), "Could not open camera."

    ema = None
    last_label = None
    stable_count = 0

    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # pick ROI
        rect = roi_from_mediapipe(frame, hands_mp) if hands_mp and hands_mp[0] else None
        if rect is None:
            rect = roi_from_skin(frame)
        if rect is None:
            disp = frame.copy()
            cv2.putText(disp, "No hand detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("fingers", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        x1, y1, x2, y2 = rect
        roi = frame[y1:y2, x1:x2]

        # ---- NEW: convert ROI to dataset style (binary mask, centered, square) ----
        ds_img = to_dataset_style(roi)  # 3-channel grayscale, IMG_SIZE x IMG_SIZE

        # Preview the model input (so you can visually confirm it matches your dataset)
        if SHOW_PROC:
            cv2.imshow("dataset_style_roi", ds_img)

        # Model input tensor
        x = to_tensor_norm(ds_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(x)
            prob = F.softmax(logits, dim=1).cpu().numpy()[0]

        # EMA smoothing of probabilities
        if ema is None:
            ema = prob
        else:
            ema = EMA_ALPHA * prob + (1 - EMA_ALPHA) * ema

        pred_idx = int(np.argmax(ema))
        conf = float(ema[pred_idx])

        if last_label == pred_idx and conf >= MIN_CONF:
            stable_count += 1
        else:
            stable_count = 1
            last_label = pred_idx

        label_txt = LABELS[pred_idx]
        color = (0, 255, 0) if conf >= MIN_CONF and stable_count >= (SMOOTH_WIN // 2) else (0, 165, 255)

        # Draw UI on original frame (also draw the ROI rectangle we cropped)
        disp = frame.copy()
        cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
        cv2.putText(disp, f"Pred: {label_txt}  conf={conf:.2f}", (x1, max(30, y1-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("fingers", disp)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
