import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


# ---- match your training model ----
NUM_CLASSES = 6  # 0..5
IMG_SIZE = 128

class FingerCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

# same transforms as training
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def draw_label(frame, label, conf):
    text = f"Fingers: {label}  ({conf*100:.1f}%)"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 2, cv2.LINE_AA)

def main(weights_path="finger_counter_torch.pth", cam_index=0, use_cuda=True):
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    model = FingerCNN().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(cam_index)  # 0 = default webcam
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    with torch.no_grad():
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            # OpenCV gives BGR; convert to RGB for the model
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # preprocess -> tensor [1,3,128,128]
            inp = preprocess(frame_rgb).unsqueeze(0).to(device)

            logits = model(inp)
            probs = torch.softmax(logits, dim=1)[0]  # [NUM_CLASSES]
            pred  = int(torch.argmax(probs).item())
            conf  = float(probs[pred].item())

            # draw on the original BGR frame
            draw_label(frame_bgr, pred, conf)
            cv2.imshow("Finger Counter (press q to quit)", frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Tip: If you see a black camera window, try main(..., cam_index=1) or another index.
    main()
