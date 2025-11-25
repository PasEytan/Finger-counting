import tensorflow as tf
import numpy as np
import cv2
import os

# --- Global Variables ---
background = None
accumulated_weight = 0.5
num_frames = 0

# ROI Coordinates (Green Box)
roi_top = 20
roi_bottom = 300
roi_right = 300
roi_left = 600

# --- Load Model Robustly ---
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'C:/Users/weill/OneDrive/Documents/github/Finger-counting/Finger-counting/realtime_fingers_detection.keras')

try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

IMAGE_SIZE = 128

class FingerClassifier(object):
    def __init__(self, model_object):
        self.detect = model_object

    def get_classification(self, img):
        img = img.reshape(1, *img.shape)
        img = tf.constant(img, dtype=float)
        
        # Check pixel count
        unique, counts = np.unique(img, return_counts=True)
        
        # DEBUG: Print what the model sees
        if len(counts) > 1:
            print(f"Hand Pixels: {counts[1]}", end="/r") # Print count to track issues
        else:
            print("Hand Pixels: 0 (Image is empty)", end="/r")

        # If image is empty or hand is too small
        if (len(counts) <= 1 or counts[1] < 2000): 
            return -1
        
        output = self.detect(img)
        return np.argmax(output)

obj = FingerClassifier(model)

# --- Helper Functions ---
def calc_accum_avg(frame, accumulated_weight):
    global background
    if background is None:
        background = frame.copy().astype("float")
        return None
    cv2.accumulateWeighted(frame, background, accumulated_weight)

def segment(frame, threshold=25):
    global background
    
    diff = cv2.absdiff(background.astype("uint8"), frame)
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None
    else:
        # Get the largest contour (the hand)
        hand_segment = max(contours, key=cv2.contourArea)
        
        # Filter out small noise (adjust 1000 if needed)
        if cv2.contourArea(hand_segment) < 1000:
            return None
            
        return (thresholded, hand_segment)

def prepare_for_model(mask_crop):
    # 1. Resize
    img = cv2.resize(mask_crop, (IMAGE_SIZE, IMAGE_SIZE))
    
    # 2. FORCE BINARY (The Fix)
    # Resizing creates grey edges (e.g. 150, 200). We threshold again to force 0 or 255.
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # 3. Normalize to 0.0 and 1.0
    img = img / 255.0
    
    # 4. Reshape
    img = np.reshape(img, (IMAGE_SIZE, IMAGE_SIZE, 1))
    return img

# --- Main Loop ---
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()

    # Extract ROI
    roi = frame[roi_top:roi_bottom, roi_right:roi_left]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Calibration Phase
    if num_frames < 60:
        calc_accum_avg(gray, accumulated_weight)
        cv2.putText(frame_copy, "WAIT! CALIBRATING...", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    else:
        # Hand Detection Phase
        hand = segment(gray)

        if hand is not None:
            thresholded, hand_segment = hand
            
            # Draw the hand contour (Visual feedback)
            cv2.drawContours(frame_copy, [hand_segment + (roi_right, roi_top)], -1, (255, 0, 0), 1)
            
            # Show the mask
            cv2.imshow("What the AI Sees", thresholded)
            
            # Predict
            model_input = prepare_for_model(thresholded)
            prediction = obj.get_classification(model_input)
            
            # Display Prediction
            text = str(prediction) if prediction != -1 else "No Hand"
            color = (0, 255, 0) if prediction != -1 else (0, 0, 255)
            
            cv2.putText(frame_copy, text, (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        else:
            cv2.putText(frame_copy, "No Hand Detected", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            # Clear the mask window if no hand
            cv2.imshow("What the AI Sees", np.zeros((roi.shape[0], roi.shape[1]), dtype='uint8'))

    # Draw ROI Box
    cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 2)
    
    num_frames += 1
    cv2.imshow("Finger Count", frame_copy)

    # Controls
    k = cv2.waitKey(1) & 0xFF
    if k == 27: # ESC to quit
        break
    elif k == ord('r'): # Press 'r' to recalibrate background
        background = None
        num_frames = 0
        print("/nRecalibrating...")

cam.release()
cv2.destroyAllWindows()