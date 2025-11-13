import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp  # Added MediaPipe

# --- MediaPipe Hand Initialization ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
# -------------------------------------

# This loads everything: architecture, weights, and optimizer state
model = tf.keras.models.load_model('./realtime_fingers_detection.keras')
IMAGE_SIZE = 128

class FingerClassifier(object):
    def __init__(self, model_object):
        self.detect = model_object

    def get_classification(self, img):
        img = img.reshape(1, *img.shape)
        img = tf.constant(img, dtype=float)
        unique, counts = np.unique(img, return_counts=True)
        # When number of white pixels is less than 1200 we return -1
        if (len(counts) <= 1 or counts[1] < 1200):
            return -1
        
        output = self.detect(img)
        return np.argmax(output)

obj = FingerClassifier(model)

def process_image(img, thresh_low=40, thresh_high=215):
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # This thresholding type (BINARY) is correct for a
    # LIGHT hand against a DARK background.
    _, img = cv2.threshold(img, thresh_low, thresh_high, cv2.THRESH_BINARY)
    
    im_floodfill = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    im_floodfill_inv = im_floodfill# cv2.bitwise_not(im_floodfill)
    img = img | im_floodfill_inv    
    
    img = img/255 
    img = np.reshape(img, (IMAGE_SIZE, IMAGE_SIZE, 1))
    return img

def process_stream_image(img):
    # Creating a border around the img to make the input similar to the training images
    img = cv2.copyMakeBorder(img.copy(), 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    # --- THIS IS THE FIX ---
    # By changing thresh_high to 255, the hand becomes 1.0 (255/255)
    # instead of 0.98 (250/255), which fixes the next line.
    img = process_image(img, thresh_low=90, thresh_high=170) # <-- Changed 250 to 255
    # -----------------------
    
    # Now, this line correctly keeps the hand (1.0) and sets 
    # everything else (< 1.0) to 0.
    
    return img

# --- Main Loop (Unchanged) ---
rval = True
cam = cv2.VideoCapture(0)

while rval:
    rval, img = cam.read()
    if not rval:
        break
    img = cv2.flip(img, 1)
    display_img = img.copy()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        h, w, _ = img.shape
        x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
        
        x_min = int(min(x_coords))
        x_max = int(max(x_coords))
        y_min = int(min(y_coords))
        y_max = int(max(y_coords))
        
        padding = 30
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        cv2.rectangle(display_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        cropped = img_gray[y_min:y_max, x_min:x_max]
        
        if cropped.size > 0:
            cropped_mask = process_stream_image(cropped)
            cv2.imshow('Mask', cropped_mask)
            fingers = obj.get_classification(cropped_mask)
            print(fingers, end=' ', flush=True)
        else:
            print("-1", end=' ', flush=True)
            
    else:
        print("-1", end=' ', flush=True)
        cv2.imshow('Mask', np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8))

    cv2.imshow('Original', display_img)
    
    if(cv2.waitKey(25) & 0xFF == 27):
        break

cam.release()
cv2.destroyAllWindows()