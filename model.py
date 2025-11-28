import tensorflow as tf
import numpy as np
import cv2


# Bouderies for the model
ROI_TOP = 20
ROI_BOTTOM = 300
ROI_RIGHT = 300
ROI_LEFT = 600
IMAGE_SIZE = 128
ACCUMULATED_WEIGHT = 0.5

# Model itself
MODEL_PATH_REL = './realtime_fingers_detection.keras'

class HandSegmenter:
    """
    Performing the Mask. 
    Removes the background by taking an original background
    frame and substracting every new frame with it.
    """
    def __init__(self, accum_weight=0.5):
        self.background = None
        self.accum_weight = accum_weight

    def update_background(self, frame):
        """Assigning background for substraction."""
        if self.background is None:
            self.background = frame.copy().astype("float")
            return
        cv2.accumulateWeighted(frame, self.background, self.accum_weight)

    def segment(self, frame, threshold=25):
        """
        Substracting the old saved background to evert new frame and makes 
        an countours every new blob that should be part of the hand. 
        """
        if self.background is None:
            return None

        # Calculate absolute difference between background and current frame
        diff = cv2.absdiff(self.background.astype("uint8"), frame)
        _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None

        # Assume largest contour is the hand
        hand_segment = max(contours, key=cv2.contourArea)

        # Some cv2 filtering
        if cv2.contourArea(hand_segment) < 1000:
            return None

        return (thresholded, hand_segment)
    
    def reset(self):
        """Resets the background copy."""
        self.background = None


class FingerModel:
    """
    Using the model to use on the processed area to 
    make a prediction. 
    """
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.image_size = IMAGE_SIZE

    def preprocess(self, mask_crop):
        """Prepares the mask for the neural network."""
        # 1. Resize
        img = cv2.resize(mask_crop, (self.image_size, self.image_size))
        
        # 2. cv2 cleanup of bad edges and resizing 
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        # 3. Transform all pixel values from a 0-255 range to 0-1 float range
        img = img / 255.0
        
        # 4. Reshape for model input (BatchSize, Height, Width, Channels)
        img = np.reshape(img, (1, self.image_size, self.image_size, 1))
        return img

    def predict(self, roi_image):
        """
        Takes a raw ROI image, checks pixel density, and returns prediction.
        Returns: class_index (int) or -1 (if empty/invalid)
        """
        # Preprocess
        processed_img = self.preprocess(roi_image)
        
  
        # Counting the quantity of white pixels
        unique, counts = np.unique(processed_img > 0, return_counts=True)
        
        # Determine pixel count
        pixel_count = 0
        if len(counts) > 1:
            pixel_count = counts[1]

        # Threshhold for what reasonably constitues as a hand
        if pixel_count < 2000:
            return -1

        output = self.model(tf.constant(processed_img, dtype=float))
        return np.argmax(output)


class FingerCounterApp:
    """
    Runs the main program and manages all ui elements.  
    """
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.segmenter = HandSegmenter(accum_weight=ACCUMULATED_WEIGHT)
        self.predictor = FingerModel(model_path=MODEL_PATH_REL)
        
        self.num_frames = 0
        self.is_running = True

    def draw_ui(self, frame, prediction, hand_contour):
        """Handles all drawing (text, boxes, contours) on the frame."""
        # Draw the box being cropped 
        cv2.rectangle(frame, (ROI_LEFT, ROI_TOP), (ROI_RIGHT, ROI_BOTTOM), (0, 255, 0), 2)

        # Drawing necessary text 
        if prediction is not None:
            text = str(prediction) if prediction != -1 else "No Hand"
            color = (0, 255, 0) if prediction != -1 else (0, 0, 255)
            cv2.putText(frame, text, (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        else:
            cv2.putText(frame, "WAIT! CALIBRATING...", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # Draw Hand Contour
        if hand_contour is not None:
             cv2.drawContours(frame, [hand_contour + (ROI_RIGHT, ROI_TOP)], -1, (255, 0, 0), 1)

    def run(self):
        # While loop of the camera feed and AI predictions. 
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            frame_copy = frame.copy()

            # 1. Extract region of interest 
            roi = frame[ROI_TOP:ROI_BOTTOM, ROI_RIGHT:ROI_LEFT]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)

            prediction = None
            hand_contour = None
            thresholded_view = np.zeros((roi.shape[0], roi.shape[1]), dtype='uint8')

            # 2. Calibrate or predict
            if self.num_frames < 60:
                self.segmenter.update_background(gray_roi)
            else:
                # Attempt to find hand
                hand_data = self.segmenter.segment(gray_roi)
                
                if hand_data is not None:
                    thresholded_view, hand_contour = hand_data
                    
                    # Predict using the model
                    prediction = self.predictor.predict(thresholded_view)
                else:
                    # No hand detected
                    prediction = -1 

            # 3. Showing webcam feed
            cv2.imshow("What the AI sees", thresholded_view)
            self.draw_ui(frame_copy, prediction, hand_contour)
            cv2.imshow("FingerCount", frame_copy)
            
            self.num_frames += 1

            # Keyboard inputs
            self.keyinput()

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

    def keyinput(self):
        # Exciting program
        k = cv2.waitKey(1) & 0xFF
        if k == 27: # ESC
            self.is_running = False

        # Restarting the background calibration 
        elif k == ord('r'):
            self.segmenter.reset()
            self.num_frames = 0
            print("\nRecalibrating...")

if __name__ == "__main__":
    app = FingerCounterApp()
    app.run()