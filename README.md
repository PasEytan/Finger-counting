# Real-Time Finger Counting AI

**Author:** Eytan Weill  
**Course:** 420-SNT-MS Object-Oriented Programming (Fall 2025)  
**Instructor:** Robert D. Vincent  

## About This Project
This project explores computer vision and deep learning to solve the problem of counting fingers in real-time using a webcam feed. By leveraging OpenCV for image processing and TensorFlow/Keras for image classification, the application detects a hand within a specific region of interest (ROI), processes the visual data, and uses a Convolutional Neural Network (CNN) to predict the number of fingers held up (0-5).

This project demonstrates the integration of:
* **Object-Oriented Programming:** Modular design using classes.
* **Computer Vision:** Background subtraction and contour detection.
* **Deep Learning:** Training and deployment of a CNN.

---

## Installation & Requirements

To run this project, you must have Python 3 installed.

### 1. Dependencies
The project relies on several external libraries. You can install them using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```


## References
- [Medium article](https://medium.com/@guptakgk14/creating-a-custom-cnn-for-real-time-finger-detection-947222db71b0) for training by Gaurang Gupta
- [Kaggle Fingers Dataset](https://www.kaggle.com/datasets/koryakinp/fingers) 




