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


## Installation & Requirements

To run this project, you must have Python 3 installed.

### Dependencies
The project relies on several external libraries. You can install them using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```


> **Note:** If you have an Nvidia graphics card and want to test wether or not Cuda (the Nvidia framework for training AI models with code) works, then you can go into the PyTorch website, install the right cuda version for your operating system and run:
```bash
python test.py
```

## How to use

### Training 
If you want to train a model yourself and make some minor tweaks to the actual training, you can do that through the ```training.ipynb``` file.

But before that, you would need to get a dataset [2](#References)


## References
1. [Medium article](https://medium.com/@guptakgk14/creating-a-custom-cnn-for-real-time-finger-detection-947222db71b0) for training by Gaurang Gupta
2. [Kaggle Fingers Dataset](https://www.kaggle.com/datasets/koryakinp/fingers) 




