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

But before that, you would need to get a dataset. I used Kaggle's "finger" dataset[[2](#References)] to train my model. In the ```train.ipynb``` file, just be sure to have your training data and testing data sgemented into the right folders and fix the formatting of your dataset to your desired resolution. 

> **NOTE:** Some of the libraries do not play nice with online notebooks, so if you plan on using a service like Google Colab or Databricks, there might be so tweaks needed to get the training to work. 

### Utilization

After installing all the dependencies, running the program with the pretrained file is as easy as having a webcam already connected to your device and running the command:
```bash
python model.py
```

This operation will use the already existing pretrained model, ```realtime_fingers_detection.keras```. 
 
### Pretained Model

In the repository, you will see a file called ```realtime_fingers_detection.keras```. This is the model that is used by default in the ```model.py``` file. It was trained for 30 Epochs and is set to be about 99.97 accurate in its predictions.

![alt text](Untitled.png)

## References
1. [Medium article](https://medium.com/@guptakgk14/creating-a-custom-cnn-for-real-time-finger-detection-947222db71b0) for training by Gaurang Gupta
2. [Kaggle Fingers Dataset](https://www.kaggle.com/datasets/koryakinp/fingers) 




