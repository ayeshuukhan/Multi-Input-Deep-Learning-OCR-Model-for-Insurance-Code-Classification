
## Project Overview

This project focuses on developing an Optical Character Recognition (OCR)-based deep learning model to classify ID codes extracted from scanned insurance documents.

The model uses both:

* Image data (64×64 grayscale)
* Type information (one-hot encoded vector)

to predict the correct category of insurance codes.


## Objective

To build a model that can automatically identify and classify insurance-related codes from document images using deep learning techniques.


## Model Architecture

The model is implemented using PyTorch and consists of three main components:

### 1. Image Processing Layer (`image_layer`)

* Conv2D (1 → 16 channels, kernel size = 3, padding = 1)
* MaxPooling (2×2)
* ReLU Activation
* Flatten Layer
* Fully Connected Layer (to 128 features)

### 2. Type Processing Layer (`type_layer`)

* Fully Connected Layer (5 → 10)
* ReLU Activation

### 3. Classifier

* Concatenation of image + type features
* Fully Connected Layer (128 + 10 → 64)
* ReLU Activation
* Output Layer (64 → 2 classes)
* 

## Technologies Used

* Python
* PyTorch
* NumPy
* Matplotlib
* Pickle


## Dataset

* Dataset is loaded from: `ocr_insurance_dataset.pkl`
* Each sample contains:

  * Image (64×64 grayscale)
  * Type vector (one-hot encoded)
  * Label (category of code)


## ⚙️ Training Details

* Optimizer: Adam
* Learning Rate: 0.001
* Loss Function: CrossEntropyLoss
* Epochs: 10
* Batch Size: 10


## Model Evaluation

The model is evaluated using:

* Training Loss
* Accuracy on training data

Example output:

```
Epoch 1, Loss: 0.65, Accuracy: 70%
Epoch 10, Loss: 0.12, Accuracy: 95%
```


## Features

* Multi-input model (image + metadata)
* Simple CNN-based architecture
* Visualization of dataset samples
* Custom prediction function for testing
* 
