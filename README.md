# CIFAR-10 Image Classification using Convolutional Neural Networks (CNN)

## Project Overview

This project focuses on building an image classification model using the **CIFAR-10 dataset**, which consists of 60,000 32x32 pixel color images classified into 10 distinct categories. The aim of this project is to train a **Convolutional Neural Network (CNN)** from scratch to recognize and classify these images with high accuracy.

### Key Features:
- **Deep Learning Framework**: We use **TensorFlow** and **Keras** for building the CNN model.
- **Data Normalization**: Images are normalized to improve model performance.
- **Model Evaluation**: The model is evaluated using accuracy and loss metrics on both the training and test datasets.
- **Visualization**: Includes visualizations of sample images, accuracy, and loss over epochs.
- **Prediction**: The trained model makes predictions on test images, with a comparison of true and predicted labels.

---

## Dataset

The **CIFAR-10 dataset** is a well-known benchmark dataset used in machine learning and computer vision tasks. It contains **10 different classes**:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

The dataset consists of:
- **50,000 training images** and **10,000 test images**.
- Each image is represented in RGB format (3 channels) with a resolution of **32x32 pixels**.

More information about the dataset can be found at [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

---

## Model Architecture

The CNN model used in this project has the following architecture:
1. **Input Layer**: Takes input images of shape (32, 32, 3).
2. **Convolutional Layers**: 
   - 3 convolutional layers with increasing filter sizes (32, 64, 64) to extract features from the images.
   - Each convolutional layer is followed by a **MaxPooling** layer to down-sample the feature maps.
3. **Fully Connected Layers**:
   - A flattening layer converts the 3D feature maps into a 1D vector.
   - A dense layer with 64 units is used, followed by a final dense layer with 10 units corresponding to the 10 classes.
4. **Activation**: 
   - ReLU activation is used for hidden layers, and **Softmax** is applied at the output to generate probabilities for each class.
   
### Model Summary

```plaintext
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 30, 30, 32)        896
max_pooling2d (MaxPooling2D) (None, 15, 15, 32)        0
conv2d_1 (Conv2D)            (None, 13, 13, 64)        18496
max_pooling2d_1 (MaxPooling2D)(None, 6, 6, 64)         0
conv2d_2 (Conv2D)            (None, 4, 4, 64)          36928
flatten (Flatten)            (None, 1024)              0
dense (Dense)                (None, 64)                65600
dense_1 (Dense)              (None, 10)                650
=================================================================
Total params: 122,570
```

# Model Training

The model is trained using the **Adam optimizer** and the **Sparse Categorical Crossentropy** loss function. The loss function is well-suited for multi-class classification tasks, and the Adam optimizer ensures efficient gradient updates.

## Training Parameters:
- **Epochs**: 10 (can be increased for better performance)
- **Batch Size**: 32
- **Metrics**: Accuracy is used as the primary evaluation metric.
- **Validation Data**: Test images are used for validation during training to monitor the model's performance.

---

# Results and Visualizations

## Accuracy and Loss Curves
After training the model for 10 epochs, we generate and display the accuracy and loss curves for both the training and validation datasets. These plots help visualize how well the model is learning over time.

- **Training Accuracy**: Increases as the model learns from the data.
- **Validation Accuracy**: Helps in assessing how well the model generalizes to unseen data.


---

## Prediction Results
The model is also evaluated on the test dataset, and predictions are made on individual images. The predicted label is compared with the true label to visually verify the model’s performance.
