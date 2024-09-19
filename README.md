# Deep_Learning
Delving into the realm of Deep learning techniques of machine learning.

# [Convolutional Neural Network (CNN) for Digit Recognition with K-Fold Cross-Validation](Digit Recognizer CNN K-Fold.ipynb)

## Model Overview
This model is a **Convolutional Neural Network (CNN)** designed for the **Digit Recognition** task using the **MNIST dataset**. It uses **K-Fold Cross-Validation** to ensure generalization and stability of the model's performance across different subsets of the data.

### Key Features:
- **Input**: Grayscale images of size 28x28, with pixel values normalized between 0 and 1.
- **Architecture**:
  - Input layer for 28x28 grayscale images.
  - 2 Convolutional layers (`Conv2D`) with ReLU activation, followed by a MaxPooling layer to reduce spatial dimensions.
  - Dropout layers to prevent overfitting.
  - Fully connected (`Dense`) layer for classification.
  - Output layer with 10 neurons and Softmax activation to classify digits 0-9.
- **Optimizer**: Adam optimizer for efficient weight updates.
- **Loss Function**: Categorical Crossentropy, suitable for multi-class classification.
- **Evaluation**: 
  - Accuracy is evaluated using K-Fold Cross-Validation with 5 folds.
  - Model performance is saved using the `ModelCheckpoint` callback, selecting the best model based on validation accuracy.

## Callbacks Used:
- **ModelCheckpoint**: Saves the best model during training.
- **ReduceLROnPlateau**: Reduces the learning rate if validation loss plateaus.
- **EarlyStopping**: Stops training when validation loss does not improve for a set number of epochs.
- **LearningRateScheduler**: Adjusts learning rate based on an exponential decay schedule.

## Training and Validation:
- The model is trained for 50 epochs on each fold, with a batch size of 64.
- Validation accuracy is tracked across folds, and predictions are made on test data using the best-performing model.

## Results:
- After training, the model achieves an average validation accuracy across the folds.
- Predictions on test data are submitted as a CSV file with the predicted digit labels.
