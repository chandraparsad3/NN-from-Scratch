# Neural Network From Scratch

## Overview

This project implements a simple neural network from scratch using PyTorch for binary classification. The model is trained on a breast cancer dataset to predict whether a tumor is malignant or benign.

The goal of this project is to understand how a neural network works internally, including forward pass, loss calculation, backpropagation, and parameter updates without relying on high-level libraries like `torch.nn`.

---

## Features

* Data preprocessing using `pandas` and `scikit-learn`
* Manual implementation of:

  * Weights and bias initialization
  * Forward propagation
  * Sigmoid activation
  * Binary cross-entropy loss
  * Backpropagation using PyTorch autograd
* Gradient descent optimization
* Model evaluation using accuracy

---

## Dataset

The dataset is loaded from an online source:

* Breast Cancer Detection Dataset
* Contains features computed from digitized images of breast masses
* Target:

  * `M` → Malignant (1)
  * `B` → Benign (0)

---

## Project Structure

* Data loading and cleaning
* Feature scaling using `StandardScaler`
* Label encoding
* Train-test split
* Tensor conversion
* Neural network class (`SimpleNN`)
* Training loop
* Evaluation

---

## Model Details

### Architecture

* Input layer: number of features
* Output layer: 1 neuron
* Activation: Sigmoid

### Forward Pass

```
z = XW + b
y_pred = sigmoid(z)
```

### Loss Function

Binary Cross Entropy:

```
loss = -(y * log(y_pred) + (1 - y) * log(1 - y_pred))
```

---

## Training

* Epochs: 25
* Learning Rate: 0.1
* Optimization: Gradient Descent

Steps:

1. Forward pass
2. Compute loss
3. Backward pass (`loss.backward()`)
4. Update weights manually
5. Reset gradients

---

## Evaluation

After training, the model is tested on unseen data:

* Predictions are thresholded at 0.7
* Accuracy is calculated by comparing predictions with actual labels

---

## How to Run

1. Install dependencies:

```
pip install numpy pandas torch scikit-learn
```

2. Run the script:

```
python neural_network_from_scratch.py
```

---

## Key Learning Points

* How neural networks compute predictions
* Role of activation functions (sigmoid)
* Why loss functions are important
* How gradients help update parameters
* How PyTorch autograd simplifies backpropagation

---

## File Reference

Main implementation file: 

---

## Limitations

* Only a single-layer model (no hidden layers)
* No regularization
* Fixed learning rate
* Threshold manually chosen

---

## Future Improvements

* Add hidden layers (deep neural network)
* Use optimizers like Adam
* Implement mini-batch training
* Add validation set
* Tune hyperparameters

---
