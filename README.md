# Digit Recognition Neural Network

A feedforward neural network implementation built entirely from scratch in TypeScript for MNIST digit classification.

## Overview

This project implements a multi-layer perceptron designed for handwritten digit recognition using the MNIST dataset. The network processes 28×28 pixel images and classifies them into 10 digit categories (0-9) without relying on external machine learning frameworks.

## Architecture

The neural network employs a three-layer architecture optimized for image classification:
- **Input Layer**: 784 neurons (28×28 flattened pixel values)
- **Hidden Layer 1**: 128 neurons with sigmoid activation
- **Hidden Layer 2**: 64 neurons with sigmoid activation  
- **Output Layer**: 10 neurons with softmax activation

## Technical Implementation

- **Activation Functions**: Sigmoid for hidden layers, softmax for output layer
- **Loss Function**: Cross-entropy loss for multi-class classification
- **Weight Initialization**: Xavier initialization to prevent gradient vanishing
- **Training Algorithm**: Mini-batch gradient descent with backpropagation
- **Batch Size**: 128 samples per training iteration

## Training Results

![Training Loss with Moving Average](Training%20Loss%20with%20Moving%20Average.png)

The training loss curve demonstrates convergence over multiple iterations, with the moving average showing consistent improvement in model performance. The decreasing cross-entropy loss indicates successful learning and optimization of network parameters.
