# Fashion-MNIST Image Classification (TensorFlow & CNN)

A Python-based machine learning project that builds, trains, and evaluates a Convolutional Neural Network (CNN) on the Fashion-MNIST dataset using TensorFlow and Keras. The model classifies 28x28 grayscale images of clothing items into 10 distinct categories, plots training history, and provides detailed error analysis alongside binary metrics for specific classes.

---

## Project Overview

The workflow covers a complete deep learning pipeline from raw data processing to diagnostic evaluation.

* Dataset Loading: Downloads and splits the Fashion-MNIST dataset into training and testing partitions
* Data Preprocessing: Normalizes pixel values to a [0, 1] range, adds a grayscale channel dimension, and applies one-hot encoding to target labels
* Network Architecture: Features a multi-layer CNN with convolutional and max pooling operations followed by dense layers to extract spatial hierarchies in clothing patterns
* Training Diagnostic: Visualizes training and validation loss curves over epochs to identify potential overfitting and determine the best performing epoch
* Error Analysis: Identifies and renders incorrect predictions to visually evaluate where the classifier struggles
* Target Class Evaluation: Extracts binary-style metrics like Sensitivity, Precision, and Accuracy for a single chosen category to assess performance on specific items

---

## Technical Features

The code demonstrates essential deep learning concepts and diagnostics:

* Layered Feature Extraction: Leverages two consecutive 2D convolutional layers with ReLU activations followed by max pooling to downsample spatial dimensions and capture features
* Categorical Crossentropy: Compiles the network with Adam optimization and categorical crossentropy loss to optimize the multi-class classification task
* Squeeze and Visual Plotting: Drops redundant single-dimensions using NumPy and utilizes Matplotlib to render grayscale clothing images with clear true and predicted titles
* Custom One-Hot Metrics: Implements boolean array masking to isolate True Positives, True Negatives, False Positives, and False Negatives directly from multi-class outputs

---

## Key Structures

* `plot_incorrect_evaluations(model, test_data, class_names, max_to_show)`: Predicts class probabilities, identifies misclassified clothing items, and displays a grid of incorrect predictions with their true and predicted labels
* `metrics_from_onehot(y_true_onehot, y_pred_ids, positive_class)`: Computes binary classification metrics including True Positives, True Negatives, False Positives, and False Negatives for a selected clothing category
* CNN Architecture Pipeline: Constructs the Sequential model with Input, Conv2D, MaxPooling2D, Flatten, and Dense layers to map image inputs to class probability vectors
