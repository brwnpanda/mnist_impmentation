# MNIST Handwritten Digit Recognition

A deep learning project implementing a Convolutional Neural Network (CNN) using Keras/TensorFlow to classify MNIST handwritten digits with over 98% validation accuracy.

## Project Overview

This project demonstrates a solid understanding of:
- CNN architecture and design principles
- Key layers: Conv2D, MaxPooling2D, Dense, Dropout
- The complete training process from data loading to model evaluation
- Achieving high accuracy (>98%) on validation data

## Model Architecture

The CNN consists of:
1. **First Convolutional Block**
   - Conv2D layer (32 filters, 3x3 kernel, ReLU activation)
   - MaxPooling2D layer (2x2 pool size)

2. **Second Convolutional Block**
   - Conv2D layer (64 filters, 3x3 kernel, ReLU activation)
   - MaxPooling2D layer (2x2 pool size)

3. **Dense Layers**
   - Flatten layer
   - Dense layer (128 units, ReLU activation)
   - Dropout layer (0.5 rate for regularization)
   - Dense output layer (10 units, Softmax activation)

## Requirements

- Python 3.8+
- TensorFlow 2.13+
- NumPy
- Matplotlib

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Run the training script:
```bash
python mnist_cnn.py
```

This will:
1. Load and preprocess the MNIST dataset
2. Create the CNN model
3. Train the model for 15 epochs
4. Display training progress and final accuracy
5. Generate visualization plots
6. Save the trained model

### Output Files

After training, the following files are generated:
- `mnist_cnn_model.h5` - Saved trained model
- `training_history.png` - Training/validation accuracy and loss curves
- `predictions.png` - Sample predictions on test images

## Results

The model achieves:
- **Validation Accuracy: >98%**
- Fast training time (~2-3 minutes on CPU)
- Robust performance on unseen test data

## Dataset

The MNIST dataset contains:
- 60,000 training images
- 10,000 test images
- 28x28 grayscale images of handwritten digits (0-9)

The dataset is automatically downloaded via Keras when running the script.

## Project Structure

```
mnist_impmentation/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── mnist_cnn.py             # Main training script
├── .gitignore               # Git ignore rules
└── training_history.png     # Generated after training
└── predictions.png          # Generated after training
```

## Key Features

- **Well-structured code** with clear documentation
- **Modular design** with separate functions for each task
- **Visualization tools** for understanding model performance
- **Reproducible results** using fixed random seeds
- **Production-ready** model saving and loading

## License

This is a personal learning project for demonstrating CNN implementation and deep learning fundamentals.
