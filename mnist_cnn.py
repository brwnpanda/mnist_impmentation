"""
MNIST Handwritten Digit Recognition using CNN
Built with Keras/TensorFlow to classify MNIST handwritten digits.
Achieves over 98% validation accuracy.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os


def create_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Create a Convolutional Neural Network model for MNIST digit classification.
    
    Architecture:
    - Conv2D layer with 32 filters, 3x3 kernel
    - MaxPooling2D layer
    - Conv2D layer with 64 filters, 3x3 kernel
    - MaxPooling2D layer
    - Flatten layer
    - Dense layer with 128 units
    - Dropout for regularization
    - Dense output layer with 10 units (one per digit)
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes (10 for digits 0-9)
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # First convolutional block
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Second convolutional block
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def load_and_preprocess_data():
    """
    Load and preprocess the MNIST dataset.
    
    Returns:
        Tuple of (x_train, y_train, x_test, y_test)
    """
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    # Reshape to add channel dimension (required for Conv2D)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    return x_train, y_train, x_test, y_test


def train_model(model, x_train, y_train, x_test, y_test, epochs=15, batch_size=128):
    """
    Train the CNN model.
    
    Args:
        model: Keras model to train
        x_train: Training images
        y_train: Training labels
        x_test: Test images
        y_test: Test labels
        epochs: Number of training epochs
        batch_size: Batch size for training
    
    Returns:
        Training history object
    """
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train the model
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        verbose=1
    )
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    return history


def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training and validation accuracy/loss curves.
    
    Args:
        history: Training history object
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nTraining history plot saved to {save_path}")
    plt.close()


def visualize_predictions(model, x_test, y_test, num_samples=10, save_path='predictions.png'):
    """
    Visualize model predictions on test samples.
    
    Args:
        model: Trained Keras model
        x_test: Test images
        y_test: Test labels
        num_samples: Number of samples to visualize
        save_path: Path to save the plot
    """
    # Get predictions for first num_samples
    predictions = model.predict(x_test[:num_samples])
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Create visualization
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(num_samples):
        axes[i].imshow(x_test[i].squeeze(), cmap='gray')
        axes[i].set_title(f"True: {y_test[i]}\nPred: {predicted_classes[i]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Predictions visualization saved to {save_path}")
    plt.close()


def main():
    """
    Main function to train and evaluate the MNIST CNN model.
    """
    print("=" * 60)
    print("MNIST Handwritten Digit Recognition - CNN Implementation")
    print("=" * 60)
    
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    
    # Create model
    print("\n2. Creating CNN model...")
    model = create_cnn_model()
    model.summary()
    
    # Train model
    print("\n3. Training model...")
    history = train_model(model, x_train, y_train, x_test, y_test, epochs=15)
    
    # Get final validation accuracy
    final_val_accuracy = history.history['val_accuracy'][-1]
    print(f"\n{'=' * 60}")
    print(f"Final Validation Accuracy: {final_val_accuracy:.4f} ({final_val_accuracy*100:.2f}%)")
    if final_val_accuracy > 0.98:
        print("✓ Target achieved: >98% validation accuracy!")
    print(f"{'=' * 60}")
    
    # Plot training history
    print("\n4. Generating visualizations...")
    plot_training_history(history)
    
    # Visualize predictions
    visualize_predictions(model, x_test, y_test)
    
    # Save model
    model_path = 'mnist_cnn_model.h5'
    model.save(model_path)
    print(f"\n5. Model saved to {model_path}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
