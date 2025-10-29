"""
Example script demonstrating how to use the trained MNIST CNN model.
Shows how to load the model and make predictions on new data.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


def load_trained_model(model_path='mnist_cnn_model.h5'):
    """
    Load the trained MNIST CNN model.
    
    Args:
        model_path: Path to the saved model file
    
    Returns:
        Loaded Keras model
    """
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    print("Model loaded successfully!")
    return model


def predict_digit(model, image):
    """
    Predict the digit in a given image.
    
    Args:
        model: Trained Keras model
        image: 28x28 grayscale image (numpy array)
    
    Returns:
        Tuple of (predicted_digit, confidence)
    """
    # Ensure image has correct shape
    if image.shape == (28, 28):
        image = np.expand_dims(image, -1)  # Add channel dimension
    
    if len(image.shape) == 3:
        image = np.expand_dims(image, 0)  # Add batch dimension
    
    # Make prediction
    prediction = model.predict(image, verbose=0)
    predicted_digit = np.argmax(prediction[0])
    confidence = prediction[0][predicted_digit]
    
    return predicted_digit, confidence


def create_sample_digit(digit, noise_level=0.05):
    """
    Create a simple synthetic digit image for demonstration.
    
    Args:
        digit: Digit to create (0-9)
        noise_level: Amount of noise to add
    
    Returns:
        28x28 numpy array
    """
    img = np.zeros((28, 28))
    center_x, center_y = 14, 14
    
    if digit == 0:  # Circle
        for i in range(28):
            for j in range(28):
                dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                if 6 < dist < 10:
                    img[i, j] = 0.8
    
    elif digit == 1:  # Vertical line
        for i in range(6, 24):
            for j in range(12, 16):
                img[i, j] = 0.8
    
    elif digit == 7:  # Top line with diagonal
        for i in range(28):
            for j in range(28):
                if i < 8 and 8 < j < 20:
                    img[i, j] = 0.8
                if abs(2*i + j - 36) < 2:
                    img[i, j] = 0.8
    
    else:  # For other digits, create a simple pattern
        for i in range(8, 22):
            for j in range(8, 22):
                if (i + j) % 3 == digit % 3:
                    img[i, j] = 0.6
    
    # Add noise
    img += np.random.randn(28, 28) * noise_level
    img = np.clip(img, 0, 1)
    
    return img.astype('float32')


def demonstrate_predictions(model, num_samples=5):
    """
    Demonstrate model predictions on sample images.
    
    Args:
        model: Trained Keras model
        num_samples: Number of samples to demonstrate
    """
    print(f"\nDemonstrating predictions on {num_samples} sample digits:")
    print("-" * 60)
    
    # Create sample digits
    test_digits = [0, 1, 2, 7, 9][:num_samples]
    
    for true_digit in test_digits:
        # Create sample image
        image = create_sample_digit(true_digit)
        
        # Make prediction
        predicted_digit, confidence = predict_digit(model, image)
        
        # Display result
        match = "✓" if predicted_digit == true_digit else "✗"
        print(f"{match} True: {true_digit}, Predicted: {predicted_digit}, Confidence: {confidence:.4f}")
    
    print("-" * 60)


def visualize_prediction(model, image, true_label=None, save_path='example_prediction.png'):
    """
    Visualize a single prediction with the image and prediction probabilities.
    
    Args:
        model: Trained Keras model
        image: 28x28 image to predict
        true_label: Optional true label for the image
        save_path: Path to save the visualization
    """
    # Make prediction
    predicted_digit, confidence = predict_digit(model, image)
    
    # Get all class probabilities
    if len(image.shape) == 2:
        image_input = np.expand_dims(np.expand_dims(image, -1), 0)
    else:
        image_input = np.expand_dims(image, 0)
    
    probabilities = model.predict(image_input, verbose=0)[0]
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Plot image
    ax1.imshow(image.squeeze(), cmap='gray')
    title = f"Predicted: {predicted_digit} ({confidence:.2%})"
    if true_label is not None:
        title += f"\nTrue: {true_label}"
    ax1.set_title(title, fontsize=12)
    ax1.axis('off')
    
    # Plot probabilities
    ax2.bar(range(10), probabilities)
    ax2.set_xlabel('Digit', fontsize=11)
    ax2.set_ylabel('Probability', fontsize=11)
    ax2.set_title('Class Probabilities', fontsize=12)
    ax2.set_xticks(range(10))
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {save_path}")
    plt.close()


def main():
    """
    Main function demonstrating model usage.
    """
    print("=" * 60)
    print("MNIST CNN Model - Usage Example")
    print("=" * 60)
    
    # Load the trained model
    try:
        model = load_trained_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please train the model first by running: python mnist_cnn.py")
        return
    
    # Display model summary
    print("\nModel Summary:")
    print("-" * 60)
    model.summary()
    print("-" * 60)
    
    # Demonstrate predictions
    demonstrate_predictions(model, num_samples=5)
    
    # Create and visualize a sample prediction
    print("\nCreating visualization for a sample digit...")
    sample_image = create_sample_digit(digit=7)
    visualize_prediction(model, sample_image, true_label=7)
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    print("\nYou can now use this model to predict digits in your own images.")
    print("Simply load the model and call predict_digit(model, your_image)")


if __name__ == "__main__":
    main()
