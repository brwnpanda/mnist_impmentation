"""
Test script to verify the trained MNIST CNN model works correctly.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np


def test_model_loading():
    """Test that the saved model can be loaded."""
    print("Testing model loading...")
    try:
        model = keras.models.load_model('mnist_cnn_model.h5')
        print("✓ Model loaded successfully")
        return model
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return None


def test_model_architecture(model):
    """Test that the model has the expected architecture."""
    print("\nTesting model architecture...")
    
    # Check input shape
    expected_input_shape = (None, 28, 28, 1)
    actual_input_shape = model.input_shape
    assert actual_input_shape == expected_input_shape, f"Expected input shape {expected_input_shape}, got {actual_input_shape}"
    print(f"✓ Input shape correct: {actual_input_shape}")
    
    # Check output shape
    expected_output_shape = (None, 10)
    actual_output_shape = model.output_shape
    assert actual_output_shape == expected_output_shape, f"Expected output shape {expected_output_shape}, got {actual_output_shape}"
    print(f"✓ Output shape correct: {actual_output_shape}")
    
    # Check number of layers
    num_layers = len(model.layers)
    print(f"✓ Model has {num_layers} layers")
    
    # Verify key layer types
    layer_types = [type(layer).__name__ for layer in model.layers]
    assert 'Conv2D' in layer_types, "Model should have Conv2D layers"
    assert 'MaxPooling2D' in layer_types, "Model should have MaxPooling2D layers"
    assert 'Dense' in layer_types, "Model should have Dense layers"
    assert 'Dropout' in layer_types, "Model should have Dropout layer"
    print("✓ Model has required layer types: Conv2D, MaxPooling2D, Dense, Dropout")


def test_model_prediction(model):
    """Test that the model can make predictions."""
    print("\nTesting model predictions...")
    
    # Create a dummy input
    dummy_input = np.random.rand(1, 28, 28, 1).astype('float32')
    
    # Make prediction
    try:
        prediction = model.predict(dummy_input, verbose=0)
        print(f"✓ Model can make predictions")
        
        # Check prediction shape
        assert prediction.shape == (1, 10), f"Expected prediction shape (1, 10), got {prediction.shape}"
        print(f"✓ Prediction shape correct: {prediction.shape}")
        
        # Check prediction is a valid probability distribution
        assert np.abs(np.sum(prediction) - 1.0) < 1e-5, "Prediction should sum to 1"
        print(f"✓ Prediction is a valid probability distribution (sum={np.sum(prediction):.6f})")
        
        # Get predicted class
        predicted_class = np.argmax(prediction)
        print(f"✓ Predicted class: {predicted_class} (confidence: {prediction[0][predicted_class]:.4f})")
        
        return True
    except Exception as e:
        print(f"✗ Failed to make prediction: {e}")
        return False


def test_batch_prediction(model):
    """Test that the model can handle batch predictions."""
    print("\nTesting batch predictions...")
    
    # Create a batch of dummy inputs
    batch_size = 10
    dummy_batch = np.random.rand(batch_size, 28, 28, 1).astype('float32')
    
    try:
        predictions = model.predict(dummy_batch, verbose=0)
        print(f"✓ Model can handle batch predictions")
        
        # Check predictions shape
        assert predictions.shape == (batch_size, 10), f"Expected shape ({batch_size}, 10), got {predictions.shape}"
        print(f"✓ Batch predictions shape correct: {predictions.shape}")
        
        # Check all predictions are valid probability distributions
        sums = np.sum(predictions, axis=1)
        assert np.allclose(sums, 1.0), "All predictions should sum to 1"
        print(f"✓ All batch predictions are valid probability distributions")
        
        return True
    except Exception as e:
        print(f"✗ Failed batch prediction: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("MNIST CNN Model Tests")
    print("=" * 60)
    
    # Test model loading
    model = test_model_loading()
    if model is None:
        print("\nTests failed: Could not load model")
        return
    
    # Test architecture
    try:
        test_model_architecture(model)
    except AssertionError as e:
        print(f"\n✗ Architecture test failed: {e}")
        return
    
    # Test predictions
    if not test_model_prediction(model):
        print("\nTests failed: Prediction test failed")
        return
    
    # Test batch predictions
    if not test_batch_prediction(model):
        print("\nTests failed: Batch prediction test failed")
        return
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
