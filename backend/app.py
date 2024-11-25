from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import base64
import io
import pickle
import logging
from werkzeug.middleware.proxy_fix import ProxyFix
from layers import Layer, Activation, Convolutional, Dense, Reshape, ReLU, Softmax

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
CORS(app)

# Build the network
def build_network():
    network = [
        Convolutional((1, 28, 28), kernel_size=3, depth=8),
        ReLU(),
        Reshape((8, 26, 26), (8 * 26 * 26, 1)),
        Dense(8 * 26 * 26, 128),
        ReLU(),
        Dense(128, 10),
        Softmax()
    ]
    return network

# Load the model parameters
def load_model(filename, network):
    with open(filename, 'rb') as f:
        params = pickle.load(f)
    for layer, layer_params in zip(network, params):
        if hasattr(layer, 'weights') and 'weights' in layer_params:
            layer.weights = layer_params['weights']
        if hasattr(layer, 'biases') and 'biases' in layer_params:
            layer.biases = layer_params['biases']

# Instantiate and load the network
network = build_network()
load_model('cnn_mnist_model3.pkl', network)
logger.info("Model loaded successfully!")

# Preprocess the input image
def preprocess_image(image_data):
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale and resize to 28x28
        image = image.convert('L').resize((28, 28))
        
        # Convert to numpy array and normalize to range [0, 1]
        image_array = 1.0 - (np.array(image, dtype=np.float32) / 255.0)
        
        # Ensure the array has the shape (1, 28, 28)
        image_array = image_array.reshape(1, 28, 28)
        logger.debug(f"Preprocessed image shape: {image_array.shape}")
        
        
        return image_array

    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': True})

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for digit prediction.
    Expects JSON with 'image' field containing base64 encoded image.
    Returns prediction and confidence scores.
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
                
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
                
        # Preprocess image
        input_image = preprocess_image(data['image'])
        
        # Run the image through the network
        output = input_image
        for layer in network:
            output = layer.forward(output)
        
        # The output is the softmax probabilities
        probabilities = output.flatten()
        prediction = int(np.argmax(probabilities))
        confidence = float(probabilities[prediction])
        
        # Return prediction and confidence
        return jsonify({
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': probabilities.tolist()
        })
            
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Start the Flask app
    app.run(host='0.0.0.0', port=5000)
