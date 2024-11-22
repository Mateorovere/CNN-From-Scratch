from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import base64
import io
import pickle

# Load your trained CNN model
with open('cnn_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def preprocess_image(image_data):
    """
    Convert the base64 image to a 28x28 grayscale image and normalize it.
    """
    image = Image.open(io.BytesIO(base64.b64decode(image_data.split(',')[1])))
    image = image.resize((28, 28)).convert('L')
    image_array = np.array(image) / 255.0
    return image_array

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_data = data['image']
    input_image = preprocess_image(image_data)
    input_image = input_image.reshape(1, 28, 28)

    logits = model.forward(input_image[0])
    prediction = np.argmax(logits)

    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
