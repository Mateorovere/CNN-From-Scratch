# CNN-From-Scratch (Work in progress)

A complete project to classify handwritten digits using a Convolutional Neural Network (CNN) built from scratch with NumPy and a web interface created with Flask (backend) and React (frontend). Users can draw a digit on a canvas and get a prediction from the trained CNN model.

---

## Features:
- CNN from Scratch: A custom-built Convolutional Neural Network implemented in Python using only NumPy.
- Web Interface:
    - Draw digits on a 28x28 canvas.
    - Classify the drawn digit with a single click.
    - View the predicted digit instantly.
- Technologies Used:
    - Backend: Python, Flask
    - Frontend: React
    - Data Handling: NumPy and Scikit-learn
    - Model Training: A custom-built CNN trained on the MNIST dataset.

---

Installation
1. Clone the Repository

```bash
git clone https://github.com/Mateorovere/CNN-From-Scratch.git
cd CNN-From-Scratch

```

2. Backend Setup

```bash
python -m venv venv
source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt

```

3. Frontend Setup

```bash
cd ../frontend
npm install

```

## Running the Application
### Step 1: Start the Flask Backend

```bash
cd backend
python app.py

```
The backend will be available at http://127.0.0.1:5000.

## Step 2: Start the React Frontend

```bash
cd frontend
npm start

```

The frontend will open automatically at http://localhost:3000.

## Usage
1. Draw a digit (0-9) on the canvas in the web interface.
2. Click the "Classify" button to send the drawing to the backend.
3. View the predicted digit below the canvas.

---

## Model Details
The CNN is implemented from scratch using NumPy. It consists of the following components:

- Convolution Layer: Extracts spatial features from input images.
- ReLU Activation: Adds non-linearity to the model.
- Max Pooling Layer: Reduces spatial dimensions to make computations efficient.
- Fully Connected Layer: Maps extracted features to class probabilities.

The model was trained on the MNIST dataset and achieves high accuracy on the test set.

---

## Future Improvements

Use docker-compose.yml
