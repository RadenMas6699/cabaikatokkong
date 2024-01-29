import tensorflow as tf
import numpy as np
import os
import cv2
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Load the trained model
model_path = 'modelresnet.h5'
model = tf.keras.models.load_model(model_path)

# Define the classes for prediction
CLASSES = ['Matang', 'Mengkal', 'Mentah']

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return "Work"

@app.route('/predict', methods=['GET'])
def predict():
    return "Predict"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        uploads_dir = os.path.join(basepath, 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        file_path = os.path.join(uploads_dir, secure_filename(file.filename))
        file.save(file_path)

        # Preprocess the image
        IMG_SIZE = 299
        img_array = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        Xt = np.array(new_array, dtype=np.float32).reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0

        # Perform prediction with the model
        prediction = model.predict(Xt)

        # Get the predicted class and confidence
        predicted_class_index = np.argmax(prediction)
        predicted_class = CLASSES[predicted_class_index]
        confidence = prediction[0][predicted_class_index]

        # Remove the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

        return jsonify({'predicted_class': predicted_class, 'confidence': float(confidence)})

if __name__ == '__main__':
    app.run(debug=False)