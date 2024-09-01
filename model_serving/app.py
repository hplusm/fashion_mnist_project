import os
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model 
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = load_model('model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = Image.open(io.BytesIO(file.read())).convert('L').resize((28, 28))
    img = np.array(img) / 255.0
    img = img.reshape(1, 28, 28)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return jsonify({'predicted_class': int(predicted_class)})

if __name__ == '__main__':
    app.run(debug=True)