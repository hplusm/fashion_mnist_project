from flask import Flask, request, jsonify
from keras._tf_keras.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = load_model('model.keras')

def preprocess_image(file):
    img = Image.open(io.BytesIO(file.read())).convert('L').resize((28, 28))
    img_array = np.array(img) / 255.0
    return img_array.reshape(1, 28, 28)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    img = preprocess_image(file)
    predictions = model.predict(img)
    predicted_class = int(np.argmax(predictions, axis=1)[0])
    return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)