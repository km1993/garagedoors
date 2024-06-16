from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your model
model = load_model('path_to_your_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    image = np.array(data['image'])  # Assuming image is sent as a list
    image = np.expand_dims(image, axis=0)  # Add batch dimension if needed

    # Make prediction
    prediction = model.predict(image)
    result = np.argmax(prediction, axis=-1)

    return jsonify(result.tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
