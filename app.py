from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Enable CORS for the entire app
# CORS(app)

with open('model/config.json', 'r') as json_file:
    json_config = json_file.read()

model = tf.keras.models.model_from_json(json_config)

# Load weights into the model
model.load_weights('model/model.weights.h5')

# Compile the model (you might need to compile it if you want to use it for predictions)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Initialize Tokenizer
tokenizer = Tokenizer(num_words=10000)  # Adjust num_words as needed


@app.route('/')
def home():
    return 'Hello, World!'


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        # Get the text from the request
        text = request.get_json()['text']
        print(text)
        # Tokenize and preprocess the input text
        sequences = tokenizer.texts_to_sequences([text])
        padded_sequences = pad_sequences(sequences, maxlen=100)  # Adjust maxlen as needed
        padded_sequences = np.expand_dims(padded_sequences, axis=-1)  # Add an extra dimension for Conv1D input
        # Make prediction
        prediction = model.predict(padded_sequences)[0]
        predicted_class = np.argmax(prediction)  # Assuming the model outputs class probabilities
        return jsonify({'prediction': str(predicted_class)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5001)

# curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"text":"Your example text here"}'
