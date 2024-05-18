from flask import Flask, request, jsonify, render_template
import os
import pickle
from PIL import Image
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)


# Load the model
MODEL_PATH = '../models/model.pkl'
with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    image_path = os.path.join('uploads', image.filename)
    image.save(image_path)

    # Preprocess the image
    img = Image.open(image_path).resize((64, 64))
    img_array = np.array(img).flatten().reshape(1, -1)

    # Get prediction from the model
    prediction = model.predict(img_array)
    prediction_label = 'Red' if prediction[0] == 0 else 'Blue'

    os.remove(image_path)  # Clean up the saved image
    return jsonify({'prediction': prediction_label})

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
