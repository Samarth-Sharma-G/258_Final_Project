from flask import Flask, request, render_template, redirect, url_for
import numpy as np
from PIL import Image
import urllib.request
import json
import os
import ssl
import json

app = Flask(__name__)

# Bypass SSL certificate verification
def allow_self_signed_https(allowed):
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allow_self_signed_https(True)

# Model prediction function
def get_prediction(image_array, url, api_key):
    data = {
        "input_data": [image_array.tolist()],
        "params": {}
    }
    body = str.encode(json.dumps(data))
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + api_key,
        'azureml-model-deployment': 'waffer-defect-1'
    }
    req = urllib.request.Request(url, body, headers)
    try:
        response = urllib.request.urlopen(req)
        result = response.read()
        ans = json.loads(result.decode('utf-8'))
        predicted_class = np.array(ans)[0]
        print(predicted_class)
        if predicted_class <0.5:
            ans = 'Normal'
        else:
            ans = 'Defect Detected'
        return ans
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        img = Image.open(file.stream)
        img_array = np.array(img)
        img_array = img_array.reshape(26,26,1)
        # Constants
        URL = 'https://team-pi-vtzdu.westus2.inference.ml.azure.com/score'
        with open('secrets.json') as f:
            secrets = json.load(f)
            API_KEY = secrets['API_KEY']
        prediction = get_prediction(img_array, URL, API_KEY)
        return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
