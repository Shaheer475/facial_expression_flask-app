# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:29:16 2022

@author: Shaheer
"""
from flask import Flask, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

from keras.models import model_from_json
import numpy as np
import cv2
import os

# CONSTANT
FACIAL_EXPRESSION = ["Happy", "Neutral", "Sad"]

# load json and create model
json_file = open('LeNet-Facial-Expression-model-version-3.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("LeNet-Facial-Expression-model-version-3.h5")
print("Loaded model from disk")

def model_predict(image_path, model):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(48,48))
    image = np.expand_dims(image, axis=0)
    image = image / 255
    predict = model.predict(image)
    prediction = np.argmax(predict)
    prediction = FACIAL_EXPRESSION[prediction]

    return prediction

app = Flask(__name__)

@app.route("/", methods=['GET'])
def index():
    return render_template("index.html")

@app.route("/predict", methods = ["GET", "POST"])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(image_path=file_path, model=loaded_model)
        
        return preds
    return None


if __name__ == '__main__':
    app.run(debug=True)