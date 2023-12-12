# -*- coding: utf-8 -*-
"""
Created on Sun Dec 03 05:34:40 2023

@author: Rohan Mehta
"""

import os
import json
import joblib
import flask
import pickle
import numpy as np


import logging

#Define the path
prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')
logging.info("Model Path" + str(model_path))

# Load the model components
clf = joblib.load(os.path.join(model_path, 'random_forest_model.pkl'))
logging.info("Classifier" + str(clf))

# The flask app for serving predictions
app = flask.Flask(__name__)
@app.route('/ping', methods=['GET'])
def ping():
    # Check if the classifier was loaded correctly
    try:
        #regressor
        status = 200
        logging.info("Status : 200")
    except:
        status = 400
    return flask.Response(response= json.dumps(' '), status=status, mimetype='application/json' )

@app.route('/invocations', methods=['POST'])
def transformation():
    request_data = flask.request.data

    try:
        input_array = pickle.loads(request_data)
    except Exception as e:
        return flask.Response(status=400, mimetype='text/plain',
                             response=f"Error deserializing data: {e}")

    # input_array = np.frombuffer(flask.request.data, dtype=np.float32)
    predictions = clf.predict_proba(input_array).tobytes()

    # Transform predictions to JSON
    # result = {
    #     'output': predictions
    #     }
    return flask.Response(response=predictions, status=200, mimetype='application/octet-stream')