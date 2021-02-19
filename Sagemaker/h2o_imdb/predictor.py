# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 10:32:40 2019

@author: 1009758/154943
"""

import os
import json
import flask
import boto3
import time
import pyarrow
from pyarrow import feather
#from boto3.s3.connection import S3Connection
#from botocore.exceptions import ClientError
#import pickle
import pandas as pd
import pickle
import gensim
from helper_init_19 import *


import h2o
h2o.init()
h2o.connect()

import logging
with open('artifacts/tfidf_pickle','rb') as f:
    tfidf_fit = pickle.load(f)  
with open('artifacts/tfidfngram_pickle','rb') as f:
    tfidfngram_fit = pickle.load(f)

#Define the path
# prefix = '/opt/ml/'
# model_path = os.path.join(prefix, 'model')
# logging.info("Model Path" + str(model_path))

# # Load the model components
# path = os.path.join(model_path, 'GBM_model_python_1611037061041_5316')
path_mdl ='artifacts/DRF_model_python_1613742661629_1'

classifier = h2o.load_model(path_mdl)

#logging.info("classifier" + str(classifier))

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
    # Get input JSON data and convert it to a DF
    input_json = flask.request.get_json()
    text = input_json['input']['text']
    print("body received is")
    print(text)
    logging.info(input_json)
    print(input_json)    
    data=pd.DataFrame([[text]],columns=['text'])
    data_transform=transform_data(data,tfidf_fit,tfidfngram_fit)
    
    print("data transformation is complete")
    scoring= h2o.H2OFrame(data_transform)
    print("scoring tranformation is complete")
    predictions = classifier.predict(scoring)
    print(data_transform.shape)
    p1=predictions.as_data_frame().p1
    print(p1)
    p1=str(p1.values)
    # Transform predictions to JSON
    result = {
        'sentiment_probablity': p1
        }

    resultjson = json.dumps(result)
    return flask.Response(response=resultjson, status=200, mimetype='application/json')
