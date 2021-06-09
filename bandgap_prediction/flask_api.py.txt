# -*- coding: utf-8 -*-

from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

pickle_in = open("model.pkl","rb")
bg_model=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome All"


@app.route('/predict_file',methods=["POST"])
def predict_bandgap_file():
    """Let's Authenticate the Bandgaps Application 
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """
    data_infer=pd.read_csv(request.files.get("file"))
    compounds_infer = data_infer['composition'].tolist()
    p=[]
    for i in range(6,82):
        p.append(i)
    X_infer = data_infer.iloc[:,p]
    y_infer = data_infer.iloc[1]
    y_infer_pred = bg_model.predict(X_infer)
    bandgap_dict = dict(zip(compounds_infer, y_infer_pred))
    print("The predicted values of Eg are:\n")
    return bandgap_dict

if __name__=='__main__':
    app.debug = True
    app.run(host='localhost',port=5000)
