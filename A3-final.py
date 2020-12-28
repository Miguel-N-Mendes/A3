# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 15:01:47 2020

@author: Miguel
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open(r'C:\Users\Miguel\Downloads\modelA3.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')



@app.route('/predict',methods=['POST'])
def predict():
    int_features=[x for x in request.form.values()]
    final=np.array(int_features)
    col = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
    data_unseen = pd.DataFrame([final], columns = col)
    print(int_features)
    print(final)
    prediction=predict_model(model, data=data_unseen, round = 0)
    prediction=str(prediction.Label[0])
    if prediction == 0:
        pred='Rejected'

    else:
        pred='Approved'
    return render_template('home.html',prediction_text='The loan is {}'.format(pred))

@app.route('/predict_api',methods=['POST'])
def predict_api():

    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)
if __name__ == '__main__':
    app.run( debug = True, use_reloader=False)



