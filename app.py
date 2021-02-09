import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template, url_for, request

model = pickle.load(open("corona.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    # return render_template('home.html')
    return render_template('home.html') 

@app.route('/predict', methods=["POST"])
def predict():
    if request.method == 'POST':
        age = request.form['age']
        crp = request.form['crp']
        diabetes = request.form['diabetes']
        blood = request.form['blood']
        fever = request.form['fever']
        oxygen = request.form['oxygen']

        age = float(age)
        crp = float(crp)
        diabetes = float(diabetes)
        blood = float(blood)
        fever = float(fever)
        oxygen = float(oxygen)
        
        message = [age, crp, diabetes, blood, fever, oxygen]

        pred = model.predict([message])

    return render_template('home.html', pred=pred[0])

if __name__ == '__main__':
    app.run(debug=True)