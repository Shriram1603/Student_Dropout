from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('dropout_form.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load the model
    model = pickle.load(open('dropout.pkl', 'rb'))

    # Get form data
    father_occupation = int(request.form['father_occupation'])
    father_education = int(request.form['father_education'])
    mother_occupation = int(request.form['mother_occupation'])
    mother_education = int(request.form['mother_education'])
    gender = int(request.form['gender'])
    religion = int(request.form['religion'])
    MOI=int(request.form['medium_of_instruction'])
    community=int(request.form['community'])
    dg=int(request.form['disability_group'])
    attendance=int(request.form['attendance'])
    long_abs=int(request.form['long_absentees'])
    single_parent=int(request.form['single_parent'])
    smoker=int(request.form['smoker'])
    conduct=int(request.form['conduct'])

    # Make prediction
    prediction = model.predict([[father_occupation,father_education,mother_occupation,mother_education,gender,religion,MOI,community,dg,attendance,long_abs,single_parent,smoker,conduct]])

    # Process prediction (e.g., display result on a new page)
    # For simplicity, just return the prediction here

    if(prediction==0):
        return f'Prediction: Will DropOut'
    elif(prediction==1):
        return f'Will continue'

if __name__ == '__main__':
    app.run(debug=True)
