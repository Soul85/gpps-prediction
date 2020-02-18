from flask import render_template, request
from mlserve import app, encoder, minmax, model
import numpy as np

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    capacity = minmax.transform(np.array([int(request.form['capacity'])]).reshape(-1, 1))
    fuel = encoder.transform(np.array([request.form['fuel']]))
    x = np.array([capacity, fuel])
    prediction = model.predict(x.reshape(1, -1))
    return render_template('index.html', result=int(prediction[0]), capacity=request.form['capacity'], fuel=request.form['fuel'])