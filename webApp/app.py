import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

loaded_model = pickle.load(open('heartAttackModel.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

def prediction_function(pred_list):
    predict = np.array(pred_list).reshape(1, 13)
    loaded_model = pickle.load(open('heartAttackModel.pkl', 'rb'))
    result = loaded_model.predict(predict)
    return result[0] 
    
@app.route('/result', methods = ['POST','GET'])
def result():
    if request.method == 'POST':
        pred_list = request.form.to_dict()
        pred_list = list(pred_list.values())
        pred_list = list(map(int, pred_list))
        result = prediction_function(pred_list=pred_list)
            
        if int(result) == 1:
            prediction = 'Risk of Heart Attack'
        else:
            prediction = 'Low Risk of Heart Attack'
        return render_template("result.html", Prediction=prediction)
    


if __name__ == "__main__":
    app.run(debug=True)