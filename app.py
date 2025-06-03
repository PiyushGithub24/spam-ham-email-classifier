from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import pickle

from src.pipeline.predict_pipeline import PredictPipeline


app=Flask(__name__)

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        email_content = request.form.get('email_content')
        predictor = PredictPipeline()
        is_spam = predictor.predict(email_content)

        if (is_spam==1):
            result="Spam"
        else:
            result="Not Spam"
        return render_template('home.html',results=result)
    

if __name__ == "__main__":
    app.run(debug=True)