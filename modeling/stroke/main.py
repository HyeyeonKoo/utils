#-*-coding:utf-8-&-

"""
JSON으로 입력
    {
        gender_Female,gender_Male,
        age,
        hypertension_0,hypertension_1,
        heart_disease_0,heart_disease_1,
        ever_married_Yes,ever_married_No,
        work_type_Private,work_type_Never_worked,work_type_children,work_type_Self-employed,work_type_Govt_job,
        Residence_type_Rural,Residence_type_Urban,
        avg_glucose_level,
        bmi,
        smoking_status_formerly smoked,smoking_status_smokes,smoking_status_never smoked
    }
"""

import joblib
import json
import numpy as np
from flask import Flask
from flask import request

app = Flask(__name__)


@app.route("/isstroke", methods=["POST"])
def is_stroke():
    data = json.loads(request.get_json())
    input_data = get_data(data)

    model = joblib.load("data/XGBoost.pkl")
    
    stroke = int(model.predict(input_data)[0])
    proba = round(model.predict_proba(input_data)[0][0]*100, 2)

    return json.dumps({
        "stroke": stroke,
        "proba": proba
    })


def get_data(data):
    input_data = []

    if data["gender"] == "Female":
        input_data += [1, 0]
    elif data["gender"] == "Male":
        input_data += [0, 1]
    else:
        RuntimeError("gender must be one of Female, Male")

    input_data.append(data["age"])

    if data["hypertension"] == 0:
        input_data += [1, 0]
    else:
        input_data += [0, 1]

    if data["heart_disease"] == 0:
        input_data += [1, 0]
    else:
        input_data += [0, 1]

    if data["ever_married"] == "Yes":
        input_data += [1, 0]
    elif data["ever_married"] == "No":
        input_data += [0, 1]
    else:
        RuntimeError("ever_married mus be one of Yes, No.")

    if data["work_type"] == "Private":
        input_data += [1, 0, 0, 0, 0]
    elif data["work_type"] == "Never_worked":
        input_data += [0, 1, 0, 0, 0]
    elif data["work_type"] == "children":
        input_data += [0, 0, 1, 0, 0]
    elif data["work_type"] == "Self-employed":
        input_data += [0, 0, 0, 1, 0]
    elif data["work_type"] == "Govt_job":
        input_data += [0, 0, 0, 0, 1]
    else:
        RuntimeError("work_type must be one of Private, Never_worked, children, Self-employed, Govt_job.")

    if data["Residence_type"] == "Rural":
        input_data += [1, 0]
    else:
        input_data += [0, 1]

    input_data.append(data["avg_glucose_level"])

    input_data.append(data["bmi"])

    if data["smoking_status"] == "formerly smoked":
        input_data += [1, 0, 0]
    elif data["smoking_status"] == "smokes":
        input_data += [0, 1, 0]
    elif data["smoking_status"] == "never smoked":
        input_data += [0, 0, 1]
    else:
        RuntimeError("smoking_status must be one of formerly smoked, smokes, never smoked.")

    return np.array(data)
    

if __name__=="__main__":
    app.run(
        host="127.0.0.1",
        port="1212",
    )