import numpy as np
import joblib

input_data = np.array([[
    0,1,
    0.24316406250000003,
    1,0,
    1,0,
    0,1,
    0,0,0,0,1,
    1,0,
    0.2011817930015696,
    0.18213058419243985,
    1,0,0
]])

model = joblib.load("model/XGBoost.pkl")
print(int(model.predict(input_data)[0]))
print(round(model.predict_proba(input_data)[0][0]*100, 2))