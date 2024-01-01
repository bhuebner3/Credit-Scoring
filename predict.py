
import pickle
import xgboost as xgb
from flask import Flask
from flask import request
from flask import jsonify


C = 1.0
input_file = f'model_C={C}.bin'

with open(input_file, 'rb') as f_in:
    (dv, model) = pickle.load(f_in)


app = Flask('churn')

# input and output will be in json format
@app.route('/predict', methods = ['POST'])
def predict():
    # parses json and turns into python dict
    customer = request.get_json()

    X = dv.transform([customer])
    dtest = xgb.DMatrix(X, feature_names=dv.get_feature_names_out())
    y_pred = model.predict(dtest)[0]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)