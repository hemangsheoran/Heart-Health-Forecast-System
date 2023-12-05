# Importing essential libraries
import pandas as pd
import joblib
from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from joblib import load
from xgboost import XGBClassifier
scaler = StandardScaler()

svm_2_layer = load('2_layer_svm.pkl')
cnn_2_layer = load_model('2_layer_cnn.keras')

svm_4_layer = load('svm_4_layer.pkl')
cnn_4_layer = load_model('cnn_4_layer.keras')

svm_8_layer = load('svm_8_layer.pkl')
cnn_8_layer = load_model('cnn_8_layer.keras')

svm_16_layer = load('svm_16_layer.pkl')
cnn_16_layer = load_model('cnn_16_layer.keras')

svm_24_layer = load('svm_24_layer.pkl')
cnn_24_layer = load_model('cnn_24_layer.keras')

svm_final = load('final_svm_linking.pkl')
cnn_final = load_model('final_cnn_linking.keras')


xgmodel = XGBClassifier()
xgmodel.load_model('xgboost_model.json')

logistic_model = joblib.load('logistic_regression_model.pkl')

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('main.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        sex = request.form.get('sex')
        cp = request.form.get('cp')
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = request.form.get('fbs')
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = request.form.get('exang')
        oldpeak = float(request.form['oldpeak'])
        slope = request.form.get('slope')




        data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope]])

        columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
                   'Oldpeak', 'ST_Slope']
        pddata = pd.DataFrame(data, columns=columns)

        pd1data = pddata.astype({
            'Age': np.int64,
            'Sex': np.int64,
            'ChestPainType': np.int64,
            'RestingBP': np.int64,
            'Cholesterol': np.int64,
            'FastingBS': np.int64,
            'RestingECG': np.int64,
            'MaxHR': np.int64,
            'ExerciseAngina': np.int64,
            'Oldpeak': float,
            'ST_Slope': np.int64
        })
        x_test = pd.read_csv('x_test.csv')
        #x_test.info()
        x_test = pd.concat([x_test, pddata], ignore_index=True)
        #x_test.info()
        #print(pd1data)
        #print(x_test)
        user_input_scaled = scaler.fit_transform(x_test)
        print(user_input_scaled[150])
        svm_model_test_predictions = []
        start_2_layer = cnn_2_layer.predict(user_input_scaled)
        result_2_layer = svm_2_layer.predict(start_2_layer)
        print(result_2_layer[150])
        svm_model_test_predictions.append([result_2_layer[150]])

        start_4_layer = cnn_4_layer.predict(user_input_scaled)
        result_4_layer = svm_4_layer.predict(start_4_layer)
        print(result_4_layer[150])
        svm_model_test_predictions.append([result_4_layer[150]])
        start_8_layer = cnn_8_layer.predict(user_input_scaled)
        result_8_layer = svm_8_layer.predict(start_8_layer)
        print(result_8_layer[150])
        svm_model_test_predictions.append([result_8_layer[150]])

        start_16_layer = cnn_16_layer.predict(user_input_scaled)
        result_16_layer = svm_16_layer.predict(start_16_layer)
        print(result_16_layer[150])
        svm_model_test_predictions.append([result_16_layer[150]])

        start_24_layer = cnn_24_layer.predict(user_input_scaled)
        result_24_layer = svm_24_layer.predict(start_24_layer)
        print(result_24_layer[150])
        svm_model_test_predictions.append([result_24_layer[150]])

        y_pred = xgmodel.predict(pd1data)
        print(y_pred)
        svm_model_test_predictions.append(y_pred)
        predictions_logistic = logistic_model.predict(pd1data)
        print(predictions_logistic)
        svm_model_test_predictions.append(predictions_logistic)

        #svm_model_test_predictions = [[result_2_layer[150],result_4_layer[150],result_8_layer[150],result_16_layer[150],result_24_layer[150],y_pred[0],predictions_logistic[0]]]
        print(svm_model_test_predictions)
        combined_outputs_test = np.array(svm_model_test_predictions).T
        print(combined_outputs_test)
        cnn_predictions_test = cnn_final.predict(combined_outputs_test)
        final_result = svm_final.predict(cnn_predictions_test)
        print(final_result[0])

        final = result_2_layer[150]+result_4_layer[150]+result_8_layer[150]+result_16_layer[150]+result_24_layer[150]+y_pred[0]+predictions_logistic[0]

        # if final >= 4:
        #     result = 2
        # else:
        #     result = 0

        accuracy = (final/7)*100


        return render_template('result.html', prediction=final_result[0], accurate=accuracy, layer_2=result_2_layer[150], layer_4=result_4_layer[150], layer_8=result_8_layer[150], layer_16=result_16_layer[150], layer_24=result_24_layer[150], xgboost = y_pred[0], logistic = predictions_logistic[0])


if __name__ == '__main__':
    app.run(debug=True)
