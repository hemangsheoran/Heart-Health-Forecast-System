import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from joblib import dump
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import joblib
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from joblib import load
from sklearn.linear_model import LogisticRegression

x_train = pd.read_csv('x_train.csv')  # assuming 'x_test.csv' contains your test data
y_train = pd.read_csv('y_train.csv')

x_test = pd.read_csv('x_test.csv')  # assuming 'x_test.csv' contains your test data
y_test = pd.read_csv('y_test.csv')  # assuming 'y_test.csv' contains corresponding labels
from xgboost import XGBClassifier
# Load your SVM models (svm_model_1, svm_model_2, ..., svm_model_5) previously trained
# x_train.info()
# x_test.info()
logistic_model = joblib.load('logistic_regression_model.pkl')
xgmodel = XGBClassifier()
xgmodel.load_model('xgboost_model.json')

x_train_list = x_train.values.tolist()
y_train_list = y_train.values.tolist()




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

user_input_scaled = scaler.fit_transform(x_train)
test_input_scaled = scaler.fit_transform(x_test)
# Make predictions using each SVM model on x_test
# svm_model_predictions = []
#
# start_2_layer = cnn_2_layer.predict(user_input_scaled)
# result_2_layer = svm_2_layer.predict(start_2_layer)
# print(result_2_layer)
# svm_model_predictions.append(result_2_layer)
#
# start_4_layer = cnn_4_layer.predict(user_input_scaled)
# result_4_layer = svm_4_layer.predict(start_4_layer)
# print(result_4_layer)
# svm_model_predictions.append(result_4_layer)
#
# start_8_layer = cnn_8_layer.predict(user_input_scaled)
# result_8_layer = svm_8_layer.predict(start_8_layer)
# print(result_8_layer)
# svm_model_predictions.append(result_8_layer)
#
# start_16_layer = cnn_16_layer.predict(user_input_scaled)
# result_16_layer = svm_16_layer.predict(start_16_layer)
# print(result_16_layer)
# svm_model_predictions.append(result_16_layer)
#
# start_24_layer = cnn_24_layer.predict(user_input_scaled)
# result_24_layer = svm_24_layer.predict(start_24_layer)
# print(result_24_layer)
# svm_model_predictions.append(result_24_layer)
#
# y_pred = xgmodel.predict(x_train)
# print(y_pred)
# svm_model_predictions.append(y_pred)
#
# predictions_logistic = logistic_model.predict(x_train)
# print(predictions_logistic)
# svm_model_predictions.append(predictions_logistic)
#
#
# combined_outputs = np.array(svm_model_predictions).T
# print(combined_outputs)
#
#







svm_model_test_predictions = []

start_2_layer = cnn_2_layer.predict(test_input_scaled)
result_2_layer = svm_2_layer.predict(start_2_layer)
#print(result_2_layer)

svm_model_test_predictions.append(result_2_layer)

start_4_layer = cnn_4_layer.predict(test_input_scaled)
result_4_layer = svm_4_layer.predict(start_4_layer)
#print(result_4_layer)

svm_model_test_predictions.append(result_4_layer)

start_8_layer = cnn_8_layer.predict(test_input_scaled)
result_8_layer = svm_8_layer.predict(start_8_layer)
#print(result_8_layer)

svm_model_test_predictions.append(result_8_layer)

start_16_layer = cnn_16_layer.predict(test_input_scaled)
result_16_layer = svm_16_layer.predict(start_16_layer)
#print(result_16_layer)

svm_model_test_predictions.append(result_16_layer)

start_24_layer = cnn_24_layer.predict(test_input_scaled)
result_24_layer = svm_24_layer.predict(start_24_layer)
#print(result_24_layer)

svm_model_test_predictions.append(result_24_layer)

y_pred = xgmodel.predict(x_test)
print(y_pred)
svm_model_test_predictions.append(y_pred)

predictions_logistic = logistic_model.predict(x_test)
print(predictions_logistic)
svm_model_test_predictions.append(predictions_logistic)


combined_outputs_test = np.array(svm_model_test_predictions).T
print(combined_outputs_test)

cnn_predictions_test = cnn_final.predict(combined_outputs_test)
final_result = svm_final.predict(cnn_predictions_test)


# model = Sequential()
# model.add(Dense(64, activation='relu', input_shape=(combined_outputs.shape[1],)))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
#
# Compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# optimizers = {
#     'SGD': SGD(learning_rate=0.01)
# }

# for optimizer_name, optimizer_instance in optimizers.items():
#     model.compile(optimizer=optimizer_instance, loss='binary_crossentropy', metrics=['accuracy'])
#
#     print(f'Training model with {optimizer_name} optimizer...')
#     model.fit(combined_outputs, y_train, epochs=10, batch_size=32, validation_data=(combined_outputs_test, y_test))
#     #model.save('final_cnn_linking.keras')
#     cnn_predictions = model.predict(combined_outputs)
#     cnn_predictions_test = model.predict(combined_outputs_test)
#     svm_model_final = SVC(kernel='linear')
#     svm_model_final.fit(cnn_predictions, y_train)
#     #dump(svm_model_final, 'final_svm_linking.pkl')
#     final_result = svm_model_final.predict(cnn_predictions_test)
#     accuracy = accuracy_score(y_test, final_result)
#     print(f'Accuracy of final_result: {accuracy}')
# model.fit(combined_outputs, y_train, epochs=10, batch_size=32, validation_data=(combined_outputs_test, y_test))
# cnn_predictions = model.predict(combined_outputs)
# cnn_predictions_test = model.predict(combined_outputs_test)
# print(cnn_predictions)
# svm_model_final = SVC(kernel='linear')
# svm_model_final.fit(cnn_predictions, y_train)
# final_result = svm_model_final.predict(cnn_predictions_test)
# print(final_result)
# Combine predictions to form final_result (voting mechanism)
# final_result = []
# for i in range(len(x_test)):
#     count_ones = sum(svm_model_predictions[j][i] for j in range(7))  # 5 is the number of SVM models
#     final_result.append(1 if count_ones >= 4 else 0)  # Adjust the threshold for majority voting

# Calculate accuracy of final_result compared to y_test
# accuracy = accuracy_score(y_test, final_result)
# print(f'Accuracy of final_result: {accuracy}')
accuracy = accuracy_score(y_test, result_2_layer)
print(f'Accuracy of 2 Layer cnn + svm Hybrid Model: {accuracy}')
accuracy = accuracy_score(y_test, result_4_layer)
print(f'Accuracy of 4 Layer cnn + svm Hybrid Model: {accuracy}')
accuracy = accuracy_score(y_test, result_8_layer)
print(f'Accuracy of 8 Layer cnn + svm Hybrid Model: {accuracy}')
accuracy = accuracy_score(y_test, result_16_layer)
print(f'Accuracy of 16 Layer cnn + svm Hybrid Model: {accuracy}')
accuracy = accuracy_score(y_test, result_24_layer)
print(f'Accuracy of 24 Layer cnn + svm Hybrid Model: {accuracy}')
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of xgtModel: {accuracy}')
accuracy = accuracy_score(y_test, predictions_logistic)
print(f'Accuracy of logistic Regression Model: {accuracy}')
accuracy = accuracy_score(y_test, final_result)
print(f'Accuracy of final output is: {accuracy}')

conf_matrix = confusion_matrix(y_test, final_result)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

print(classification_report(y_test, final_result))
plt.show()