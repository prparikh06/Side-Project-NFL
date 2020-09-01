import pandas as pd
import nltk 
import sys
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

PATH = 'dataframe_ints.xlsx'
data = pd.read_excel(PATH)
# print(data)
X = data[['OffenseTeam','DefenseTeam','YardLine','Down','ToGo']]
Y = data['IsTouchdown']

# split
seed = 420
test_size = 0.30
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# fit model to training data
model = XGBClassifier()
model.fit(X_train, y_train)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# evaluate one sample
test = np.array([0,22,4,4,20]).reshape(1,5)
df = pd.DataFrame(test, columns=X.columns)
print(model.predict(df)[0])

# save model
# joblib.dump(model, 'model_xgboost.txt') 
#load model
# xgb = joblib.load('model_xgboost.txt')