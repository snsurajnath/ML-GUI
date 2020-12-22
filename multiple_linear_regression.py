# multiple linear regression
# importing the  libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

## importing datasset
def data(data):
    try:
        dataset = pd.read_csv(data)
        x = dataset.iloc[ : , : -1].values
        y = dataset.iloc[ : , -1].values
        # print(x)
        # print(y)


        # encode categorical data
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[3])], remainder='passthrough')
        x = np.array(ct.fit_transform(x))
        x_train , x_test , y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=0)

        # print(x)

        # training the Multiple Linear Regression Model on Training set
        regressor = LinearRegression()
        regressor.fit(x_train, y_train)

        # predicting the test set result
        y_predict = regressor.predict(x_test)
        np.set_printoptions(precision=2)
        print(np.concatenate((y_predict.reshape(len(y_predict),1), y_test.reshape(len(y_test),1)), 1))
        return np.concatenate((y_predict.reshape(len(y_predict),1), y_test.reshape(len(y_test),1)), 1)
    except Exception as e:
        msg = QtWidgets.QMessageBox()
        # msg.setWindowIcon(QtGui.QIcon('logo.png'))
        msg.setWindowTitle("Error")
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText(f'LLR: {str(e)}')
        msg.exec_()