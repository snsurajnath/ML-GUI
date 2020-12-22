################# Simple linear regression
#  import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from PyQt5 import QtCore, QtGui, QtWidgets
# from machineLearning3 import *

#importing dataset
def data(data):
    try:
        dataset = pd.read_csv(data)
        # x is taking all feature data i.e. independent variables
        x = dataset.iloc[ : , :-1 ].values # include all rows except the last result column
        # y is taking all dependent data i.e. result column
        y= dataset.iloc[ : , -1].values # getting last column of the data

        # print(x)
        # print(y)

        # splitting the dataset into training set and test set
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
        # print(x_train)
        # print(x_test)
        # print(y_train)
        # print(y_test)
        # from sklearn.preprocessing import StandardScaler
        # sc = StandardScaler()
        # x_train[: , : ] = sc.fit_transform(x_train[: , : ])
        # x_test[: , : ] = sc.transform(x_test[: , : ])

        ## trining the simple linear regression model on the training set

        regressor = LinearRegression()
        regressor.fit(x_train, y_train)

        ### predicting the test set result
        y_predict = regressor.predict(x_test)

        # visualise the training set result
        plt.scatter(x_train, y_train, color = 'red')
        plt.plot(x_train, regressor.predict(x_train), color = 'blue', label ='Training Simple regressor line')
        plt.title('Salary vs Experience (Training set)')
        plt.xlabel('years pf experience')
        plt.ylabel('salary')
        # plt.show()
        # visualise the test set result
        plt.scatter(x_test, y_test, color = 'green')
        plt.plot(x_train, regressor.predict(x_train), color = 'black', label ='Test Simple regressor line')
        plt.title('Salary vs Experience (Test set)')
        plt.xlabel('years pf experience')
        plt.ylabel('salary')
        plt.legend()
        # plt.show()
        plt.savefig('polyplot.png')
        m = cv2.imread('polyplot.png')
        r = cv2.resize(m, (375, 291))
        cv2.imwrite('polyplot.png', r)
        plt.close()
        # plt.show()
        res = "Successfull"
        return y_predict
    except Exception as e:
        print(e)
        msg = QtWidgets.QMessageBox()
        # msg.setWindowIcon(QtGui.QIcon('logo.png'))
        msg.setWindowTitle("Error")
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText(f'LLR: {str(e)}')
        msg.exec_()

        # return e