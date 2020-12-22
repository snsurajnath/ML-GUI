#Polinomial linear regression
# importing the libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from PyQt5 import QtWidgets
import cv2

#importing dataset
def data(data):
    try:
        dataset = pd.read_csv(data)
        x = dataset.iloc[  : ,1 :-1 ].values
        y = dataset.iloc[ : ,-1 ].values

        # print(x)
        # print("----------------------------")
        # print(y)

        # Training the Linear regression model on the whole data set
        lin_regressor = LinearRegression()
        lin_regressor.fit(x ,y)

        # Training the Polynomial regression model on the whole data set
        poly_regressor = PolynomialFeatures(degree = 4)
        x_poly = poly_regressor.fit_transform(x)
        lin_reg2 = LinearRegression()
        lin_reg2.fit(x_poly, y)

        #Visualising the Linear Regression Result
        plt.scatter(x, y, color = 'red')
        plt.plot(x, lin_regressor.predict(x), color='green', label ='Linear')
        plt.title('Linear Regression')
        plt.xlabel('position Lable')
        plt.ylabel('salary')
        # plt.legend()

        # plt.show()

        #Visualising the Polynomial Regression Result
        plt.scatter(x, y, color = 'red')
        plt.plot(x, lin_reg2.predict(x_poly), color='blue', label ='Polynomial')
        plt.title('Polynomial Regression')
        plt.xlabel('position Lable')
        plt.ylabel('salary')
        plt.legend()
        # plt.show()
        plt.savefig('polyplot.png')
        m= cv2.imread('polyplot.png')
        r = cv2.resize(m,(375, 291))
        cv2.imwrite('polyplot.png',r)
        plt.close()
        # cv2.waitKey()
        # scene = QtWidgets.QGraphicsScene()
        # scene.setPixmap()
        # predicting a new result with Linear regression
        # print(lin_regressor.predict([[6.5]]))
        # # predicting a new result with Polynomial regression
        # print(lin_reg2.predict(poly_regressor.fit_transform([[6.5]])))
        return lin_reg2.predict(poly_regressor.fit_transform([[6.5]]))

    except Exception as e:
        msg = QtWidgets.QMessageBox()
        # msg.setWindowIcon(QtGui.QIcon('logo.png'))
        msg.setWindowTitle("Error")
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText(f'LLR: {str(e)}')
        msg.exec_()