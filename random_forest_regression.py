# Decision Tree Regression
# importing libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

def data(data):
    try:
        # importing dataset
        dataset = pd.read_csv(data)
        x = dataset.iloc[ : , 1: -1].values
        y = dataset.iloc[ :, -1].values
        # print(y)


        # Training the Decision Tree Regressoion model on the whole dataset
        regressor = RandomForestRegressor(n_estimators= 10, random_state= 0)
        regressor.fit(x, y)

        # Predicting a new result
        predict = regressor.predict([[7]])
        print(predict)

        # Visualising the Decision tree Regression (High resolution)
        x_grid = np.arange(min(x), max(x), 0.1)
        x_grid = x_grid.reshape(len(x_grid), 1)
        plt.scatter(x, y, color = 'red')
        plt.plot(x_grid, regressor.predict(x_grid), color='blue')
        plt.title('Random Forest Regression Smoother ')
        plt.xlabel('position Lable')
        plt.ylabel('salary')
        # plt.show()
        plt.legend()
        # plt.show()
        plt.savefig('polyplot.png')
        m = cv2.imread('polyplot.png')
        r = cv2.resize(m, (375, 291))
        cv2.imwrite('polyplot.png', r)
        plt.close()
        return predict
    except Exception as e:
        msg = QtWidgets.QMessageBox()
        # msg.setWindowIcon(QtGui.QIcon('logo.png'))
        msg.setWindowTitle("Error")
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText(f'LLR: {str(e)}')
        msg.exec_() 