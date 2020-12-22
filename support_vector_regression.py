# Support Vector Regression (SVR)
# importing libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from PyQt5 import QtWidgets
import cv2

def data(data):
    try:

        # importing dataset
        dataset = pd.read_csv(data)
        x = dataset.iloc[ : , 1: -1].values
        y = dataset.iloc[ :, -1].values
        # re-shaping y to 1D array to 2D array
        y = y.reshape(len(y), 1)
        # print(y)

        # feature scaling
        sc_x = StandardScaler() 
        sc_y = StandardScaler()

        x = sc_x.fit_transform(x)
        y = sc_y.fit_transform(y)

        print(x)
        print(y)

        # Training the SVR model on the whole dataset
        regressor = SVR(kernel='rbf')
        regressor.fit(x, y)

        # Predicting new result
        predict = sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])))
        print(predict)
        # visualising the SVR result
        plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color = 'red')
        plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x)), color='blue')
        plt.title('SVR Regression')
        plt.xlabel('position Lable')
        plt.ylabel('salary')
        # plt.show()

        # visualising the SVR with high resolution and smoother curve
        x_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), 0.1)
        x_grid = x_grid.reshape(len(x_grid), 1)
        plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color = 'red')
        plt.plot(x_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid))), color='blue')
        plt.title('SVR Regression Smoother ')
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