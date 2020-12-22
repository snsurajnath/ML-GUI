import os

from PyQt5 import QtCore, QtGui, QtWidgets
from tkinter.filedialog import askopenfilename
from tkinter import Tk
import pandas as pd
import cv2
# import preprocessing
from PyQt5.QtGui import QPixmap

import simple_linear_regression
import multiple_linear_regression
import polinomial_linear_regression
import random_forest_regression
import decision_tree_regression
import support_vector_regression

class Ui_MainWindow(object):
    try:
        def setupUi(self, MainWindow):
            MainWindow.setObjectName("MainWindow")
            MainWindow.setFixedSize(798, 602)
            self.centralwidget = QtWidgets.QWidget(MainWindow)
            self.centralwidget.setObjectName("centralwidget")
            self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
            self.verticalLayout.setObjectName("verticalLayout")
            self.tabView = QtWidgets.QTabWidget(self.centralwidget)
            self.tabView.setTabPosition(QtWidgets.QTabWidget.North)
            self.tabView.setTabShape(QtWidgets.QTabWidget.Triangular)
            self.tabView.setIconSize(QtCore.QSize(28, 28))
            self.tabView.setDocumentMode(False)
            self.tabView.setTabsClosable(False)
            self.tabView.setMovable(False)
            self.tabView.setObjectName("tabView")
            self.preprocessing = QtWidgets.QWidget()
            self.preprocessing.setObjectName("preprocessing")
            self.lineEdit = QtWidgets.QLineEdit(self.preprocessing)
            self.lineEdit.setGeometry(QtCore.QRect(10, 30, 441, 31))
            self.lineEdit.setObjectName("lineEdit")
            self.pushButton = QtWidgets.QPushButton(self.preprocessing)
            self.pushButton.setGeometry(QtCore.QRect(450, 29, 101, 33))
            self.pushButton.setAutoDefault(False)
            self.pushButton.setDefault(False)
            self.pushButton.setFlat(False)
            self.pushButton.setObjectName("pushButton")
            self.label = QtWidgets.QLabel(self.preprocessing)
            self.label.setGeometry(QtCore.QRect(11, 10, 271, 16))
            self.label.setObjectName("label")
            self.plainTextEdit = QtWidgets.QPlainTextEdit(self.preprocessing)
            self.plainTextEdit.setGeometry(QtCore.QRect(10, 80, 541, 411))
            self.plainTextEdit.setObjectName("plainTextEdit")
            self.tabView.addTab(self.preprocessing, "")
            self.regression = QtWidgets.QWidget()
            self.regression.setObjectName("regression")
            self.lineEdit_2 = QtWidgets.QLineEdit(self.regression)
            self.lineEdit_2.setGeometry(QtCore.QRect(9, 30, 441, 31))
            self.lineEdit_2.setText("")
            self.lineEdit_2.setObjectName("lineEdit_2")
            self.pushButton_2 = QtWidgets.QPushButton(self.regression)
            self.pushButton_2.setGeometry(QtCore.QRect(449, 29, 101, 33))
            self.pushButton_2.setAutoDefault(False)
            self.pushButton_2.setDefault(False)
            self.pushButton_2.setFlat(False)
            self.pushButton_2.setObjectName("pushButton_2")
            self.label_2 = QtWidgets.QLabel(self.regression)
            self.label_2.setGeometry(QtCore.QRect(10, 10, 271, 16))
            self.label_2.setObjectName("label_2")
            self.comboBox = QtWidgets.QComboBox(self.regression)
            self.comboBox.setGeometry(QtCore.QRect(10, 120, 271, 31))
            self.comboBox.setObjectName("comboBox")
            self.comboBox.addItem("")
            self.comboBox.addItem("")
            self.comboBox.addItem("")
            self.comboBox.addItem("")
            self.comboBox.addItem("")
            self.comboBox.addItem("")
            self.label_4 = QtWidgets.QLabel(self.regression)
            self.label_4.setGeometry(QtCore.QRect(13, 90, 181, 16))
            self.label_4.setObjectName("label_4")
            self.tabView.addTab(self.regression, "")
            self.classification = QtWidgets.QWidget()
            self.classification.setObjectName("classification")
            self.lineEdit_3 = QtWidgets.QLineEdit(self.classification)
            self.lineEdit_3.setGeometry(QtCore.QRect(9, 30, 441, 31))
            self.lineEdit_3.setText("")
            self.lineEdit_3.setObjectName("lineEdit_3")
            self.pushButton_3 = QtWidgets.QPushButton(self.classification)
            self.pushButton_3.setGeometry(QtCore.QRect(449, 29, 101, 33))
            self.pushButton_3.setAutoDefault(False)
            self.pushButton_3.setDefault(False)
            self.pushButton_3.setFlat(False)
            self.pushButton_3.setObjectName("pushButton_3")
            self.regressionResult = QtWidgets.QPlainTextEdit(self.regression)
            self.regressionResult.setGeometry(QtCore.QRect(10, 170, 371, 291))
            self.regressionResult.setObjectName("regressionResult")

            self.graphicsView = QtWidgets.QGraphicsView(self.regression)
            self.graphicsView.setGeometry(QtCore.QRect(385, 170, 375, 291))
            self.graphicsView.setObjectName("graphicsView")

            self.pushButton_4 = QtWidgets.QPushButton(self.regression)
            self.pushButton_4.setGeometry(QtCore.QRect(290, 119, 91, 33))
            self.pushButton_4.setObjectName("pushButton_4")
            self.label_3 = QtWidgets.QLabel(self.classification)
            self.label_3.setGeometry(QtCore.QRect(10, 10, 271, 16))
            self.label_3.setObjectName("label_3")
            self.tabView.addTab(self.classification, "")
            self.clustering = QtWidgets.QWidget()
            self.clustering.setObjectName("clustering")
            self.tabView.addTab(self.clustering, "")
            self.association = QtWidgets.QWidget()
            self.association.setObjectName("association")
            self.tabView.addTab(self.association, "")
            self.verticalLayout.addWidget(self.tabView)
            MainWindow.setCentralWidget(self.centralwidget)
            self.menubar = QtWidgets.QMenuBar(MainWindow)
            self.menubar.setGeometry(QtCore.QRect(0, 0, 798, 26))
            self.menubar.setObjectName("menubar")
            MainWindow.setMenuBar(self.menubar)
            self.statusbar = QtWidgets.QStatusBar(MainWindow)
            self.statusbar.setObjectName("statusbar")
            MainWindow.setStatusBar(self.statusbar)

            self.retranslateUi(MainWindow)
            self.tabView.setCurrentIndex(1)
            QtCore.QMetaObject.connectSlotsByName(MainWindow)
            ############################## pushButton click events ##########################

            self.pushButton.clicked.connect(self.open)
            self.pushButton_2.clicked.connect(self.open)
            self.pushButton_3.clicked.connect(self.open)
            self.pushButton_4.clicked.connect(self.comboIndex)
            # self.d = self.comboBox.activated(0)
            # print(self.d)


        def retranslateUi(self, MainWindow):
            _translate = QtCore.QCoreApplication.translate
            MainWindow.setWindowTitle(_translate("MainWindow", "Machine Learning Python"))
            # self.tabView.setToolTip(_translate("MainWindow", "<html><head/><body><p><br/></p></body></html>"))
            self.pushButton.setText(_translate("MainWindow", "Open"))
            self.label.setText(_translate("MainWindow", "Load Training Data"))
            self.tabView.setTabText(self.tabView.indexOf(self.preprocessing), _translate("MainWindow", "Preprocessing"))
            self.pushButton_2.setText(_translate("MainWindow", "Open"))
            self.label_2.setText(_translate("MainWindow", "Load Test Data"))
            self.comboBox.setItemText(0, _translate("MainWindow", "Simple Linear Regression"))
            self.comboBox.setItemText(1, _translate("MainWindow", "Multi Linear Regression"))
            self.comboBox.setItemText(2, _translate("MainWindow", "Polynomial Regression"))
            self.comboBox.setItemText(3, _translate("MainWindow", "Support Vector Regression"))
            self.comboBox.setItemText(4, _translate("MainWindow", "Decision Tree Regression"))
            self.comboBox.setItemText(5, _translate("MainWindow", "random Forest Regression"))
            self.label_4.setText(_translate("MainWindow", "Select Algorithm"))
            self.tabView.setTabText(self.tabView.indexOf(self.regression), _translate("MainWindow", "Regression"))
            self.pushButton_3.setText(_translate("MainWindow", "Open"))
            self.pushButton_4.setText(_translate("MainWindow", "Process"))

            self.label_3.setText(_translate("MainWindow", "Load Test Data"))
            self.tabView.setTabText(self.tabView.indexOf(self.classification), _translate("MainWindow", "Classification"))
            self.tabView.setTabText(self.tabView.indexOf(self.clustering), _translate("MainWindow", "Clustering"))
            self.tabView.setTabText(self.tabView.indexOf(self.association), _translate("MainWindow", "Association"))

        def open(self):

            Tk().withdraw()
            self.file = askopenfilename()
            self.lineEdit.setText(self.file)
            data = pd.read_csv(self.file)
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values
            with open(f'text.txt', 'w') as file:
                f = file.write(f"{data}")
            with open('text.txt', 'r+') as r:
                rr = r.read()
                self.plainTextEdit.setPlainText(rr)
        def comboIndex(self):
            if self.comboBox.currentIndex() == 0:
                self.simpleLinear()
            if self.comboBox.currentIndex() == 1:
                self.multiLinear()
            if self.comboBox.currentIndex() == 2:
                self.polynomial()
            if self.comboBox.currentIndex() == 3:
                self.supportVector()
            if self.comboBox.currentIndex() == 4:
                self.decisionTree()
            if self.comboBox.currentIndex() == 5:
                self.randomForest()
        def plot_draw(self):
            scene = QtWidgets.QGraphicsScene()
            pixmap = QPixmap('polyplot.png')
            item = QtWidgets.QGraphicsPixmapItem(pixmap)
            scene.addItem(item)
            self.graphicsView.setScene(scene)
            # os.remove('polyplot.png')

        def simpleLinear(self):
            print("Simple Linear Regression")
            self.regressionResult.setPlainText("Simple Linear Regression -------")
            path = self.lineEdit.text()

            result = simple_linear_regression.data(path)
            print(result)
            self.plot_draw()
            self.regressionResult.setPlainText(f"Simple Linear Regression ------- \n{result}")
            #
            # msg = QtWidgets.QMessageBox()
            # # msg.setWindowIcon(QtGui.QIcon('logo.png'))
            # msg.setWindowTitle("Error")
            # msg.setIcon(QtWidgets.QMessageBox.Information)
            # msg.setText(f'LLR: {str(result)}')

            # msg.exec_()
        def multiLinear(self):
            print("Multi linear Regression")
            self.regressionResult.setPlainText("Multi Linear Regression -------")
            result = multiple_linear_regression.data(self.file)
            print(result)
            self.regressionResult.setPlainText(f"Multi Linear Regression ------- \n{result}")

        def polynomial(self):
            print("Polynomial Regression")
            self.regressionResult.setPlainText("Polynomial Regression -------")
            result = polinomial_linear_regression.data(self.file)
            print(result)
            self.regressionResult.setPlainText(f"Polynomial Regression ------- \n{result}")
            self.plot_draw()

            # self.regressionResult.setPlainText(f"Multi Linear Regression ------- \n{result}")

        def supportVector(self):
            print("Support Vector Regression")
            self.regressionResult.setPlainText("Support Vector Regression -------")
            result = support_vector_regression.data(self.file)
            print(result)
            self.regressionResult.setPlainText(f"Support Vector Regression ------- \n{result}")
            self.plot_draw()
        def decisionTree(self):
            print("Decison tree Regression")
            self.regressionResult.setPlainText("Decison tree Regression -------")
            result = decision_tree_regression.data(self.file)
            print(result)
            self.regressionResult.setPlainText(f"Decison tree Regression ------- \n{result}")
            self.plot_draw()
        def randomForest(self):
            print("Random forest Regression")
            self.regressionResult.setPlainText("Random forest Regression -------")
            result = random_forest_regression.data(self.file)
            print(result)
            self.regressionResult.setPlainText(f"Random forest Regression ------- \n{result}")
            self.plot_draw()
    except Exception as e:
        msg = QtWidgets.QMessageBox()
        # msg.setWindowIcon(QtGui.QIcon('logo.png'))
        msg.setWindowTitle("Error")
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText(f'LLR: {str(e)}')
        msg.exec_()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
