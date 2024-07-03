from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
import cv2
import numpy as np
from predict import main as img_predict
from plant_name import name_list

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1038, 621)
        MainWindow.setStyleSheet("background-color: rgb(118, 192, 139)")
        # MainWindow.setStyleSheet("background-image: url(1/1.jpg);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(320, 30, 491, 101))
        self.label.setStyleSheet("font: 44pt \"微软雅黑\";\n"
"color: rgb(255, 255, 255);")
        self.label.setObjectName("label")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(920, 30, 31, 21))
        self.label_4.setStyleSheet("image: url(1/Wi-Fi.png)")
        self.label_4.setText("")
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(30, 70, 81, 51))
        self.label_5.setStyleSheet("font: 35pt \"微软雅黑\";\n"
"color: rgb(255, 255, 255);")
        self.label_5.setObjectName("label_5")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(30, 20, 121, 31))
        self.label_7.setStyleSheet("font: 15pt \"微软雅黑\";\n"
"color: rgb(255, 255, 255);")
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(130, 100, 51, 21))
        self.label_8.setStyleSheet("font: 12pt \"微软雅黑\";\n"
"color: rgb(255, 255, 255);")
        self.label_8.setObjectName("label_8")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(570, 170, 451, 411))
        self.label_11.setStyleSheet("image:url(1/白框区域.png);")
        self.label_11.setText("")
        self.label_11.setObjectName("label_11")
        self.label_25 = QtWidgets.QLabel(self.centralwidget)
        self.label_25.setGeometry(QtCore.QRect(10, 160, 541, 431))
        self.label_25.setStyleSheet("color: rgb(255, 255, 255);\n"
"font: 36pt \"微软雅黑\";\n"
"image: url(1/显示背景.png);\n"
"")
        self.label_25.setText("")
        self.label_25.setAlignment(QtCore.Qt.AlignCenter)
        self.label_25.setObjectName("label_25")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(50, 200, 461, 351))
        self.label_13.setStyleSheet("background-color: transparent;\n"
"background-color: teansparent;")
        self.label_13.setText("")
        self.label_13.setObjectName("label_13")
        self.label_15 = QtWidgets.QLabel(self.centralwidget)
        self.label_15.setGeometry(QtCore.QRect(770, 310, 71, 31))
        self.label_15.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"font: 75 12pt \"Arial\";")
        self.label_15.setObjectName("label_15")
        self.label_16 = QtWidgets.QLabel(self.centralwidget)
        self.label_16.setGeometry(QtCore.QRect(770, 360, 91, 31))
        self.label_16.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"font: 75 10pt \"Arial\";")
        self.label_16.setObjectName("label_16")
        self.label_name = QtWidgets.QLabel(self.centralwidget)
        self.label_name.setGeometry(QtCore.QRect(890, 280, 91, 31))
        self.label_name.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"font: 16pt \"Arial\";")
        self.label_name.setText("")
        self.label_name.setAlignment(QtCore.Qt.AlignCenter)
        self.label_name.setObjectName("label_name")
        self.label_type = QtWidgets.QLabel(self.centralwidget)
        self.label_type.setGeometry(QtCore.QRect(870, 360, 91, 31))
        self.label_type.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"font: 10pt \"Arial\";\n"
"color: rgb(255, 0, 0);")
        self.label_type.setAlignment(QtCore.Qt.AlignCenter)
        self.label_type.setObjectName("label_type")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(960, 30, 41, 21))
        self.label_6.setStyleSheet("image: url(1/battery.png);")
        self.label_6.setText("")
        self.label_6.setObjectName("label_6")
        self.label_type_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_type_2.setGeometry(QtCore.QRect(620, 210, 161, 51))
        self.label_type_2.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"font: 75 12pt \"Arial\";")
        self.label_type_2.setObjectName("label_type_2")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(610, 290, 371, 1))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(880, 190, 111, 71))
        self.pushButton.setStyleSheet("image: url(1/文件夹.png);\n"
"background-color: transparent;")
        self.pushButton.setText("")
        self.pushButton.setObjectName("pushButton")
        self.label_type_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_type_3.setGeometry(QtCore.QRect(870, 310, 91, 31))
        self.label_type_3.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"font: 10pt \"Arial\";\n"
"color: rgb(255, 0, 0);")
        self.label_type_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_type_3.setObjectName("label_type_3")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(640, 330, 81, 41))
        self.pushButton_2.setStyleSheet("background-color: rgb(218, 218, 218);")
        self.pushButton_2.setObjectName("pushButton_2")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(620, 400, 361, 161))
        self.textEdit.setStyleSheet("background-color: rgb(221, 221, 221);")
        self.textEdit.setObjectName("textEdit")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(920, 80, 71, 61))
        self.pushButton_3.setStyleSheet("image: url(1/首页.png);\n"
                                        "background-color: transparent;\n"
                                        "")
        self.pushButton_3.setText("")
        self.pushButton_3.setObjectName("pushButton_3")
        self.label_25.raise_()
        self.label_11.raise_()
        self.label.raise_()
        self.label_4.raise_()
        self.label_5.raise_()
        self.label_7.raise_()
        self.label_8.raise_()
        self.label_13.raise_()
        self.label_15.raise_()
        self.label_16.raise_()
        self.label_name.raise_()
        self.label_type.raise_()
        self.label_6.raise_()
        self.label_type_2.raise_()
        self.line.raise_()
        self.pushButton.raise_()
        self.label_type_3.raise_()
        self.pushButton_2.raise_()
        self.textEdit.raise_()
        self.pushButton_3.raise_()
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.pushButton.clicked.connect(self.select_img)
        self.pushButton_2.clicked.connect(self.detect_img)
    def select_img(self):
        self.img_path, _ = QFileDialog.getOpenFileName(None, 'open img', '', "*.png;*.jpg;;All Files(*)")
        print(self.img_path)
        # img = cv2.imread(self.img_path)
        img = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), -1)
        input_img = cv2.resize(img, (461, 351))
        cv2.imwrite('resize_img.png', input_img)
        self.label_13.setStyleSheet("image: url(./resize_img.png)")
    def detect_img(self):
        img = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), -1)
        small_img = cv2.resize(img, (177, 177))
        cv2.imwrite('resize_smallimg.png',small_img)
        res,num = img_predict(self.img_path)
        print('识别结果为：' + res)
        name = name_list[int(res.split('_')[-1])]
        from plant_data import plant_inf
        information = plant_inf[name]
        self.textEdit.setText(information)
        self.label_type_3.setText(name)
        self.label_type.setText(str(round(num,3)))



    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "植物/中草药识别"))
        self.label_5.setText(_translate("MainWindow", "5"))
        self.label_7.setText(_translate("MainWindow", "2023/12"))
        self.label_8.setText(_translate("MainWindow", "星期二"))
        self.label_15.setText(_translate("MainWindow", "类别："))
        self.label_16.setText(_translate("MainWindow", "置信度："))
        self.label_type.setText(_translate("MainWindow", ""))
        self.label_type_2.setText(_translate("MainWindow", "识别结果"))
        self.label_type_3.setText(_translate("MainWindow", ""))
        self.pushButton_2.setText(_translate("MainWindow", "识别"))
        self.textEdit.setHtml(_translate("MainWindow",
                                         "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                         "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                         "p, li { white-space: pre-wrap; }\n"
                                         "</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
                                         "<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())