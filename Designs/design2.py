# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled2.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ReaultW(object):
    def setupUi(self, ReaultW):
        ReaultW.setObjectName("ReaultW")
        ReaultW.resize(557, 664)
        self.centralwidget = QtWidgets.QWidget(ReaultW)
        self.centralwidget.setObjectName("centralwidget")
        self.result = QtWidgets.QLabel(self.centralwidget)
        self.result.setGeometry(QtCore.QRect(20, 70, 512, 512))
        self.result.setText("")
        self.result.setObjectName("result")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 30, 151, 16))
        self.label.setObjectName("label")
        self.SaveB = QtWidgets.QPushButton(self.centralwidget)
        self.SaveB.setGeometry(QtCore.QRect(330, 600, 93, 28))
        self.SaveB.setObjectName("SaveB")
        self.CloseB = QtWidgets.QPushButton(self.centralwidget)
        self.CloseB.setGeometry(QtCore.QRect(430, 600, 93, 28))
        self.CloseB.setObjectName("CloseB")
        ReaultW.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(ReaultW)
        self.statusbar.setObjectName("statusbar")
        ReaultW.setStatusBar(self.statusbar)

        self.retranslateUi(ReaultW)
        QtCore.QMetaObject.connectSlotsByName(ReaultW)

    def retranslateUi(self, ReaultW):
        _translate = QtCore.QCoreApplication.translate
        ReaultW.setWindowTitle(_translate("ReaultW", "MainWindow"))
        self.label.setText(_translate("ReaultW", "Идет обработка"))
        self.SaveB.setText(_translate("ReaultW", "Сохранить"))
        self.CloseB.setText(_translate("ReaultW", "Отмена"))
