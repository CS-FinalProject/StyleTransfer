import sys
import random
from PySide6 import QtCore, QtWidgets, QtGui
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *


# class Window(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.title = 'Style Transfer'
#         self.top = 100
#         self.left = 100
#         self.width = 680
#         self.height = 500
#         self.opening_frame()
#
#     def opening_frame(self):
#         self.setWindowTitle(self.title)
#         self.setGeometry(self.top, self.left, self.width, self.height)
#
#         button_window1 = QPushButton("Let's Start", self)
#         button_window1.move(100, 100)
#         button_window1.clicked.connect(self.button_let_start_on_click)
#         self.lineEdit1 = QLineEdit("Type here what you want to transfer for [Window1].", self)
#         self.lineEdit1.setGeometry(250, 100, 400, 30)
#
#         buttonWindow2 = QPushButton('Window2', self)
#         buttonWindow2.move(100, 200)
#         buttonWindow2.clicked.connect(self.button_upload_on_click)
#         self.lineEdit2 = QLineEdit("Type here what you want to transfer for [Window2].", self)
#         self.lineEdit2.setGeometry(250, 200, 400, 30)
#         self.show()
#
#     @pyqtSlot()
#     def button_let_start_on_click(self):
#         self.statusBar().showMessage("Switched to window 1")
#         self.cams = Window1(self.lineEdit1.text())
#         self.cams.show()
#         self.close()
#
#     @pyqtSlot()
#     def button_upload_on_click(self):
#         self.statusBar().showMessage("Switched to window 2")
#         self.cams = Window2(self.lineEdit2.text())
#         self.cams.show()
#         self.close()


class StartWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Style Transfer'
        self.top = 100
        self.left = 100
        self.width = 800
        self.height = 600
        self.setWindowTitle(self.title)
        self.setGeometry(self.top, self.left, self.width, self.height)

        self.hello = ["Hallo Welt", "Hei maailma", "Hola Mundo", "Привет мир"]

        self.start_button = QtWidgets.QPushButton("Let's Start")
        self.text = QtWidgets.QLabel('Welcome to the Style Transfer application!!!', alignment=QtCore.Qt.AlignCenter)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.text)
        self.layout.addWidget(self.start_button)

        self.start_button.clicked.connect(self.button_let_start_on_click)

    @QtCore.Slot()
    def button_let_start_on_click(self):
        # self.statusBar().showMessage("Switched to window 1")
        self.cams = Window1("wi")
        self.cams.show()
        self.close()


class Window1(QDialog):
    def __init__(self, value, parent=None):
        super().__init__(parent)
        self.title = 'Style Transfer'
        self.top = 100
        self.left = 100
        self.width = 800
        self.height = 600
        self.setWindowTitle(self.title)
        self.setGeometry(self.top, self.left, self.width, self.height)

        self.hello = ["Hallo Welt", "Hei maailma", "Hola Mundo", "Привет мир"]

        self.start_button = QtWidgets.QPushButton("Let's Start")
        self.text = QtWidgets.QLabel('Welcome to the Style Transfer application!!!', alignment=QtCore.Qt.AlignCenter)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.text)
        self.layout.addWidget(self.start_button)

        self.start_button.clicked.connect(self.button_let_start_on_click)

    def button_upload_image_on_click(self):
        self.cams = StartWindow()
        self.cams.show()
        self.close()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    start_window = StartWindow()
    start_window.show()

    sys.exit(app.exec())

# import sys
# from PyQt5.QtGui import *
# from PyQt5.QtCore import *
# from PyQt5.QtWidgets import *
#
#
# class Window(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.title = "App"
#         self.top = 100
#         self.left = 100
#         self.width = 680
#         self.height = 500
#         self.opening_frame()
#
#     def opening_frame(self):
#         self.setWindowTitle(self.title)
#         self.setGeometry(self.top, self.left, self.width, self.height)
#
#         label1 = QLabel('Welcome to the Style Transfer application!!!')
#
#         layoutV = QVBoxLayout()
#         button_window1 = QPushButton("Let's Start", self)
#         button_window1.move(100, 100)
#         # button_window1.setStyleSheet('background-color: rgb(0,0,255); color: #fff')
#         # button_window1.setText("Let's Start")
#         button_window1.clicked.connect(self.button_let_start_on_click)
#         layoutV.addWidget(button_window1)
#
#         layoutH = QHBoxLayout()
#         layoutH.addWidget(label1)
#         layoutH.addWidget(button_window1)
#         layoutV.addLayout(layoutH)
#         self.setLayout(layoutV)
#
#
#         #
#         # buttonWindow1.move(100, 100)
#         # buttonWindow1.clicked.connect(self.button_let_start_on_click)
#         # self.lineEdit1 = QLineEdit("Type here what you want to transfer for [Window1].", self)
#         # self.lineEdit1.setGeometry(250, 100, 400, 30)
#         #
#         # buttonWindow2 = QPushButton('Window2', self)
#         # buttonWindow2.move(100, 200)
#         # buttonWindow2.clicked.connect(self.button_upload_on_click)
#         # self.lineEdit2 = QLineEdit("Type here what you want to transfer for [Window2].", self)
#         # self.lineEdit2.setGeometry(250, 200, 400, 30)
#         # self.show()
#
#     @pyqtSlot()
#     def button_let_start_on_click(self):
#         self.statusBar().showMessage("Switched to window 1")
#         self.cams = Window1(self.lineEdit1.text())
#         self.cams.show()
#         self.close()
#
#     @pyqtSlot()
#     def button_upload_on_click(self):
#         self.statusBar().showMessage("Switched to window 2")
#         self.cams = Window2(self.lineEdit2.text())
#         self.cams.show()
#         self.close()
#
#
# class Window1(QDialog):
#     def __init__(self, value, parent=None):
#         super().__init__(parent)
#         self.setWindowTitle('Window1')
#         self.setWindowIcon(self.style().standardIcon(QStyle.SP_FileDialogInfoView))
#
#         label1 = QLabel(value)
#         self.button = QPushButton()
#         self.button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
#         self.button.setIcon(self.style().standardIcon(QStyle.SP_ArrowLeft))
#         self.button.setIconSize(QSize(200, 200))
#
#         layoutV = QVBoxLayout()
#         self.pushButton = QPushButton(self)
#         self.pushButton.setStyleSheet('background-color: rgb(0,0,255); color: #fff')
#         self.pushButton.setText('Click me!')
#         self.pushButton.clicked.connect(self.goMainWindow)
#         layoutV.addWidget(self.pushButton)
#
#         layoutH = QHBoxLayout()
#         layoutH.addWidget(label1)
#         layoutH.addWidget(self.button)
#         layoutV.addLayout(layoutH)
#         self.setLayout(layoutV)
#
#     def goMainWindow(self):
#         self.cams = Window()
#         self.cams.show()
#         self.close()
#
#
# class Window2(QDialog):
#     def __init__(self, value, parent=None):
#         super().__init__(parent)
#         self.setWindowTitle('Window2')
#         self.setWindowIcon(self.style().standardIcon(QStyle.SP_FileDialogInfoView))
#
#         label1 = QLabel(value)
#         self.button = QPushButton()
#         self.button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
#         self.button.setIcon(self.style().standardIcon(QStyle.SP_ArrowLeft))
#         self.button.setIconSize(QSize(200, 200))
#
#         layoutV = QVBoxLayout()
#         self.pushButton = QPushButton(self)
#         self.pushButton.setStyleSheet('background-color: rgb(0,0,255); color: #fff')
#         self.pushButton.setText('Click me!')
#         self.pushButton.clicked.connect(self.goMainWindow)
#         layoutV.addWidget(self.pushButton)
#
#         layoutH = QHBoxLayout()
#         layoutH.addWidget(label1)
#         layoutH.addWidget(self.button)
#         layoutV.addLayout(layoutH)
#         self.setLayout(layoutV)
#
#     def goMainWindow(self):
#         self.cams = Window()
#         self.cams.show()
#         self.close()
#
#
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     ex = Window()
#     sys.exit(app.exec_())
