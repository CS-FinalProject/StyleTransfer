import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QFileDialog, QPushButton
from PyQt5.QtGui import QIcon, QPixmap
# from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtCore, QtWidgets

TOP = 100
LEFT = 100
WIDTH = 650
HEIGHT = 450

NAME = 'Style Transfer'


class Controller:

    def __init__(self):
        self.start_window = StartWindow()
        self.upload_window = UploadWindow()
        self.results_window = None

    # self.results_window = ResultsWindow()
    # self.finish_window = FinishWindow()

    def show_start_window(self):
        self.start_window.switch_window.connect(self.show_upload_window)
        self.start_window.show()

    def show_upload_window(self):
        self.upload_window.switch_window.connect(self.show_results_window)
        self.start_window.close()
        self.upload_window.show()

    def show_results_window(self, text):
        self.results_window = ResultsWindow(text)
        self.upload_window.close()
        self.results_window.show()

    # def show_results_window(self):
    #     self.results_window.switch_window.connect(self.show_finish_window)
    #     self.upload_window.close()
    #     self.results_window.show()
    #
    # def show_finish_window(self):
    #     self.finish_window = FinishWindow()
    #     self.results_window.close()
    #     self.finish_window.show()


class StartWindow(QtWidgets.QWidget):
    switch_window = QtCore.pyqtSignal()

    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.setWindowTitle(NAME)
        self.setGeometry(TOP, LEFT, WIDTH, HEIGHT)

        # font
        font = self.font()
        font.setPointSize(12)

        self.layout = QtWidgets.QGridLayout()

        # Button Text
        self.start_button = QtWidgets.QPushButton("Let's Start")
        self.start_button.setFont(font)
        self.start_button.clicked.connect(self.button_let_start_on_click)
        self.setLayout(self.layout)

        font.setPointSize(18)

        # Application Text
        self.text = QtWidgets.QLabel('Welcome to the Style Transfer application!!!', self)
        self.text.setAlignment(QtCore.Qt.AlignCenter)
        self.text.setFont(font)
        self.layout.addWidget(self.text)
        self.layout.addWidget(self.start_button)

    def button_let_start_on_click(self):
        self.switch_window.emit()


class UploadWindow(QtWidgets.QWidget):
    switch_window = QtCore.pyqtSignal(str)

    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.setWindowTitle(NAME)
        self.setGeometry(TOP, LEFT, WIDTH, HEIGHT)

        self.layout = QtWidgets.QGridLayout()

        self.image = QtWidgets.QLabel(self)
        self.image.move(WIDTH // 2 - 128, 50)

        # font
        font = self.font()
        font.setPointSize(16)

        # Application Text
        self.text = QtWidgets.QLabel("Let's Choose Your Image To Transform", self)
        self.text.setAlignment(QtCore.Qt.AlignHCenter)
        self.text.setFont(font)
        self.layout.addWidget(self.text)

        # Upload Button Text
        font.setPointSize(11)
        self.upload_button = QtWidgets.QPushButton("Upload Button")
        self.upload_button.setFont(font)
        self.upload_button.setToolTip('This is load picture button')
        self.upload_button.clicked.connect(self.upload_button_on_click)
        self.setLayout(self.layout)
        self.layout.addWidget(self.upload_button)

        # Choose from sample Button Text
        font.setPointSize(11)
        self.choose_button = QtWidgets.QPushButton("Choose Button")
        self.choose_button.setFont(font)
        self.choose_button.setToolTip('This is for choosing image from the dataset samples')
        self.choose_button.clicked.connect(self.choose_button_on_click)
        self.setLayout(self.layout)
        self.layout.addWidget(self.choose_button)

        # Button Text
        self.convert_button = QtWidgets.QPushButton("Convert Button")
        self.convert_button.setFont(font)
        self.convert_button.setToolTip('Transfer the image style')
        self.convert_button.clicked.connect(self.convert_button_on_click)
        self.layout.addWidget(self.convert_button)

        self.setLayout(self.layout)

    def convert_button_on_click(self):
        self.switch_window.emit("resu")

    def choose_button_on_click(self):
        pass

    def upload_button_on_click(self):
        imagePath, _ = QFileDialog.getOpenFileName(None, 'OpenFile', '', "Image file(*.jpg  *.png, *.jpeg)")
        pixmap = QPixmap(imagePath).scaled(256, 256)

        self.image.setPixmap(pixmap)
        self.image.adjustSize()

        self.upload_button.setText(imagePath)
        self.upload_button.setStyleSheet('QPushButton {color: green;}')

        print(imagePath)


class ResultsWindow(QtWidgets.QWidget):

    def __init__(self, text):
        QtWidgets.QWidget.__init__(self)
        self.setWindowTitle(NAME)
        self.setGeometry(TOP, LEFT, WIDTH, HEIGHT)

        layout = QtWidgets.QGridLayout()

        self.label = QtWidgets.QLabel(text)
        layout.addWidget(self.label)

        self.button = QtWidgets.QPushButton('Close')
        self.button.clicked.connect(self.close)

        layout.addWidget(self.button)

        self.setLayout(layout)


def main():
    app = QtWidgets.QApplication(sys.argv)
    controller = Controller()
    controller.show_start_window()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()


