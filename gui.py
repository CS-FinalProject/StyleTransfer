# import sys
# from PyQt5 import QtCore, QtWidgets
#
# TOP = 100
# LEFT = 100
# WIDTH = 800
# HEIGHT = 600
#
# NAME = 'Style Transfer'
#
#
# class Controller:
#
#     def __init__(self):
#         self.start_window = StartWindow()
#         self.upload_window = UploadWindow()
#         self.results_window = ResultsWindow()
#         self.finish_window = FinishWindow()
#
#     def show_start_window(self):
#         self.start_window.switch_window.connect(self.show_upload_window)
#         self.start_window.show()
#
#     def show_upload_window(self):
#         self.upload_window.switch_window.connect(self.show_results_window)
#         self.start_window.close()
#         self.upload_window.show()
#
#     def show_results_window(self):
#         self.results_window.switch_window.connect(self.show_finish_window)
#         self.upload_window.close()
#         self.results_window.show()
#
#     def show_finish_window(self):
#         self.finish_window = FinishWindow()
#         self.results_window.close()
#         self.finish_window.show()
#
#
# class StartWindow(QtWidgets.QWidget):
#     switch_window = QtCore.pyqtSignal()
#
#     def __init__(self):
#         QtWidgets.QWidget.__init__(self)
#         self.setWindowTitle(NAME)
#         self.setGeometry(TOP, LEFT, WIDTH, HEIGHT)
#
#         # font
#         font = self.font()
#         font.setPointSize(12)
#
#         self.layout = QtWidgets.QGridLayout()
#
#         # Button Text
#         self.start_button = QtWidgets.QPushButton("Let's Start")
#         self.start_button.setFont(font)
#         self.start_button.clicked.connect(self.button_let_start_on_click)
#         self.setLayout(self.layout)
#
#         font.setPointSize(18)
#
#         # Application Text
#         self.text = QtWidgets.QLabel('Welcome to the Style Transfer application!!!', self)
#         self.text.setAlignment(QtCore.Qt.AlignCenter)
#         self.text.setFont(font)
#         self.layout.addWidget(self.text)
#         self.layout.addWidget(self.start_button)
#
#     def button_let_start_on_click(self):
#         self.switch_window.emit()
#
#
# class UploadWindow(QtWidgets.QWidget):
#     switch_window = QtCore.pyqtSignal(str)
#
#     def __init__(self):
#         QtWidgets.QWidget.__init__(self)
#         self.setWindowTitle(NAME)
#         self.setGeometry(TOP, LEFT, WIDTH, HEIGHT)
#
#         # font
#         font = self.font()
#         font.setPointSize(12)
#
#         self.layout = QtWidgets.QGridLayout()
#
#         # Button Text
#         self.convert_button = QtWidgets.QPushButton("Convert Button")
#         self.convert_button.setFont(font)
#         self.convert_button.clicked.connect(self.convert_button_on_click)
#         self.setLayout(self.layout)
#
#         font.setPointSize(11)
#
#         # Application Text
#         self.text = QtWidgets.QLabel("Let's Choose Your Image To Transform", self)
#         self.text.setAlignment(QtCore.Qt.AlignTop)
#         self.text.setFont(font)
#         self.layout.addWidget(self.text)
#         self.layout.addWidget(self.convert_button)
#
#         # QtWidgets.QWidget.__init__(self)
#         # self.setWindowTitle(NAME)
#         # self.setGeometry(TOP, LEFT, WIDTH, HEIGHT)
#         #
#         # self.layout = QtWidgets.QGridLayout()
#         #
#         # # font
#         # font = self.font()
#         # font.setPointSize(10)
#         #
#         # # Application Text
#         # self.text = QtWidgets.QLabel("Let's Choose Your Image To Transform", self)
#         # self.text.setAlignment(QtCore.Qt.AlignTop)
#         # self.text.setFont(font)
#         # self.layout.addWidget(self.text)
#         # # self.layout.addWidget(self.start_button)
#         #
#         # # # Upload Button Text
#         # # font.setPointSize(11)
#         # # self.upload_button = QtWidgets.QPushButton("Upload Button")
#         # # self.upload_button.setFont(font)
#         # # self.upload_button.clicked.connect(self.upload_button_on_click)
#         # # self.setLayout(self.layout)
#         # # self.layout.addWidget(self.upload_button)
#         # #
#         # # # Choose from sample Button Text
#         # # font.setPointSize(11)
#         # # self.choose_button = QtWidgets.QPushButton("Choose Button")
#         # # self.choose_button.setFont(font)
#         # # self.choose_button.clicked.connect(self.choose_button_on_click)
#         # # self.setLayout(self.layout)
#         # # self.layout.addWidget(self.choose_button)
#         #
#         # # Convert Button Text
#         # font.setPointSize(11)
#         # self.convert_button = QtWidgets.QPushButton("Convert Button")
#         # self.convert_button.setFont(font)
#         # self.convert_button.clicked.connect(self.convert_button_on_click)
#         # self.setLayout(self.layout)
#         # self.layout.addWidget(self.convert_button)
#         #
#         # # self.line_edit = QtWidgets.QLineEdit()
#         # # self.layout.addWidget(self.line_edit)
#         #
#         # # self.button = QtWidgets.QPushButton('Switch Window')
#         # # self.button.clicked.connect(self.switch)
#         # # self.layout.addWidget(self.button)
#         # #
#         # self.setLayout(self.layout)
#
#     def convert_button_on_click(self):
#         self.switch_window.emit()
#
#     def choose_button_on_click(self):
#         pass
#         # self.switch_window.emit()
#
#     def upload_button_on_click(self):
#         pass
#         # self.switch_window.emit()
#
#
# class ResultsWindow(QtWidgets.QWidget):
#     switch_window = QtCore.pyqtSignal()
#
#     def __init__(self, text=None):
#         QtWidgets.QWidget.__init__(self)
#         self.setWindowTitle(NAME)
#         self.finish = False
#
#         layout = QtWidgets.QGridLayout()
#
#         self.label = QtWidgets.QLabel(text)
#         layout.addWidget(self.label)
#
#         self.button = QtWidgets.QPushButton('Close')
#         self.button.clicked.connect(self.close)
#
#         layout.addWidget(self.button)
#
#         self.setLayout(layout)
#
#     def is_finish(self):
#         return self.finish
#
#
# class FinishWindow(QtWidgets.QWidget):
#
#     def __init__(self, text=None):
#         QtWidgets.QWidget.__init__(self)
#         self.setWindowTitle(NAME)
#         self.finish = False
#
#         layout = QtWidgets.QGridLayout()
#
#         self.label = QtWidgets.QLabel(text)
#         layout.addWidget(self.label)
#
#         self.button = QtWidgets.QPushButton('Close')
#         self.button.clicked.connect(self.close)
#
#         layout.addWidget(self.button)
#
#         self.setLayout(layout)
#
#
# def main():
#     app = QtWidgets.QApplication(sys.argv)
#     controller = Controller()
#     controller.show_start_window()
#     sys.exit(app.exec_())
#
#
# if __name__ == '__main__':
#     main()
#
#


import sys
from PyQt5 import QtCore, QtWidgets


class WindowTwo(QtWidgets.QWidget):

    def __init__(self, text):
        QtWidgets.QWidget.__init__(self)
        self.setWindowTitle('Window Two')

        layout = QtWidgets.QGridLayout()

        self.label = QtWidgets.QLabel(text)
        layout.addWidget(self.label)

        self.button = QtWidgets.QPushButton('Close')
        self.button.clicked.connect(self.close)

        layout.addWidget(self.button)

        self.setLayout(layout)


TOP = 100
LEFT = 100
WIDTH = 800
HEIGHT = 600

NAME = 'Style Transfer'


class Controller:

    def __init__(self):
        self.start_window = StartWindow()
        self.upload_window = UploadWindow()
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
        self.window_two = WindowTwo(text)
        self.upload_window.close()
        self.window_two.show()

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

        # font
        font = self.font()
        font.setPointSize(10)

        # Application Text
        self.text = QtWidgets.QLabel("Let's Choose Your Image To Transform", self)
        self.text.setAlignment(QtCore.Qt.AlignTop)
        self.text.setFont(font)
        self.layout.addWidget(self.text)

        # Upload Button Text
        font.setPointSize(11)
        self.upload_button = QtWidgets.QPushButton("Upload Button")
        self.upload_button.setFont(font)
        self.upload_button.clicked.connect(self.upload_button_on_click)
        self.setLayout(self.layout)
        self.layout.addWidget(self.upload_button)

        # Choose from sample Button Text
        font.setPointSize(11)
        self.choose_button = QtWidgets.QPushButton("Choose Button")
        self.choose_button.setFont(font)
        self.choose_button.clicked.connect(self.choose_button_on_click)
        self.setLayout(self.layout)
        self.layout.addWidget(self.choose_button)

        # Button Text
        self.convert_button = QtWidgets.QPushButton("Convert Button")
        self.convert_button.setFont(font)
        self.convert_button.clicked.connect(self.convert_button_on_click)
        self.layout.addWidget(self.convert_button)

        self.setLayout(self.layout)

    def convert_button_on_click(self):
        self.switch_window.emit("resu")

    def choose_button_on_click(self):
        pass

    def upload_button_on_click(self):
        pass


def main():
    app = QtWidgets.QApplication(sys.argv)
    controller = Controller()
    controller.show_start_window()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
