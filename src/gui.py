import os
import sys
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5 import QtCore, QtWidgets
import torch
import torchvision.utils as vutils
from torchvision.transforms.functional import to_tensor
from PIL import Image
from src.cyclegan.models.cycle_gan import CycleGAN

TOP = 100
LEFT = 100
WIDTH = 650
HEIGHT = 450

NAME = 'Style Transfer'
IMAGE_TO_CONVERT = None


def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create model
    model = CycleGAN(0, 0, True, device, 0, 0)

    # load_checkpoint(model, run, device)
    checkpoint = torch.load(os.path.join("..", "final_model.pth"), map_location=device)
    model.generator_A2B.load_state_dict(checkpoint.genA2B)
    model.generator_B2A.load_state_dict(checkpoint.genB2A)

    model = model.to(device)
    return model


def forward_model(model, path):
    input_image = to_tensor(Image.open(path))
    output_image = model(input_image)
    p = os.path.join("..", "outputs", "gui", "gui_result.png")
    vutils.save_image(output_image, p, normalize=True)
    return p


class Controller:

    def __init__(self):
        self.start_window = StartWindow()
        self.upload_window = UploadWindow()
        self.results_window = None

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
        self.upload_button = QtWidgets.QPushButton("Upload")
        self.upload_button.setFont(font)
        self.upload_button.setToolTip('This is load picture button')
        self.upload_button.clicked.connect(self.upload_button_on_click)
        self.setLayout(self.layout)
        self.layout.addWidget(self.upload_button)

        # Button Text
        self.convert_button = QtWidgets.QPushButton("Convert")
        self.convert_button.setFont(font)
        self.convert_button.setToolTip('Transfer the image style')
        self.convert_button.clicked.connect(self.convert_button_on_click)
        self.layout.addWidget(self.convert_button)

        self.setLayout(self.layout)

    def convert_button_on_click(self):
        self.switch_window.emit("resu")

    def upload_button_on_click(self):
        global IMAGE_TO_CONVERT

        imagePath, _ = QFileDialog.getOpenFileName(None, 'OpenFile', os.path.join("..", "assets"),
                                                   "Image file(*.jpg  *.png, *.jpeg)")
        pixmap = QPixmap(imagePath).scaled(256, 256)

        self.image.setPixmap(pixmap)
        self.image.adjustSize()

        self.upload_button.setText(imagePath)
        self.upload_button.setStyleSheet('QPushButton {color: green;}')

        IMAGE_TO_CONVERT = imagePath


class ResultsWindow(QtWidgets.QWidget):

    def __init__(self, text):
        QtWidgets.QWidget.__init__(self)
        self.setWindowTitle(NAME)
        self.setGeometry(TOP, LEFT, WIDTH, HEIGHT)

        self.layout = QtWidgets.QGridLayout()

        # font
        font = self.font()
        font.setPointSize(16)

        # Title
        self.text = QtWidgets.QLabel("Results", self)
        self.text.setAlignment(QtCore.Qt.AlignHCenter)
        self.text.setFont(font)
        self.layout.addWidget(self.text)

        # image
        self.image1 = QtWidgets.QLabel(self)
        self.image1.move(50, 100)
        pixmap = QPixmap(IMAGE_TO_CONVERT).scaled(256, 256)
        self.image1.setPixmap(pixmap)
        self.image1.adjustSize()

        pic2_offset = 256 + 100
        self.image1 = QtWidgets.QLabel(self)
        self.image1.move(pic2_offset, 100)

        model = load_model()
        path = forward_model(model, IMAGE_TO_CONVERT)

        pixmap = QPixmap(path).scaled(256, 256)
        self.image1.setPixmap(pixmap)
        self.image1.adjustSize()

        font.setPointSize(11)
        self.button = QtWidgets.QPushButton('Close')
        self.button.setFont(font)
        self.button.clicked.connect(self.close)
        self.layout.addWidget(self.button)

        self.setLayout(self.layout)


def main():
    app = QtWidgets.QApplication(sys.argv)
    icon_path = os.path.join("..", "media", "logo.png")
    app.setWindowIcon(QIcon(icon_path))
    controller = Controller()
    controller.show_start_window()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
