import sys

# GUI use case
from PyQt5.QtWidgets import QDesktopWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QDialog
from PyQt5.QtWidgets import QPushButton, QGroupBox, QLineEdit, QApplication, QMessageBox
from PyQt5.QtGui import QDoubleValidator, QIntValidator, QFont
from PyQt5.QtCore import Qt, pyqtSlot

from quizes import *

class App(QDialog):

    def __init__(self):
        super().__init__()
        self.initUI()
    
    def center(self):
        frameGm = self.frameGeometry()
        screen = QApplication.desktop().screenNumber(QApplication.desktop().cursor().pos())
        centerPoint = QApplication.desktop().screenGeometry(screen).center()
        frameGm.moveCenter(centerPoint)
        self.move(frameGm.topLeft())

    def initUI(self):
        self.setWindowTitle("Homework 1")
        self.createFirst3Quiz()
        self.createLast2Quiz()
        
        windowLayout = QVBoxLayout()
        windowLayout.addLayout(self.horizontalGroupBoxes1)
        windowLayout.addLayout(self.horizontalGroupBoxes2)
        self.setLayout(windowLayout)
        
        self.center()
        self.show()
    
    def testPrint(self):
        print('test')
        
    def createQuiz1(self):
        self.quiz1 = Quiz1()

        self.quiz1groupbox = QGroupBox("1. Image Processing")
        quiz1_layout = QVBoxLayout()
        quiz1_layout.setAlignment(Qt.AlignTop)
        
        show_img_btn = QPushButton('1.1 Load Image')
        show_img_btn.clicked.connect(self.quiz1.showImage)
        quiz1_layout.addWidget(show_img_btn)

        color_conversion_btn = QPushButton('1.2 Color Conversion')
        color_conversion_btn.clicked.connect(self.quiz1.colorConversion)
        quiz1_layout.addWidget(color_conversion_btn)

        image_flip_btn = QPushButton('1.3 Image Flipping')
        image_flip_btn.clicked.connect(self.quiz1.imageFlipping)
        quiz1_layout.addWidget(image_flip_btn)

        image_blend_btn = QPushButton('1.4 Blending')
        image_blend_btn.clicked.connect(self.quiz1.imageBlending)
        quiz1_layout.addWidget(image_blend_btn)

        self.quiz1groupbox.setLayout(quiz1_layout)

    def createQuiz2(self):
        self.quiz2 = Quiz2()

        self.quiz2groupbox = QGroupBox("2. Adaptive Threshold")
        quiz2_layout = QVBoxLayout()
        quiz2_layout.setAlignment(Qt.AlignTop)

        global_thresh_btn = QPushButton('2.1 Global Threshold')
        global_thresh_btn.clicked.connect(self.quiz2.globalThreshold)
        quiz2_layout.addWidget(global_thresh_btn)

        local_thresh_btn = QPushButton('2.2 Local Threshold')
        local_thresh_btn.clicked.connect(self.quiz2.localThreshold)
        quiz2_layout.addWidget(local_thresh_btn)

        self.quiz2groupbox.setLayout(quiz2_layout)

    def createQuiz3(self):

        self.quiz3groupbox = QGroupBox("3. Transforms")
        quiz3_layout = QVBoxLayout()

        quiz_3_1_gb = QGroupBox("3.1 Rotation, Scale, Translate")
        quiz_3_1_gb_layout = QVBoxLayout()

        quiz_3_1_sub_gb = QGroupBox("Parameter")
        quiz_3_1_sub_gb_layout = QFormLayout()

        angle = QLineEdit()
        angle.setValidator(QDoubleValidator())
        angle.setText('45')
        angle.setFont(QFont("Arial",10))

        scale = QLineEdit()
        scale.setValidator(QDoubleValidator())
        scale.setText('0.8')
        scale.setFont(QFont("Arial",10))

        tx = QLineEdit()
        tx.setValidator(QIntValidator())
        tx.setText('150')
        tx.setFont(QFont("Arial",10))

        ty = QLineEdit()
        ty.setValidator(QIntValidator())
        ty.setText('50')
        ty.setFont(QFont("Arial",10))

        quiz_3_1_sub_gb_layout.setAlignment(Qt.AlignTop)
        quiz_3_1_sub_gb_layout.addRow("Angle (deg):", angle)
        quiz_3_1_sub_gb_layout.addRow("Scale :", scale)
        quiz_3_1_sub_gb_layout.addRow("Tx (px):", tx)
        quiz_3_1_sub_gb_layout.addRow("Ty (px):", ty)
        quiz_3_1_sub_gb.setLayout(quiz_3_1_sub_gb_layout)

        quiz_3_1_gb_layout.addWidget(quiz_3_1_sub_gb)

        self.quiz3 = Quiz3(angle, scale, tx, ty)

        quiz_3_1_btn = QPushButton('3.1 Rotation, Scale, Translate')
        quiz_3_1_btn.clicked.connect(self.quiz3.imageTransforms)
        quiz_3_1_gb_layout.addWidget(quiz_3_1_btn)

        quiz_3_1_gb.setLayout(quiz_3_1_gb_layout)

        quiz_3_2_gb = QGroupBox("3.2 Perspective Transformation")
        quiz_3_2_gb_layout = QVBoxLayout()

        quiz_3_2_btn = QPushButton("3.2 Perspective Transformation")
        quiz_3_2_btn.clicked.connect(self.quiz3.imagePerspective)
        quiz_3_2_gb_layout.addWidget(quiz_3_2_btn)

        quiz_3_2_gb.setLayout(quiz_3_2_gb_layout)

        quiz3_layout.addWidget(quiz_3_1_gb)
        quiz3_layout.addWidget(quiz_3_2_gb)

        self.quiz3groupbox.setLayout(quiz3_layout)

    def createFirst3Quiz(self):
        self.createQuiz1()
        self.createQuiz2()
        self.createQuiz3()

        quiz_12_vlayout = QVBoxLayout()
        quiz_12_vlayout.addWidget(self.quiz1groupbox)
        quiz_12_vlayout.addWidget(self.quiz2groupbox)

        quiz_12_3_hlayout = QHBoxLayout()
        quiz_12_3_hlayout.addLayout(quiz_12_vlayout)
        quiz_12_3_hlayout.addWidget(self.quiz3groupbox)

        self.horizontalGroupBoxes1 = quiz_12_3_hlayout
    
    def createQuiz4(self):
        self.quiz4 = Quiz4()

        self.quiz4groupbox = QGroupBox("4. Edge Detection")
        quiz4_layout = QVBoxLayout()
        quiz4_layout.setAlignment(Qt.AlignTop)

        gaussian_btn = QPushButton('4.1 Gaussian')
        gaussian_btn.clicked.connect(self.quiz4.gaussianSmooth)
        quiz4_layout.addWidget(gaussian_btn)

        sobel_x_btn = QPushButton('4.2 Sobel X')
        sobel_x_btn.clicked.connect(self.quiz4.sobel_x)
        quiz4_layout.addWidget(sobel_x_btn)

        sobel_y_btn =QPushButton('4.3 Sobel Y')
        sobel_y_btn.clicked.connect(self.quiz4.sobel_y)
        quiz4_layout.addWidget(sobel_y_btn)

        magnitude_btn = QPushButton('4.4 Magnitude')
        magnitude_btn.clicked.connect(self.quiz4.magnitude)
        quiz4_layout.addWidget(magnitude_btn)
        self.quiz4groupbox.setLayout(quiz4_layout)

    def createQuiz5(self):
        self.quiz5 = Quiz5()

        self.quiz5groupbox = QGroupBox("5. Training MNIST Classifier Using LeNet5")
        quiz5_layout = QVBoxLayout()
        quiz5_layout.setAlignment(Qt.AlignTop)

        quiz5_1_btn = QPushButton('5.1 Show Train Images')
        quiz5_1_btn.clicked.connect(self.quiz5.showTrainImage)
        quiz5_layout.addWidget(quiz5_1_btn)

        quiz5_2_btn = QPushButton('5.2 Show Hyperparameters')
        quiz5_2_btn.clicked.connect(self.quiz5.showHyperparemeters)
        quiz5_layout.addWidget(quiz5_2_btn)

        quiz5_3_btn = QPushButton('5.3 Train 1 Epoch')
        quiz5_3_btn.clicked.connect(self.quiz5.train_1_epoch)
        quiz5_layout.addWidget(quiz5_3_btn)
        
        quiz5_4_btn = QPushButton('5.4 Show Trianing Result')
        quiz5_4_btn.clicked.connect(self.quiz5.train_model)
        quiz5_layout.addWidget(quiz5_4_btn)
        
        
        form_layout = QFormLayout()
        index = QLineEdit()
        index.setText('1')
        index.setValidator(QIntValidator())
        index.setFont(QFont("Arial",10))
        form_layout.addRow("Test Image Index: ", index)
        self.quiz5.test_number = index

        quiz5_layout.addLayout(form_layout)

        quiz5_5_btn = QPushButton('5.5 Inference')
        quiz5_5_btn.clicked.connect(self.quiz5.predit_number)
        quiz5_layout.addWidget(quiz5_5_btn)
        self.quiz5groupbox.setLayout(quiz5_layout)

    def createLast2Quiz(self):
        self.createQuiz4()
        self.createQuiz5()

        self.horizontalGroupBoxes2 = QHBoxLayout()
        self.horizontalGroupBoxes2.addWidget(self.quiz4groupbox)
        self.horizontalGroupBoxes2.addWidget(self.quiz5groupbox)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
