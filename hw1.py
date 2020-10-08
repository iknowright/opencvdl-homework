import sys

# GUI use case
from PyQt5.QtWidgets import QDesktopWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QDialog
from PyQt5.QtWidgets import QPushButton, QGroupBox, QLineEdit, QApplication, QMessageBox
from PyQt5.QtGui import QDoubleValidator, QIntValidator, QFont
from PyQt5.QtCore import Qt, pyqtSlot

from quiz import *

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
        self.createFirst2Quiz()
        self.createLast2Quiz()

        windowLayout = QHBoxLayout()
        windowLayout.addLayout(self.verticalGroupBoxes1)
        windowLayout.addLayout(self.verticalGroupBoxes2)
        self.setLayout(windowLayout)

        self.center()
        self.show()


    def createQuiz1(self):
        self.quiz1 = Quiz1()

        self.quiz1groupbox = QGroupBox("1. Stereo")
        quiz1_layout = QVBoxLayout()
        quiz1_layout.setAlignment(Qt.AlignTop)

        show_img_btn = QPushButton('1.1 Disparity')
        show_img_btn.clicked.connect(self.quiz1.depth_map)
        quiz1_layout.addWidget(show_img_btn)

        self.quiz1groupbox.setLayout(quiz1_layout)


    def createQuiz2(self):
        self.quiz2 = Quiz2()

        self.quiz2groupbox = QGroupBox("2. Background subtraction")
        quiz2_layout = QVBoxLayout()
        quiz2_layout.setAlignment(Qt.AlignTop)

        bg_sub_btn = QPushButton('2.1 Background subtraction')
        bg_sub_btn.clicked.connect(self.quiz2.backgroundSubtraction)
        quiz2_layout.addWidget(bg_sub_btn)

        self.quiz2groupbox.setLayout(quiz2_layout)


    def createFirst2Quiz(self):
        self.createQuiz1()
        self.createQuiz2()

        quiz_12_vlayout = QVBoxLayout()
        quiz_12_vlayout.addWidget(self.quiz1groupbox)
        quiz_12_vlayout.addWidget(self.quiz2groupbox)

        self.verticalGroupBoxes1 = quiz_12_vlayout


    def createQuiz3(self):
        self.quiz3 = Quiz3()

        self.quiz3groupbox = QGroupBox("3. Feature Tracking")
        quiz3_layout = QVBoxLayout()

        preprocess_btn = QPushButton('3.1 Preprocessing')
        preprocess_btn.clicked.connect(self.quiz3.preprocessing)
        quiz3_layout.addWidget(preprocess_btn)

        video_tracking_btn = QPushButton('3.2 Video tracking')
        video_tracking_btn.clicked.connect(self.quiz3.opticalFlow)
        quiz3_layout.addWidget(video_tracking_btn)

        self.quiz3groupbox.setLayout(quiz3_layout)

    def createQuiz4(self):
        self.quiz4 = Quiz4()

        self.quiz4groupbox = QGroupBox("4. Augmented reality")
        quiz4_layout = QVBoxLayout()
        quiz4_layout.setAlignment(Qt.AlignTop)

        ar_btn = QPushButton('4.1 Augmented reality')
        ar_btn.clicked.connect(self.quiz4.projection)
        quiz4_layout.addWidget(ar_btn)

        self.quiz4groupbox.setLayout(quiz4_layout)

    def createLast2Quiz(self):
        self.createQuiz3()
        self.createQuiz4()

        self.verticalGroupBoxes2 = QVBoxLayout()
        self.verticalGroupBoxes2.addWidget(self.quiz3groupbox)
        self.verticalGroupBoxes2.addWidget(self.quiz4groupbox)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
