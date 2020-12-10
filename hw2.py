import sys
from functools import partial

# GUI use case
from PyQt5.QtWidgets import QDesktopWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QDialog
from PyQt5.QtWidgets import QPushButton, QGroupBox, QLineEdit, QApplication, QMessageBox, QLabel, QComboBox
from PyQt5.QtGui import QDoubleValidator, QIntValidator, QFont
from PyQt5.QtCore import Qt, pyqtSlot

from quiz import Quiz1, Quiz2, Quiz3, Quiz4

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

        self.setWindowTitle("Homework 2")
        # self.createLast2Quiz()

        windowLayout = QVBoxLayout()
        windowLayout.addWidget(self.createQuiz1())
        windowLayout.addWidget(self.createQuiz2())
        windowLayout.addWidget(self.createQuiz3())
        windowLayout.addWidget(self.createQuiz4())
        self.setLayout(windowLayout)

        self.center()
        self.show()


    def createQuiz1(self):
        self.quiz1 = Quiz1()

        quiz1groupbox = QGroupBox("1. Background Subtraction")
        quiz1_layout = QVBoxLayout()
        quiz1_layout.setAlignment(Qt.AlignTop)

        btn_11 = QPushButton('1.1 Background Subtraction')
        btn_11.clicked.connect(self.quiz1.background_subtraction)
        quiz1_layout.addWidget(btn_11)

        quiz1groupbox.setLayout(quiz1_layout)

        return quiz1groupbox


    def createQuiz2(self):
        self.quiz2 = Quiz2()

        quiz2groupbox = QGroupBox("2. Optical Flow")
        quiz2_layout = QVBoxLayout()
        quiz2_layout.setAlignment(Qt.AlignTop)

        btn_21 = QPushButton('2.1 Preprocessing')
        btn_21.clicked.connect(self.quiz2.preprocessing)
        quiz2_layout.addWidget(btn_21)

        btn_21 = QPushButton('2.2 Video Tracking')
        btn_21.clicked.connect(self.quiz2.optical_flow)
        quiz2_layout.addWidget(btn_21)

        quiz2groupbox.setLayout(quiz2_layout)

        return quiz2groupbox


    def createQuiz3(self):
        self.quiz3 = Quiz3()

        quiz3groupbox = QGroupBox("3. Perspective Transform")
        quiz3_layout = QVBoxLayout()
        quiz3_layout.setAlignment(Qt.AlignTop)

        btn_31 = QPushButton('3.1 Perspective Transform')
        # btn_31.clicked.connect()
        quiz3_layout.addWidget(btn_31)

        quiz3groupbox.setLayout(quiz3_layout)

        return quiz3groupbox

    def createQuiz4(self):
        self.quiz4 = Quiz4()

        quiz4groupbox = QGroupBox("4. PCA")
        quiz4_layout = QVBoxLayout()
        quiz4_layout.setAlignment(Qt.AlignTop)

        btn_41 = QPushButton('4.1 Image Reconstruction')
        # btn_41.clicked.connect()
        quiz4_layout.addWidget(btn_41)

        btn_42 = QPushButton('4.2 Compute Reconstruction Error')
        # btn_42.clicked.connect()
        quiz4_layout.addWidget(btn_42)

        quiz4groupbox.setLayout(quiz4_layout)

        return quiz4groupbox


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
