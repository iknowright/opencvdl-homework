import sys

# GUI use case
from PyQt5.QtWidgets import QDesktopWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QDialog
from PyQt5.QtWidgets import QPushButton, QGroupBox, QLineEdit, QApplication, QMessageBox, QLabel, QComboBox
from PyQt5.QtGui import QDoubleValidator, QIntValidator, QFont
from PyQt5.QtCore import Qt, pyqtSlot

from quiz import Quiz1, Quiz3

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
        # self.createLast2Quiz()

        windowLayout = QVBoxLayout()
        windowLayout.addWidget(self.createQuiz1())
        windowLayout.addWidget(self.createQuiz2())
        windowLayout.addWidget(self.createQuiz3())
        self.setLayout(windowLayout)

        self.center()
        self.show()


    def createQuiz1(self):
        self.quiz1 = Quiz1()

        quiz1groupbox = QGroupBox("1. Calibration")
        quiz1_layout = QHBoxLayout()
        quiz1_layout.setAlignment(Qt.AlignTop)

        quiz_11_12_14_layout = QVBoxLayout()
        quiz_11_12_14_layout.setAlignment(Qt.AlignTop)

        quiz_13_groupbox = QGroupBox("1.3 Find Extrinsic")

        btn_11 = QPushButton('1.1 Find Corners')
        btn_11.clicked.connect(self.quiz1.chessboard_corners)
        quiz_11_12_14_layout.addWidget(btn_11)

        btn_12 = QPushButton('1.2 Find Intrinsic')
        btn_12.clicked.connect(self.quiz1.get_intrinsic_matrix)
        quiz_11_12_14_layout.addWidget(btn_12)

        btn_14 = QPushButton('1.4 Find Distortion')
        btn_14.clicked.connect(self.quiz1.get_distortion_matrix)
        quiz_11_12_14_layout.addWidget(btn_14)

        quiz_13_layout = QVBoxLayout()
        quiz_13_layout.addWidget(QLabel("Select image"))

        image_combo_box = QComboBox()
        image_combo_box.addItem('')
        image_combo_box.addItems(self.quiz1.filenames)
        image_combo_box.activated[str].connect(self.quiz1.set_current_image)

        quiz_13_layout.addWidget(image_combo_box)

        btn_13 = QPushButton('1.3 Find Extrinsic')
        btn_13.clicked.connect(self.quiz1.get_extrinsic_matrix)
        quiz_13_layout.addWidget(btn_13)

        quiz1_layout.addLayout(quiz_11_12_14_layout)
        quiz1_layout.addLayout(quiz_13_layout)
        quiz1groupbox.setLayout(quiz1_layout)

        return quiz1groupbox


    def createQuiz2(self):
        # self.quiz2 = Quiz2()

        quiz2groupbox = QGroupBox("2. Augmented Reality")
        quiz2_layout = QVBoxLayout()
        quiz2_layout.setAlignment(Qt.AlignTop)

        btn_21 = QPushButton('2.1 Show tetrahedron')
        # btn_21.clicked.connect(self.quiz2.backgroundSubtraction)
        quiz2_layout.addWidget(btn_21)

        quiz2groupbox.setLayout(quiz2_layout)

        return quiz2groupbox


    def createQuiz3(self):
        self.quiz3 = Quiz3()

        quiz3groupbox = QGroupBox("3. Stereo Disparity Map")
        quiz3_layout = QVBoxLayout()
        quiz3_layout.setAlignment(Qt.AlignTop)

        btn_31 = QPushButton('3.1 Show disparity map')
        btn_31.clicked.connect(self.quiz3.depth_map)
        quiz3_layout.addWidget(btn_31)

        quiz3groupbox.setLayout(quiz3_layout)

        return quiz3groupbox


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
