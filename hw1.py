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

        self.setWindowTitle("Homework 1")
        # self.createLast2Quiz()

        windowLayout = QVBoxLayout()
        windowLayout.addWidget(self.createQuiz1())
        windowLayout.addWidget(self.createQuiz2())
        windowLayout.addWidget(self.createQuiz3())
        windowLayout.addWidget(self.createQuiz4())
        windowLayout.addWidget(self.createQuiz5())
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


        btn_11 = QPushButton('1.1 Find Corners')
        btn_11.clicked.connect(self.quiz1.chessboard_corners)
        quiz_11_12_14_layout.addWidget(btn_11)

        btn_12 = QPushButton('1.2 Find Intrinsic')
        btn_12.clicked.connect(self.quiz1.get_intrinsic_and_extrinsic_matrix)
        quiz_11_12_14_layout.addWidget(btn_12)

        btn_14 = QPushButton('1.4 Find Distortion')
        btn_14.clicked.connect(self.quiz1.get_distortion_matrix)
        quiz_11_12_14_layout.addWidget(btn_14)

        quiz_13_groupbox = QGroupBox("1.3 Find Extrinsic")
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
        quiz_13_groupbox.setLayout(quiz_13_layout)
        quiz1_layout.addWidget(quiz_13_groupbox)
        quiz1groupbox.setLayout(quiz1_layout)

        return quiz1groupbox


    def createQuiz2(self):
        self.quiz2 = Quiz2()

        quiz2groupbox = QGroupBox("2. Augmented Reality")
        quiz2_layout = QVBoxLayout()
        quiz2_layout.setAlignment(Qt.AlignTop)

        btn_21 = QPushButton('2.1 Show tetrahedron')
        btn_21.clicked.connect(self.quiz2.projection)
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

    def createQuiz4(self):
        self.quiz4 = Quiz4()

        quiz4groupbox = QGroupBox("4. SIFT")
        quiz4_layout = QVBoxLayout()
        quiz4_layout.setAlignment(Qt.AlignTop)

        btn_41 = QPushButton('4.1 KeyPoints')
        btn_41.clicked.connect(self.quiz4.sift)
        quiz4_layout.addWidget(btn_41)

        btn_42 = QPushButton('4.2 Matched Points')
        btn_42.clicked.connect(self.quiz4.matcher)
        quiz4_layout.addWidget(btn_42)

        quiz4groupbox.setLayout(quiz4_layout)

        return quiz4groupbox

    def createQuiz5(self):
        # self.quiz4 = Quiz4()

        quiz5groupbox = QGroupBox("5. Cifar10 - VGG16")
        quiz5_layout = QVBoxLayout()
        quiz5_layout.setAlignment(Qt.AlignTop)

        btn_51 = QPushButton('5.1 Show Train Images')
        # btn_51.clicked.connect()
        quiz5_layout.addWidget(btn_51)

        btn_52 = QPushButton('5.2 Show Hyperparameters')
        # btn_52.clicked.connect()
        quiz5_layout.addWidget(btn_52)

        btn_53 = QPushButton('5.3 Show Model Structure')
        # btn_53.clicked.connect()
        quiz5_layout.addWidget(btn_53)

        btn_54 = QPushButton('5.4 Show Accuracy')
        # btn_54.clicked.connect()
        quiz5_layout.addWidget(btn_54)

        self.textbox = QLineEdit(self)
        quiz5_layout.addWidget(self.textbox)

        btn_55 = QPushButton('5.5 Test')
        btn_55.clicked.connect(self.on_click)
        quiz5_layout.addWidget(btn_55)

        quiz5groupbox.setLayout(quiz5_layout)

        return quiz5groupbox

    @pyqtSlot()
    def on_click(self):
        textboxValue = self.textbox.text()
        print(textboxValue)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
