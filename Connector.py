from PyQt5.QtWidgets import QMainWindow
from MyApplication import Ui_MainWindow
from Functions import MyFunctions
from Project_1 import Project_1
from Project_2 import Project_2


class MyConnector(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self,self)
        self.setWindowTitle("My Application")
        self.icon_name_widget.setHidden(True)
        self.functions = MyFunctions()

        self.functions.set_stackedWidget(self.stackedWidget)
        self.project_1 = Project_1()
        self.project_1.set_LineEdit(self.lineEdit_5)
        self.project_1.set_label53(self.label_53)
        self.project_1.set_label55(self.label_55)
        self.project_1.set_comboBox(self.comboBox_8)

        self.project_2 = Project_2()

        self.project_2.set_label58(self.label_58)
        self.project_2.set_label60(self.label_60)
        self.project_2.set_comboBox9(self.comboBox_9)
        self.project_2.set_comboBox10(self.comboBox_10)

        self.dashboard_1.clicked.connect(self.functions.switch_to_dashboardPage)
        self.dashboard_2.clicked.connect(self.functions.switch_to_dashboardPage)

        self.profile_1.clicked.connect(self.functions.switch_to_profilePage)
        self.profile_2.clicked.connect(self.functions.switch_to_profilePage)

        self.project_img_1.clicked.connect(self.functions.switch_to_project_1)
        self.project_img_name_1.clicked.connect(self.functions.switch_to_project_1)

        self.project_img_2.clicked.connect(self.functions.switch_to_project_2)
        self.project_img_name_2.clicked.connect(self.functions.switch_to_project_2)

        self.project_img_3.clicked.connect(self.functions.switch_to_project_3)
        self.project_img_name_3.clicked.connect(self.functions.switch_to_project_3)

        self.pushButton.clicked.connect(self.project_2.hough_functions)
        self.pushButton_2.clicked.connect(self.project_2.calculate_dark_green_properties)
        self.pushButton_5.clicked.connect(self.project_2.deblur_image)
        self.pushButton_6.clicked.connect(self.project_2.sigmoid_functions)
        self.pushButton_4.clicked.connect(self.project_2.load_image)


        self.pushButton_3.clicked.connect(self.project_1.load_image)
        self.pushButton_15.clicked.connect(lambda: self.project_1.rotate())
        self.pushButton_14.clicked.connect(lambda: self.project_1.enlargeImage())
        self.pushButton_13.clicked.connect(lambda: self.project_1.reduceImage())
        self.label_55.wheelEvent = lambda event: self.project_1.zoom(event)
