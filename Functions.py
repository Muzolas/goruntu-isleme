from PyQt5.QtWidgets import QMainWindow


class MyFunctions(QMainWindow):
    def __init__(self):
        super().__init__()
        self.stackedWidget = None


    def set_stackedWidget(self, stackedWidget):
        self.stackedWidget = stackedWidget

    def switch_to_dashboardPage(self):
        self.stackedWidget.setCurrentIndex(0)

    def switch_to_profilePage(self):
        self.stackedWidget.setCurrentIndex(4)

    def switch_to_project_1(self):
        self.stackedWidget.setCurrentIndex(2)

    def switch_to_project_2(self):
        self.stackedWidget.setCurrentIndex(1)

    def switch_to_project_3(self):
        self.stackedWidget.setCurrentIndex(3)

