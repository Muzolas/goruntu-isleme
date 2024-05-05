# Main.py

import sys
from PyQt5.QtWidgets import QApplication
from Connector import MyConnector


def main():
    app = QApplication(sys.argv)
    window = MyConnector()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
