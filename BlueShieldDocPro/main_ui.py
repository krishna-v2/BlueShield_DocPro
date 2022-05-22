import sys
from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QPixmap
from PySide2.QtWidgets import QFileDialog, QMessageBox
from os.path import dirname, join, isfile
import cv2
from BS_MainFile import BlueShield


class BlueShieldDocProcUI(QtWidgets.QMainWindow):
    def __init__(self, ui_file='main.ui', parent=None):
        super(BlueShieldDocProcUI, self).__init__(parent)
        self.ui = QUiLoader().load(QtCore.QFile(ui_file))

        self.show_img('ui/logo.jpg')

        self.ui.browse.clicked.connect(self.load_file)
        self.ui.extract.clicked.connect(self.extract_data)

        self.docpro = BlueShield('brc_template1')

    def load_file(self):
        fpath, _ = QFileDialog.getOpenFileName(self, "Load Image",
                                               join(dirname(dirname(__file__)), "filled_forms"),
                                               "Images (*.png)")
        if not isfile(fpath):
            return
        self.ui.fpath.setText(fpath)
        self.show_img(fpath)

    def show_img(self, fpath):
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        pixmap = QPixmap(fpath)
        if h > w:
            pixmap = pixmap.scaled(2000, 1000, QtCore.Qt.KeepAspectRatio)
        else:
            pixmap = pixmap.scaled(800, 400, QtCore.Qt.KeepAspectRatio)
        self.ui.img.setPixmap(pixmap)

    @staticmethod
    def label_to_ans(label):
        if label == 1:
            return 'Yes'
        return 'No'

    def extract_data(self):
        file = self.ui.fpath.text()
        if not isfile(file):
            QMessageBox.information(self, "Info", "File does not exist.")
            return
        r = self.docpro.extract_data(file)

        res = '\n'.join([f'ID: {r["ID"]}',
                         f'Name: {r["name"]}',
                         f'Address: {r["address"]}',
                         f'Phone: {r["phone"]}',
                         f'Email: {r["email_canvas"]["email"]}',
                         f'Email Score: {r["email_canvas"]["count_score"]:.2f}',
                         f'Mail Checkbox: {self.label_to_ans(r["mail_checkbox"])}',
                         f'Phone Checkbox: {self.label_to_ans(r["phone_checkbox"])}',
                         f'Email Checkbox: {self.label_to_ans(r["email_checkbox"])}'])
        self.ui.result.setPlainText(res)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = BlueShieldDocProcUI()
    window.ui.show()
    sys.exit(app.exec_())
