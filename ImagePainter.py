import sys
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PIL import Image


def resize_image(input_image_path, output_image_path, size):
    original_image = Image.open(input_image_path)
    resized_image = original_image.resize(size)
    resized_image.save(output_image_path)


COLORS = ['#000000',
          '#141923',
          '#414168',
          '#3a7fa7',
          '#35e3e3',
          '#8fd970',
          '#5ebb49',
          '#458352',
          '#dcd37b',
          '#fffee5',
          '#ffd035',
          '#cc9245',
          '#a15c3e',
          '#a42f3b',
          '#f45b7a',
          '#c24998',
          '#81588d',
          '#bcb0c2',
          '#ffffff'
          ]


class PaletteButton(QtWidgets.QPushButton):
    def __init__(self, color):
        super().__init__()
        self.setFixedSize(QtCore.QSize(24, 24))
        self.color = color
        self.setStyleSheet("background-color: %s;" % color)


class Canvas(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()
        pixmap = QtGui.QPixmap(500, 500)
        pixmap.fill()
        self.setPixmap(pixmap)

        self.last_x, self.last_y = None, None
        self.pen_color = QtGui.QColor('#000000')

        self.spray = False
        self.width = 1

    def set_pen_color(self, c):
        self.pen_color = QtGui.QColor(c)

    def mouseMoveEvent(self, e):
        if self.last_x is None:
            self.last_x = e.x()
            self.last_y = e.y()
            return

        painter = QtGui.QPainter(self.pixmap())
        p = painter.pen()
        p.setWidth(self.width)
        p.setColor(self.pen_color)
        painter.setPen(p)
        painter.drawLine(self.last_x,
                         self.last_y,
                         e.x(),
                         e.y())
        painter.end()

        self.update()

        self.last_x = e.x()
        self.last_y = e.y()

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None

    def reImage(self):
        pixmap = QtGui.QPixmap(500, 500)
        pixmap.fill()
        self.setPixmap(pixmap)

    def saveImage(self, fname):
        i, okBtnPressed = QtWidgets.QInputDialog.getText(self, "Размер",
                                                         "Введите размеры картинки(через пробел)")
        if okBtnPressed:
            i = list(map(lambda x: int(x), i.split()))
            if len(i) == 2:
                pixmap = self.pixmap()
                pixmap.save(fname)
                resize_image(fname,
                             fname,
                             (i[0],
                              i[1]))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Paint")
        self.canvas = Canvas()

        w = QtWidgets.QWidget()
        o = QtWidgets.QVBoxLayout()
        w.setLayout(o)
        o.addWidget(self.canvas)

        palette = QtWidgets.QHBoxLayout()
        self.add_palette_buttons(palette)
        o.addLayout(palette)

        self.setCentralWidget(w)

        self.setRadioButtons()
        o.addWidget(self.groupBox)
        self.setFixedSize(500, 600)

    def add_palette_buttons(self, layout):
        for c in COLORS:
            b = PaletteButton(c)
            b.pressed.connect(lambda c=c: self.canvas.set_pen_color(c))
            layout.addWidget(b)

    def setRadioButtons(self):
        self.groupBox = QtWidgets.QGroupBox("Функции")

        hboxLayout = QtWidgets.QHBoxLayout()

        self.radiobtn6 = QtWidgets.QRadioButton("Ширина кисти")
        hboxLayout.addWidget(self.radiobtn6)
        self.radiobtn6.clicked.connect(self.setwidth)

        self.radiobtn15 = QtWidgets.QRadioButton("Очистить холст")
        hboxLayout.addWidget(self.radiobtn15)
        self.radiobtn15.clicked.connect(self.setparamsReImage)

        self.radiobtn16 = QtWidgets.QRadioButton("Сохранить")
        hboxLayout.addWidget(self.radiobtn16)
        self.radiobtn16.clicked.connect(self.setsaveImage)

        self.groupBox.setLayout(hboxLayout)

    def setwidth(self):
        i, okBtnPressed = QtWidgets.QInputDialog.getText(self, "Ширина кисти",
                                                         "Введите ширину кисти")
        i = int(i)

        if i > 0:
            self.canvas.width = i

    def setparamsReImage(self):
        self.canvas.reImage()

    def setsaveImage(self):
        fname = QtWidgets.QFileDialog.getSaveFileName(self,
                                                      "Сохранить",
                                                      '',
                                                      "*.png")
        self.canvas.saveImage(fname[0])


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()
