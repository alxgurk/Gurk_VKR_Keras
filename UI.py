import sys
import keras_cv
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from googletrans import Translator
from tensorflow import keras
import tensorflow as tf
import math
from PIL import Image
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap
from Designs import design
from Designs import de
class MainApp(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        # Это здесь нужно для доступа к переменным, методам
        # и т.д. в файле design.py
        super().__init__()
        self.setupUi(self)  # Это нужно для инициализации нашего дизайна


class Result_Win(QtWidgets.QMainWindow, design2.Ui_ReaultW):
    def __init__(self):
        super().__init__()
        self.setupUi(self)


images = []


def main():
    result_window = QtWidgets.QApplication(sys.argv)
    window2 = Result_Win()

    def on_click():
        print(window.TypeGeneration.currentText())
        global images

        print("1")
        model = keras_cv.models.StableDiffusion()
        if len(window.Line.toPlainText()) == 0:
            error_message("Текстовая подсказка пуста!")
            return

        prompt = translate_prompt(window.Line.toPlainText())

        if len(prompt) > 500:
            error_message("Слишком большая текстовая подсказка!")
            return

        print("3")
        encoding = tf.squeeze(model.encode_text(prompt))
        print("4")

        if window.TypeGeneration.currentText() == "Анимация":
            walk_steps = 6
            batch_size = 2
        else:
            walk_steps = 1
            batch_size = 1

        batches = walk_steps // batch_size


        try:
            seed = int(window.LineSeed.text())
        except:
            error_message("Сид не указан")
            return

        if len(str(seed)) != 5:
            error_message("Сид должен быть пятизначным числом")
            return

        window2.show()

        noise = tf.random.normal((512 // 8, 512 // 8, 4), seed=seed)

        walk_noise_x = tf.random.normal(noise.shape, dtype=tf.float64)
        walk_noise_y = tf.random.normal(noise.shape, dtype=tf.float64)

        walk_scale_x = tf.cos(tf.linspace(0, 2, walk_steps) * math.pi)
        walk_scale_y = tf.sin(tf.linspace(0, 2, walk_steps) * math.pi)
        noise_x = tf.tensordot(walk_scale_x, walk_noise_x, axes=0)
        noise_y = tf.tensordot(walk_scale_y, walk_noise_y, axes=0)
        noise = tf.add(noise_x, noise_y)
        batched_noise = tf.split(noise, batches)
        result_window.exec_()
        print("5")
        # Тут читаем галку на промежуточный результат
        pr = bool(window.checkBox.clicked)
        for batch in range(batches):
            images += [
                Image.fromarray(img)
                for img in model.generate_image(
                    encoding,
                    batch_size=batch_size,
                    num_steps=8,
                    diffusion_noise=batched_noise[batch],
                    pre=pr,
                )
            ]
        print("6")
        if window.TypeGeneration.currentText() == "Анимация":
            export_as_gif(fr"C:\Users\hp\PycharmProjects\qt\results\result.gif", images)
        else:
            images[0].save(
                fr"C:\Users\hp\PycharmProjects\qt\results\result.gif",
                save_all=True,
                duration=10,
                loop=0,
            )
        print("7")
        qPixmap = QPixmap(fr"C:\Users\hp\PycharmProjects\qt\results\result.gif")
        print("8")
        window2.result.setPixmap(QPixmap(qPixmap))

    def on_click_save():
        global images
        print("9")
        filePath = QFileDialog.getSaveFileName()
        print("10")
        if filePath == "":
            return
        print(filePath)

        if window.TypeGeneration.currentText() == "Анимация":
            print("11")
            export_as_gif(filePath[0], images)
            print("12")
        else:
            print("13")
            images[0].save(
                filePath[0],
                save_all=True,
                duration=10,
                loop=0,
            )
            print("14")

    app = QtWidgets.QApplication(sys.argv)  # Новый экземпляр QApplication
    window = MainApp()  # Создаём объект класса ExampleApp
    window.show()  # Показываем окно

    print(window.StartButton.clicked)
    print(window.StartButton.clicked.connect(on_click))
    window2.SaveB.clicked.connect(on_click_save)
    app.exec_()  # и запускаем приложение


def translate_prompt(text):
    translator = Translator()

    translation = translator.translate(text=text, dest="en")
    return translation.text


def export_as_gif(filename, _images, frames_per_second=10, rubber_band=False):
    if rubber_band:
        _images += _images[2:-1][::-1]
    _images[0].save(
        filename,
        save_all=True,
        append_images=_images[1:],
        duration=1000 // frames_per_second,
        loop=0,
    )


def error_message(mes: str):
    msg = QMessageBox()

    msg.setIcon(QMessageBox.Critical)
    msg.setWindowTitle("Ошибка")

    msg.setText(mes)
    msg.setStandardButtons(QMessageBox.Ok)

    msg.exec_()


if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()
