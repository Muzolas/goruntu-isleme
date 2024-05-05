import math

import numpy as np
from PIL import Image

from PIL.ImageQt import ImageQt
from PyQt5.QtCore import Qt, QPoint, QRectF
from PyQt5.QtGui import QPixmap, QPainter, QTransform, QImage, QColor, qRgb
from PyQt5.QtWidgets import QMainWindow, QFileDialog


class Project_1(QMainWindow):
    def __init__(self):
        super().__init__()
        self.lineEdit = None
        self.label_55 = None
        self.label_53 = None
        self.comboBox = 0

    def set_LineEdit(self, lineEdit):
        self.lineEdit = lineEdit


    def set_label53(self, label53):
        self.label_53 = label53

    def set_label55(self, label55):
        self.label_55 = label55

    def set_comboBox(self, combobox):
        self.comboBox = combobox

    # Resim Yükleme
    def load_image(self):
        # Kullanıcıya dosya seçme iletişim kutusunu açar ve seçilen dosya yolu alınır
        img_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")

        # Eğer bir dosya seçildiyse devam edilir
        if img_path:
            # Seçilen görüntü dosyası bir NumPy dizisine dönüştürülür ve 'img' özelliğine atanır
            self.img = np.array(Image.open(img_path))

            # Görüntü dosyası bir QPixmap nesnesine dönüştürülür
            pixmap = QPixmap(img_path)

            # Görüntüyü göstermek için etiketlere atanır
            self.label_55.setPixmap(pixmap)  # Büyütme etiketi
            self.label_53.setPixmap(pixmap)  # Küçültme etiketi

    # RESMİ DÖNDERME
    def rotate(self):
        # Kullanıcının girdiği dönme açısını alır
        degree_text = self.lineEdit.text().strip()
        if degree_text == "":
            degree = 0
        else:
            degree = int(degree_text)
        # Dönme fonksiyonuna gerekli açıyı ileterek çağrılır
        self.rotate_image(degree)

    def rotate_image(self, degree):
        # Eğer bir görüntü mevcutsa işleme devam edilir
        if hasattr(self, 'img'):
            # Dereceyi radyan cinsine dönüştürür
            theta_rad = math.radians(degree)

            height, width, _ = self.img.shape

            # Görüntünün orta noktasını bulur
            center_x = width / 2
            center_y = height / 2

            # Döndürülmüş köşe noktalarını hesaplar
            corners = [
                (0, 0),
                (width, 0),
                (0, height),
                (width, height)
            ]

            rotated_corners = []
            for corner_x, corner_y in corners:
                # Döndürülmüş köşe noktalarını hesaplar
                new_x = int((corner_x - center_x) * math.cos(theta_rad) - (corner_y - center_y) * math.sin(
                    theta_rad) + center_x)
                new_y = int((corner_x - center_x) * math.sin(theta_rad) + (corner_y - center_y) * math.cos(
                    theta_rad) + center_y)
                rotated_corners.append((new_x, new_y))

            # Döndürülen görüntünün yeni boyutlarını hesaplar
            min_x = min(rotated_corners, key=lambda x: x[0])[0]
            max_x = max(rotated_corners, key=lambda x: x[0])[0]
            min_y = min(rotated_corners, key=lambda x: x[1])[1]
            max_y = max(rotated_corners, key=lambda x: x[1])[1]

            new_width = max_x - min_x
            new_height = max_y - min_y

            # Döndürülen görüntü için boş bir matris oluşturur
            transformed_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)

            for y in range(new_height):
                for x in range(new_width):
                    # Döndürülen pikselin orijinal koordinatlarını hesaplar
                    src_x = int((x + min_x - center_x) * math.cos(-theta_rad) - (y + min_y - center_y) * math.sin(
                        -theta_rad) + center_x)
                    src_y = int((x + min_x - center_x) * math.sin(-theta_rad) + (y + min_y - center_y) * math.cos(
                        -theta_rad) + center_y)

                    # Orijinal görüntü sınırları içindeyse, dönüştürülen görüntüye atar
                    if 0 <= src_x < width and 0 <= src_y < height:
                        transformed_img[y, x] = self.img[src_y, src_x, :3]

            # Dönüştürülen görüntüyü QImage formatına dönüştürür
            image = Image.fromarray(transformed_img)
            qimage = QImage(image.tobytes(), image.width, image.height, QImage.Format_RGB888)

            # QImage'i QPixmap'e dönüştürür ve etikete atar
            rotated_pixmap = QPixmap.fromImage(qimage)
            self.label_55.setPixmap(rotated_pixmap)

    # RESMİ BÜYÜLTME
    def enlargeImage(self):
        # Eğer 'img' adında bir özellik varsa işlem yapılır
        if hasattr(self, 'img'):
            # Ölçek faktörünü belirlemek için comboBox'tan bir indis alınır
            scale_factor = float(self.comboBox.currentIndex() + 1)
            height, width, channels = self.img.shape

            # Yeni boyutlar hesaplanır
            new_height = int(height * scale_factor)
            new_width = int(width * scale_factor)

            # Yeniden boyutlandırılmış boş bir görüntü oluşturulur
            enlarged_img = np.zeros((new_height, new_width, channels), dtype=np.uint8)

            # Yeni boyutlar için orijinal görüntüyü kullanarak interpolasyon yapılır
            for i in range(new_height):
                for j in range(new_width):
                    x = int(i / scale_factor)
                    y = int(j / scale_factor)
                    enlarged_img[i, j] = self.img[x, y]

            # QImage'e dönüştürülür
            image = Image.fromarray(enlarged_img)
            qimage = QImage(image.tobytes(), image.width, image.height, QImage.Format_RGB888)

            # QPixmap'e dönüştürülür
            enlarged_pixmap = QPixmap.fromImage(qimage)

            # Büyütülmüş görüntü etikete atanır
            self.label_55.setPixmap(enlarged_pixmap)

    # RESMİ KÜÇÜLTME
    def reduceImage(self):
        # Eğer 'img' adında bir özellik varsa işlem yapılır
        if hasattr(self, 'img'):
            # Ölçek faktörünü belirlemek için comboBox'tan bir indis alınır
            scale_factor = float(self.comboBox.currentIndex() + 1)
            height, width, channels = self.img.shape

            # Yeni boyutlar hesaplanır
            new_height = int(height / scale_factor)
            new_width = int(width / scale_factor)

            # Küçültülmüş boş bir görüntü oluşturulur
            reduced_img = np.zeros((new_height, new_width, channels), dtype=np.uint8)

            # Yeni boyutlar için orijinal görüntüdeki piksellerin ortalaması alınarak interpolasyon yapılır
            for i in range(new_height):
                for j in range(new_width):
                    # İlgili piksel aralığının koordinatları belirlenir
                    y_start = int(i * scale_factor)
                    y_end = min(int((i + 1) * scale_factor), height)
                    x_start = int(j * scale_factor)
                    x_end = min(int((j + 1) * scale_factor), width)

                    # Piksel aralığının ortalaması alınır ve yeni görüntüye atanır
                    reduced_img[i, j] = np.mean(self.img[y_start:y_end, x_start:x_end], axis=(0, 1))

            # Küçültülmüş görüntü bir QImage'e dönüştürülür
            image = Image.fromarray(reduced_img)
            qimage = QImage(image.tobytes(), image.width, image.height, QImage.Format_RGB888)

            # QImage, QPixmap'e dönüştürülerek etikete atanır
            reduced_pixmap = QPixmap.fromImage(qimage)
            self.label_55.setPixmap(reduced_pixmap)

    zoom_factor = 1.1  # Yakınlaştırma faktörü, varsayılan olarak 1.1 olarak tanımlanır

    def zoom(self, event):
        if hasattr(self, 'img'):  # 'img' özelliğinin mevcut olup olmadığını kontrol eder
            factor = 1.2  # Yakınlaştırma faktörü, varsayılan olarak 1.2 olarak tanımlanır
            if event.angleDelta().y() < 0:  # Fare tekerleği aşağı yönlü döndürüldüğünde
                factor = 1.0 / factor  # Yakınlaştırma faktörünü tersine çevirir

            self.zoom_factor *= factor  # Yakınlaştırma faktörünü günceller

            current_pixmap = self.label_55.pixmap()  # Mevcut pixmap'i alır

            if current_pixmap is not None:
                q_image = current_pixmap.toImage()  # Pixmap'i QImage'e dönüştürür

                new_width = round(current_pixmap.width() * self.zoom_factor)  # Yeni genişliği hesaplar
                new_height = round(current_pixmap.height() * self.zoom_factor)  # Yeni yüksekliği hesaplar

                zoomed_image = QImage(new_width, new_height, QImage.Format_RGB32)  # Yeni boyutta bir QImage oluşturur

                for y in range(new_height):
                    for x in range(new_width):
                        original_x = x / self.zoom_factor  # Orijinal x koordinatını hesaplar
                        original_y = y / self.zoom_factor  # Orijinal y koordinatını hesaplar

                        color = self.bilinear_interpolation(q_image, original_x,
                                                            original_y)  # İkili interpolasyon uygular
                        zoomed_image.setPixel(x, y, color.rgb())  # Pikselin rengini ayarlar

                pixmap = QPixmap.fromImage(zoomed_image)  # QImage'i QPixmap'e dönüştürür
                self.label_55.setPixmap(pixmap)  # Etiket üzerine QPixmap'i yerleştirir

    def bilinear_interpolation(self, image, x, y):
        x_floor, y_floor = int(x), int(y)  # x ve y'nin tam kısmını alır
        x_ceil, y_ceil = x_floor + 1, y_floor + 1  # Üst tam kısmını alır

        q11 = QColor(image.pixel(x_floor, y_floor))  # Köşe piksellerin renklerini alır
        q12 = QColor(image.pixel(x_ceil, y_floor))
        q21 = QColor(image.pixel(x_floor, y_ceil))
        q22 = QColor(image.pixel(x_ceil, y_ceil))

        dx, dy = x - x_floor, y - y_floor  # Aradaki farkı hesaplar

        r1 = self.interpolate(q11.red(), q12.red(), dx)  # Kırmızı renk bileşenleri için interpolasyon yapar
        r2 = self.interpolate(q21.red(), q22.red(), dx)
        red = int(self.interpolate(r1, r2, dy))

        g1 = self.interpolate(q11.green(), q12.green(), dx)  # Yeşil renk bileşenleri için interpolasyon yapar
        g2 = self.interpolate(q21.green(), q22.green(), dx)
        green = int(self.interpolate(g1, g2, dy))

        b1 = self.interpolate(q11.blue(), q12.blue(), dx)  # Mavi renk bileşenleri için interpolasyon yapar
        b2 = self.interpolate(q21.blue(), q22.blue(), dx)
        blue = int(self.interpolate(b1, b2, dy))

        return QColor(red, green, blue)  # İnterpolasyon sonucunda elde edilen rengi döndürür

    def interpolate(self, p1, p2, ratio):
        return p1 + ratio * (p2 - p1)  # İki değer arasındaki oranı kullanarak interpolasyon yapar




