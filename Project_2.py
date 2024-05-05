import math


import cv2
import numpy as np
from PIL import Image

from PIL.ImageQt import ImageQt
from PyQt5.QtCore import Qt, QPoint, QRectF
from PyQt5.QtGui import QPixmap, QPainter, QTransform, QImage, QColor, qRgb
from PyQt5.QtWidgets import QMainWindow, QFileDialog
import pandas as pd

class Project_2(QMainWindow):
    def __init__(self):
        super().__init__()

        self.label_60 = None
        self.label_58 = None
        self.comboBox9 = 0
        self.comboBox10 = 0

    def set_LineEdit(self, lineEdit):
        self.lineEdit = lineEdit


    def set_label58(self, label58):
        self.label_58 = label58

    def set_label60(self, label60):
        self.label_60 = label60

    def set_comboBox9(self, combobox9):
        self.comboBox9 = combobox9

    def set_comboBox10(self, combobox10):
            self.comboBox10 = combobox10

    # Global değişken
    global_img_path = ""

    # Resim Yükleme
    def load_image(self):
        global global_img_path
        # Kullanıcıya dosya seçme iletişim kutusunu açar ve seçilen dosya yolu alınır
        img_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        global_img_path = img_path
        # Eğer bir dosya seçildiyse devam edilir
        if img_path:
            # Seçilen görüntü dosyası bir NumPy dizisine dönüştürülür ve 'img' özelliğine atanır
            self.img = np.array(Image.open(img_path))

            # Görüntü dosyası bir QPixmap nesnesine dönüştürülür
            pixmap = QPixmap(img_path)

            # Orijinal boyutlar alınıyor
            width = self.label_58.width()
            height = self.label_58.height()

            # Görüntünün boyutlarını orijinal boyutlara göre ayarlamak için bir oran hesaplanır
            pixmap = pixmap.scaled(width, height, Qt.KeepAspectRatio)

            # Görüntüyü göstermek için etiketlere atanır
            self.label_58.setPixmap(pixmap)
            self.label_60.setPixmap(pixmap)

    def calculate_dark_green_properties(self):

        image = self.label_58.pixmap().toImage()
    
        # Resmi numpy dizisine dönüştür
        genişlik = image.width()
        yükseklik = image.height()
        buffer = image.bits().asstring(genişlik * yükseklik * 4)
        img = np.frombuffer(buffer, dtype=np.uint8).reshape((yükseklik, genişlik, 4))
    
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Matplotlib RGB formatını kullan
    
        # Resmi gri tonlamaya dönüştür
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # Gürültüyü azaltmak için Gaussian Bulanıklığı uygula
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
        # Koyu yeşil rengi algılamak için eşikleme uygula
        alt_yeşil = np.array([0, 100, 0], dtype="uint8")
        üst_yeşil = np.array([50, 255, 50], dtype="uint8")
        mask = cv2.inRange(image_rgb, alt_yeşil, üst_yeşil)
    
        # Eşiklenmiş resimde konturları bul
        konturlar, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
        # Özellikleri saklamak için bir DataFrame oluştur
        veri = []
    
        # Konturlar üzerinde döngü
        for i, kontur in enumerate(konturlar):
            # Konturun alanını hesapla
            alan = cv2.contourArea(kontur)
    
            # Konturun ağırlık merkezini hesapla
            M = cv2.moments(kontur)
            if M['m00'] != 0:  # Sıfıra bölme hatasını önle
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = 0, 0  # Alan sıfır ise merkez sıfır olacak
    
            # Konturun sınırlayıcı dikdörtgenini bul
            x, y, w, h = cv2.boundingRect(kontur)
    
            # Sınırlayıcı dikdörtgenin çapını (uzunluğunu) hesapla
            çap = np.sqrt(w ** 2 + h ** 2)
    
            # Enerji ve Entropi hesapla
            enerji = cv2.moments(blurred[y:y + h, x:x + w])['nu20'] + cv2.moments(blurred[y:y + h, x:x + w])['nu02']
            entropi = -np.sum(np.where((blurred[y:y + h, x:x + w] != 0),
                                       (blurred[y:y + h, x:x + w] / 255) * np.log(blurred[y:y + h, x:x + w] / 255), 0))
    
            # Ortalama ve Medyanı hesapla
            ortalama_değer = np.mean(blurred[y:y + h, x:x + w])
            medyan_değer = np.median(blurred[y:y + h, x:x + w])
    
            # Özellikleri DataFrame'e ekle
            veri.append([i + 1, (cx, cy), w, h, çap, enerji, entropi, ortalama_değer, medyan_değer])
    
        # DataFrame oluştur
        sütunlar = ["No", "Merkez", "Uzunluk", "Genişlik", "Çap", "Enerji", "Entropi", "Ortalama", "Medyan"]
        df = pd.DataFrame(veri, columns=sütunlar)
    
        # DataFrame'i Excel dosyasına kaydet
        df.to_excel("hiperspektral_özellikler.xlsx", index=False)
        print("Excel dosyası oluşturuldu.")


    def deblur_image(self):
        # Resmi al
        image = self.label_58.pixmap().toImage()

        # Resmi numpy dizisine dönüştür
        width = image.width()
        height = image.height()
        buffer = image.bits().asstring(width * height * 4)
        img = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 4))

        # Bulanıklığı azalt
        blurred = cv2.medianBlur(img, 3)  # Medyan filtreleme kullanarak bulanıklığı daha fazla azalt

        # Keskinleştirme
        kernel_sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # Keskinleştirme için kernel
        sharpened = cv2.filter2D(blurred, -1, kernel_sharpening)

        # Optik akışı kullanarak hareket eden nesnelerin etkisini alatma işelemi
        motion_blur_kernel = np.array([[1, 1, 1],
                                       [1, 3, 1],
                                       [1, 1, 1]]) / 9.0
        motion_blurred = cv2.filter2D(sharpened, -1, motion_blur_kernel)

        # Netlenmiş kareyi QPixmap'a dönüştür
        deblurred_image = cv2.cvtColor(motion_blurred.astype(np.uint8), cv2.COLOR_BGRA2RGB)  # uint8'ye dönüştür
        height, width, channel = deblurred_image.shape
        qImg = QImage(deblurred_image.data, width, height, QImage.Format_RGB888)
        deblurred_pixmap = QPixmap.fromImage(qImg)

        # Sonucu label_60'a ayarla
        self.label_60.setPixmap(deblurred_pixmap)



    def hough_functions(self):
        selected_index = int(self.comboBox10.currentIndex())
        if selected_index == 0:
            # Label_58'deki resmi al

            pixmap = self.label_58.pixmap()

            # Resmi QPixmap'ten numpy dizisine dönüştür

            img = pixmap.toImage()

            buffer = img.bits().asstring(img.width() * img.height() * 4)

            image = np.frombuffer(buffer, dtype=np.uint8).reshape((img.height(), img.width(), 4))

            # RGB formatına dönüştür

            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

            # Gri tonlamaya çevir

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Kenarları algıla
            edges = cv2.Canny(gray_image, 100, 250, apertureSize=3)

            # Hough dönüşümü uygula
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=125, minLineLength=125, maxLineGap=10)

            # Her bir çizgiyi resme ekle
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Sonucu QLabel içine yerleştir

            height, width, channel = image.shape

            bytesPerLine = 3 * width

            qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()

            pixmap = QPixmap(qImg)

            self.label_60.setPixmap(pixmap)

        elif selected_index == 1:

            # Label_58'deki resmi al

            pixmap = self.label_58.pixmap()

            # Resmi QPixmap'ten numpy dizisine dönüştür

            img = pixmap.toImage()

            buffer = img.bits().asstring(img.width() * img.height() * 4)

            image = np.frombuffer(buffer, dtype=np.uint8).reshape((img.height(), img.width(), 4))

            # RGB formatına dönüştür

            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

            # Gri tonlamaya çevir

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Yüz tespiti

            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

            # Göz tespiti için göz tespit sınıflandırıcısı

            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

            # Her bir yüz bölgesi için gözleri tespit et

            for (x, y, w, h) in faces:

                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Yüzü çerçevele

                # Yüz bölgesini al

                roi_gray = gray_image[y:y + h, x:x + w]

                roi_color = image[y:y + h, x:x + w]

                # Gözleri tespit et

                eyes = eye_cascade.detectMultiScale(roi_gray)

                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)  # Gözleri çerçevele

            # Sonucu QLabel içine yerleştir

            height, width, channel = image.shape

            bytesPerLine = 3 * width

            qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()

            pixmap = QPixmap(qImg)

            self.label_60.setPixmap(pixmap)

        else:
                return True

    def sigmoid_functions(self):

        def standard_sigmoid(x, alpha=1, beta=0):
            return 1 / (1 + np.exp(-alpha * (x - beta)))

        def sloped_sigmoid(x, alpha=1, beta=0, gamma=1):
            return 1 / (1 + np.exp(-alpha * (x - beta))) ** gamma

        def shifted_sigmoid(x, alpha=1, beta=0, gamma=1):
            return 1 / (1 + np.exp(-alpha * (x - beta))) ** (1 / gamma)

        def my_function(image):
            # Renkli görüntüyü gri tonlamaya dönüştür
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image

            # Histogram eşitleme uygula
            enhanced_image = cv2.equalizeHist(gray_image)

            # Renkli görüntüyse tekrar renklendir
            if len(image.shape) == 3:
                enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)

            return enhanced_image

        selected_index =  int(self.comboBox9.currentIndex())
        if selected_index == 0:
            alpha = 16
            beta = 0.5

            # Label_58'deki resmi al
            pixmap = self.label_58.pixmap()

            # Resmi QPixmap'ten numpy dizisine dönüştür
            img = pixmap.toImage()
            buffer = img.bits().asstring(img.width() * img.height() * 4)
            image = np.frombuffer(buffer, dtype=np.uint8).reshape((img.height(), img.width(), 4))

            # RGB formatına dönüştür
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

            # Gri tonlamaya çevir
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Sigmoid fonksiyonunu uygula
            enhanced_image = standard_sigmoid(gray_image / 255.0, alpha, beta)

            # Piksel değerlerini 0-255 aralığına getir
            enhanced_image = (enhanced_image * 255).astype(np.uint8)

            # Sonucu QImage'den QPixmap'e dönüştür
            height, width = enhanced_image.shape
            bytesPerLine = width
            qImg = QImage(enhanced_image.data, width, height, bytesPerLine, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qImg)

            # Sonucu görsel olarak görüntüle
            self.label_60.setPixmap(pixmap)


        elif selected_index == 1:
            alpha = 11
            beta = 0.6
            gamma = 1

            # Label_58'deki resmi al
            pixmap = self.label_58.pixmap()

            # Resmi QPixmap'ten numpy dizisine dönüştür
            img = pixmap.toImage()
            buffer = img.bits().asstring(img.width() * img.height() * 4)
            image = np.frombuffer(buffer, dtype=np.uint8).reshape((img.height(), img.width(), 4))

            # RGB formatına dönüştür
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

            # Gri tonlamaya çevir
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Sigmoid fonksiyonunu uygula
            enhanced_image = shifted_sigmoid(gray_image / 255.0, alpha, beta, gamma)

            # Piksel değerlerini 0-255 aralığına getir
            enhanced_image = (enhanced_image * 255).astype(np.uint8)

            # Sonucu QImage'den QPixmap'e dönüştür
            height, width = enhanced_image.shape[:2]

            bytesPerLine = width
            qImg = QImage(enhanced_image.data, width, height, bytesPerLine, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qImg)

            # Sonucu görsel olarak görüntüle
            self.label_60.setPixmap(pixmap)




        elif selected_index == 2:

            alpha = 13
            beta = 0.5
            gamma = 0.9

            # Label_58'deki resmi al
            pixmap = self.label_58.pixmap()

            # Resmi QPixmap'ten numpy dizisine dönüştür
            img = pixmap.toImage()
            buffer = img.bits().asstring(img.width() * img.height() * 4)
            image = np.frombuffer(buffer, dtype=np.uint8).reshape((img.height(), img.width(), 4))

            # RGB formatına dönüştür
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

            # Gri tonlamaya çevir
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Sigmoid fonksiyonunu uygula
            enhanced_image = sloped_sigmoid(gray_image / 255.0, alpha, beta, gamma)

            # Piksel değerlerini 0-255 aralığına getir
            enhanced_image = (enhanced_image * 255).astype(np.uint8)

            # Sonucu QImage'den QPixmap'e dönüştür
            height, width = enhanced_image.shape[:2]
            bytesPerLine = width
            qImg = QImage(enhanced_image.data, width, height, bytesPerLine, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qImg)

            # Sonucu görsel olarak görüntüle
            self.label_60.setPixmap(pixmap)


        elif selected_index == 3:

            # Label_58'deki resmi al

            pixmap = self.label_58.pixmap()

            # Resmi QPixmap'ten numpy dizisine dönüştür

            img = pixmap.toImage()

            buffer = img.bits().asstring(img.width() * img.height() * 4)

            image = np.frombuffer(buffer, dtype=np.uint8).reshape((img.height(), img.width(), 4))

            # RGB formatına dönüştür

            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

            # Histogram eşitleme işlemini uygula

            enhanced_image = my_function(image)

            # Sonucu QImage'den QPixmap'e dönüştür

            height, width = enhanced_image.shape[:2]

            bytesPerLine = width * 3  # Renkli görüntü olduğu için her piksel 3 byte'dır

            qImg = QImage(enhanced_image.data, width, height, bytesPerLine, QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(qImg)

            # Sonucu görsel olarak görüntüle

            self.label_60.setPixmap(pixmap)


        else:

            return True




