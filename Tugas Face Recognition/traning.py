 # Impor modul OpenCV,os,Modul Numpy
import cv2 
import os 
import numpy as np
from PIL import Image  # Impor kelas Image dari modul PIL untuk manipulasi gambar

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Mendefinisikan fungsi untuk memuat gambar wajah dan label dari dataset
def getImagesWithLabels(path):
    # Mengumpulkan path file gambar dari folder dataset
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # Inisialisasi list kosong untuk menyimpan sampel gambar wajah dan label (ID)
    faceSample = []
    Ids = []
    # Iterasi melalui setiap path file gambar dalam folder dataset
    for imagePath in imagePaths:
        # Membuka gambar menggunakan PIL dan mengonversinya ke skala abu-abu
        pilImage = Image.open(imagePath).convert('L')
        # Mengonversi gambar menjadi array NumPy dengan tipe data uint8
        imageNp = np.array(pilImage, 'uint8')
        # Mendapatkan ID (label) dari nama file gambar
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        # Mendeteksi wajah dalam gambar menggunakan detektor wajah yang telah dibuat sebelumnya
        wajah = detector.detectMultiScale(imageNp)
        # Iterasi melalui setiap wajah yang terdeteksi dalam gambar
        for (x, y, w, h) in wajah:
            # Menambahkan sampel gambar wajah dan label yang sesuai ke dalam list
            faceSample.append(imageNp[y:y+h, x:x+w])
            Ids.append(Id)
    # Mengembalikan dua list yang berisi sampel gambar wajah dan label yang terkait
    return faceSample, Ids

# Memuat sampel gambar wajah dan label dari folder dataset
wajah, Ids = getImagesWithLabels('Dataset')
# Melatih recognizer menggunakan sampel gambar wajah dan label yang telah dimuat
recognizer.train(wajah, np.array(Ids))
# Menyimpan model yang telah dilatih dalam file XML untuk digunakan nanti dalam pengenalan wajah
recognizer.save('Dataset/training.xml')
