#Import cv2 dan os
import cv2
import os

video = cv2.VideoCapture(0)

deteksiwajah = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #Membuat objek detektor wajah menggunakan file XML yang terlatih untuk mendeteksi wajah.
recognizer = cv2.face.LBPHFaceRecognizer_create() #Membuat objek recognizer menggunakan algoritma LBPH
recognizer.read('Dataset/training.xml')

a = 0
while True:
    a = a + 1
    check, frame = video.read() #Membaca satu frame dari video.
    abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Mengonversi frame menjadi citra skala abu-abu untuk deteksi wajah.
    wajah = deteksiwajah.detectMultiScale(abu, 1.1, 5) #Mendeteksi wajah dalam citra skala abu-abu menggunakan metode multi-scale detection.
    for(x,y,w,h) in wajah:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2) #Menggambar kotak pembatas di sekitar wajah yang terdeteksi pada frame.
        id, conf = recognizer.predict(abu[y:y+h,x:x+w]) #: Menggunakan model recognizer untuk mengenali wajah yang terdeteksi.
        if id == 15:
            id = 'Joshep Rizz level 0'
        elif id == 7:
            id = 'Julianz Sigma Male'
        elif id == 11:
            id = 'Viano Mewing'
        elif id == 35:
            id = 'Tirta Budak itam'
        elif id == 30:
            id = 'Nafis gyat'
        else:
            id = "SUKI"
        cv2.putText(frame, str(id), (x+40, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0)) #Menampilkan teks dengan label pengenal pada frame di dekat wajah yang terdeteksi.
        
    cv2.imshow("Face Recognation", frame) #Menampilkan frame yang telah diproses dengan deteksi dan pengenalan wajah.
    key = cv2.waitKey(1)
    if key == ord('q'): #Jika tombol 'q' ditekan, keluar dari loop.
        break
    
video.release()
cv2.destroyAllWindows