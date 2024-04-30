import cv2

video = cv2.VideoCapture(0)
deteksiwajah = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #Membuat objek detektor wajah menggunakan file XML yang terlatih untuk mendeteksi wajah.

id = input('ID : ') #Masukan id yang mau di masukan di dataset
a = 0
while True:
    a = a + 1
    check, frame = video.read()
    abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #mengubah warna menjadi hitam putih
    wajah = deteksiwajah.detectMultiScale(abu, 1.1, 5) #mendeteksi wajah dengan ukuran tersebut
    print(wajah)
    for(x,y,w,h) in wajah:
        cv2.imwrite('Dataset/User.' + str(id) + '.' + str(a) + '.jpg', abu[y:y+h, x:x+w]) #Mengsave di folder dataset dengan id dan berformat jpg
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) #Menggambar kotak pembatas di sekitar wajah yang terdeteksi pada frame.
    cv2.imshow("Face Recognition Window", frame) #Menampilkan frame asli yang telah diberi kotak dan nama wajah jika ada.
    if (a > 200): #Memeriksa gambar apakah sudah 100 gambar yang di simpan di dataset
        break
video.release()
cv2.destroyAllWindows()