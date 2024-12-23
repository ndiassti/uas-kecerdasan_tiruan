import cv2

# Memuat classifier Haar Cascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Membuka kamera (0 adalah ID default kamera, untuk kamera lain bisa menggunakan 1 atau lainnya)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Gagal membuka kamera! Periksa apakah kamera tersedia dan telah diizinkan.")
    exit()

while True:
    # Membaca frame dari kamera
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari kamera.")
        break

    # Mengonversi frame ke grayscale untuk deteksi wajah
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Mendeteksi wajah dalam frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Menggambar kotak di sekitar wajah yang terdeteksi
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Menampilkan frame dengan deteksi wajah
    cv2.imshow('Deteksi Wajah', frame)

    # Menutup aplikasi ketika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Melepaskan sumber daya
cap.release()
cv2.destroyAllWindows()
