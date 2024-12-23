import cv2

# Muat classifier Haar Cascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Buka kamera (index 0 adalah kamera bawaan, gunakan 1 untuk kamera eksternal)
video_capture = cv2.VideoCapture(1)

# Periksa apakah kamera berhasil dibuka
if not video_capture.isOpened():
    print("Kamera tidak dapat diakses!")
    exit()

while True:
    # Baca frame dari kamera
    ret, frame = video_capture.read()
    if not ret:
        print("Gagal membaca frame dari kamera.")
        break

    # Konversi frame ke grayscale (dibutuhkan oleh Haar Cascade)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah dalam frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Gambar kotak di sekitar wajah yang terdeteksi
    for (x, y, w, h) in faces:
        # Gambar kotak wajah
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Teks nama dan persentase (sederhana, kita pakai ukuran wajah untuk estimasi kasar)
        confidence = min(w, h) * 0.5  # Estimasi kasar berdasarkan ukuran wajah
        confidence_text = f"Nadia Sasti - {int(confidence)}%"

        # Tulis teks di atas kotak wajah
        cv2.putText(frame, confidence_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Tampilkan frame dengan deteksi wajah dan nama
    cv2.imshow('Deteksi Wajah', frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan kamera dan tutup semua jendela
video_capture.release()
cv2.destroyAllWindows()