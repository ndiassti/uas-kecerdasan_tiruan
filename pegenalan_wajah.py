import cv2
import numpy as np

# Muat model YOLOv4
yolo_net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

# Muat file kelas YOLO (untuk deteksi objek)
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Buka kamera
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    # Deteksi objek dengan YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    outputs = yolo_net.forward(output_layers)

    # Mencari wajah dalam deteksi YOLO (class ID 0 = "person" di coco.names)
    faces = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Class 0 adalah 'person' di COCO
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = center_x - w // 2
                y = center_y - h // 2
                faces.append((x, y, w, h))  # Menyimpan koordinat wajah

    # Menampilkan kotak dan label untuk wajah yang terdeteksi
    for (x, y, w, h) in faces:
        # Gambar kotak hijau di sekitar wajah yang terdeteksi
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Menampilkan label "Nadia Sasti" atau "Person" pada wajah yang terdeteksi
        cv2.putText(frame, "Person", (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Tampilkan frame dengan wajah yang terdeteksi
    cv2.imshow('YOLOv4 Face Detection', frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan kamera dan tutup semua jendela
video_capture.release()
cv2.destroyAllWindows()
