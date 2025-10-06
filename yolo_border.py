import cv2
import torch
from ultralytics import YOLO

# --- 1. Konfigurasi ---
CAMERA_INDEX = 0 # Ganti ke indeks kamera Anda (misal: 0 atau 1)
MODEL_NAME = 'yolov8n.pt' # 'n' adalah versi nano, yang paling ringan dan cepat

# --- 2. Deteksi Perangkat & Muat Model YOLO ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Menggunakan perangkat: {device}")

print(f"Memuat model YOLO ({MODEL_NAME})...")
# Model akan diunduh secara otomatis saat pertama kali dijalankan
model = YOLO(MODEL_NAME)
model.to(device)


# --- 3. Loop Deteksi Real-time ---
print("Membuka kamera... Tekan 'q' untuk keluar.")
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"Error: Kamera dengan indeks {CAMERA_INDEX} tidak dapat dibuka.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Lakukan Deteksi Objek ---
    # Berikan frame langsung ke model
    results = model(frame, stream=True)

    # --- Loop melalui hasil deteksi ---
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Dapatkan koordinat kotak (bounding box)
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # konversi ke integer

            # Dapatkan skor kepercayaan (confidence)
            conf = box.conf[0]
            
            # Dapatkan nama kelas
            cls_index = int(box.cls[0])
            class_name = model.names[cls_index]

            # --- Gambar Kotak dan Label di Frame ---
            # Hanya gambar jika kepercayaan di atas ambang batas (misal: 25%)
            if conf > 0.25:
                # Gambar kotak (border)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Warna hijau, ketebalan 2

                # Siapkan teks label
                label = f'{class_name} {conf*100:.1f}%'

                # Gambar latar belakang untuk teks
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - h - 5), (x1 + w, y1), (0, 255, 0), -1)

                # Tulis teks di atas kotak
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Tampilkan video
    cv2.imshow('YOLOv8 Real-time Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 4. Bersihkan ---
print("Menutup program.")
cap.release()
cv2.destroyAllWindows()
