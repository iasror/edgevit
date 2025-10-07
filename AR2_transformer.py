import cv2
import torch
import collections
import numpy as np
from transformers import VideoMAEImageProcessor, TimesformerForVideoClassification
from PIL import Image
from huggingface_hub import login

# --- 1. Konfigurasi ---
CAMERA_INDEX = 0                 # Ganti ke indeks kamera Anda
CLIP_LENGTH = 8                  # TimeSformer base dilatih dengan 8 frame
PREDICTION_BUFFER_SIZE = 30

# --- 2. Deteksi Perangkat & Muat Model & Processor ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Menggunakan perangkat: {device}")

# Ganti "hf_xxx..." dengan token Anda yang sebenarnya
login(token="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") 
# --------------------------------

print("Memuat model TimeSformer...")
# Model akan diunduh otomatis saat pertama kali dijalankan
MODEL_NAME = "facebook/timesformer-base-finetuned-kinetics-400"
model = TimesformerForVideoClassification.from_pretrained(MODEL_NAME)
processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)

model = model.to(device)
model.eval()

# --- 3. Inisialisasi & Loop Utama ---
frame_buffer = collections.deque(maxlen=CLIP_LENGTH)
display_prediction = "Initializing..."
prediction_buffer_count = 0

print("Membuka kamera... Lakukan aksi! Tekan 'q' untuk keluar.")
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"Error: Kamera dengan indeks {CAMERA_INDEX} tidak dapat dibuka.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Konversi frame ke RGB dan tambahkan ke buffer
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_buffer.append(frame_rgb)

    # --- Jika buffer sudah penuh, lakukan inferensi ---
    if len(frame_buffer) == CLIP_LENGTH:
        
        # 1. Siapkan input untuk processor
        # Processor butuh list dari NumPy array
        clip = list(frame_buffer)
        
        # 2. Gunakan processor untuk menyiapkan input sesuai kebutuhan model
        inputs = processor(clip, return_tensors="pt")
        inputs = inputs.to(device)

        # 3. Lakukan inferensi
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # 4. Dapatkan prediksi
        probabilities = torch.nn.functional.softmax(logits[0], dim=0)
        top1_prob, top1_catid = torch.topk(probabilities, 1)
        
        # Dapatkan label dari konfigurasi model
        prediction_label = model.config.id2label[top1_catid[0].item()]
        prediction_score = top1_prob[0].item()
        
        display_prediction = f"{prediction_label}: {prediction_score*100:.1f}%"
        prediction_buffer_count = PREDICTION_BUFFER_SIZE
        
        # Kosongkan buffer untuk memulai klip baru (pendekatan tumbling window)
        frame_buffer.clear()

    # Tampilkan prediksi di layar (menggunakan frame asli BGR)
    if prediction_buffer_count > 0:
        cv2.putText(frame, display_prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        prediction_buffer_count -= 1

    # Tampilkan video
    cv2.imshow('Real-time Action Recognition (TimeSformer)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 4. Bersihkan ---
print("Menutup program.")
cap.release()
cv2.destroyAllWindows()
