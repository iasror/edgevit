import cv2
import torch
import torchvision
from torchvision import transforms
import collections
from PIL import Image
import numpy as np

# --- 1. Konfigurasi ---
CAMERA_INDEX = 1
CLIP_LENGTH = 16
BUFFER_SIZE = 100
INFERENCE_INTERVAL = 8
MODEL_INPUT_SIZE = 128
MODEL_CROP_SIZE = 112
PREDICTION_BUFFER_SIZE = 30

# --- 2. Deteksi Perangkat & Muat Model ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Menggunakan perangkat: {device}")
print("Memuat model Action Recognition (R(2+1)D)...")
model = torchvision.models.video.r2plus1d_18(pretrained=True)
model = model.to(device)
model.eval()

# --- 3. Dapatkan Label & Siapkan Transformasi ---
with open("kinetics_400_labels.csv") as f:
    categories = [line.strip() for line in f.readlines()]
transform = transforms.Compose([
    transforms.Resize(MODEL_INPUT_SIZE),
    transforms.CenterCrop(MODEL_CROP_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
])

# --- 4. Inisialisasi & Loop Utama ---
frame_buffer = collections.deque(maxlen=BUFFER_SIZE)
display_prediction = "Initializing..."
prediction_buffer_count = 0
frame_count = 0

print("Membuka kamera... Lakukan aksi! Tekan 'q' untuk keluar.")
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"Error: Kamera dengan indeks {CAMERA_INDEX} tidak dapat dibuka.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    frame_buffer.append(frame)

    # --- Lakukan inferensi pada interval tertentu ---
    if frame_count % INFERENCE_INTERVAL == 0 and len(frame_buffer) >= CLIP_LENGTH:
        clip_frames = list(frame_buffer)[-CLIP_LENGTH:]
        clip = []
        for f in clip_frames:
            img_pil = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            transformed_frame = transform(img_pil)
            clip.append(transformed_frame)
        
        input_tensor = torch.stack(clip, dim=1)
        input_for_model = input_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_for_model)
        
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top1_prob, top1_catid = torch.topk(probabilities, 1)
        
        prediction_label = categories[top1_catid[0].item()]
        prediction_score = top1_prob[0].item()
        
        display_prediction = f"{prediction_label}: {prediction_score*100:.1f}%"
        prediction_buffer_count = PREDICTION_BUFFER_SIZE

    # --- Tampilkan prediksi di layar utama ---
    if prediction_buffer_count > 0:
        cv2.putText(frame, display_prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        prediction_buffer_count -= 1
    
    # --- Tampilkan video di layar utama ---
    cv2.imshow('Layar Utama - Realtime', frame)

    # -----------------------------------------------------------------
    # --- BLOK BARU: Visualisasi Buffer di Layar Kedua ---
    # -----------------------------------------------------------------
    if len(frame_buffer) > 0:
        # Buat thumbnail kecil untuk setiap frame di buffer
        thumbnail_height = 90
        thumbnails = []
        for f in list(frame_buffer):
            h, w, _ = f.shape
            scale = thumbnail_height / h
            thumbnail_width = int(w * scale)
            thumb = cv2.resize(f, (thumbnail_width, thumbnail_height))
            thumbnails.append(thumb)
        
        # Gabungkan semua thumbnail secara horizontal
        buffer_visualization = cv2.hconcat(thumbnails)
        
        # Tampilkan di jendela kedua
        cv2.imshow('Layar Kedua - Processing Buffer', buffer_visualization)
    # -----------------------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 5. Bersihkan ---
print("Menutup program.")
cap.release()
cv2.destroyAllWindows()
