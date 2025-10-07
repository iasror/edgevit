import cv2
import torch
import torchvision
from torchvision.transforms import v2 as transforms
import collections

# --- 1. Konfigurasi ---
CAMERA_INDEX = 0                 # Ganti ke indeks kamera Anda
CLIP_LENGTH = 16                 # Jumlah frame dalam satu 'clip' untuk dianalisis
MODEL_INPUT_SIZE = (128, 171)    # (Tinggi, Lebar) untuk resize
MODEL_CROP_SIZE = (112, 112)     # (Tinggi, Lebar) untuk crop
PREDICTION_BUFFER_SIZE = 30      # Tampilkan prediksi selama 30 frame

# --- 2. Deteksi Perangkat & Muat Model ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Menggunakan perangkat: {device}")

print("Memuat model Action Recognition (R(2+1)D)...")
model = torchvision.models.video.r2plus1d_18(pretrained=True)
model = model.to(device)
model.eval()

# --- 3. Dapatkan Label Kelas Kinetics-400 ---
with open("kinetics_400_labels.csv") as f:
    categories = [line.strip() for line in f.readlines()]

# --- 4. Siapkan Transformasi Input Video ---
# Transformasi ini akan diterapkan ke seluruh 'clip'
transform = transforms.Compose([
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Resize(MODEL_INPUT_SIZE),
    transforms.CenterCrop(MODEL_CROP_SIZE),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
])

# --- 5. Inisialisasi & Loop Utama ---
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

    # Tambahkan frame ke buffer setelah di-resize (format HWC, BGR)
    frame_buffer.append(frame)

    # --- Jika buffer sudah penuh, lakukan inferensi ---
    if len(frame_buffer) == CLIP_LENGTH:
        # 1. Konversi buffer (list of NumPy arrays) ke tensor
        # Awal: (T, H, W, C) -> (T, C, H, W)
        input_tensor = torch.from_numpy(
            cv2.cvtColor(
                cv2.vconcat(list(frame_buffer)), cv2.COLOR_BGR2RGB)
            ).view(CLIP_LENGTH, frame.shape[0], frame.shape[1], 3).permute(0, 3, 1, 2)
        
        # 2. Terapkan transformasi
        input_tensor_transformed = transform(input_tensor)
        
        # 3. Ubah urutan dimensi untuk model: (T, C, H, W) -> (C, T, H, W)
        input_for_model = input_tensor_transformed.permute(1, 0, 2, 3).unsqueeze(0).to(device)

        # 4. Lakukan inferensi
        with torch.no_grad():
            outputs = model(input_for_model)
        
        # 5. Dapatkan prediksi
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top1_prob, top1_catid = torch.topk(probabilities, 1)
        
        prediction_label = categories[top1_catid[0].item()]
        prediction_score = top1_prob[0].item()
        
        # Simpan prediksi untuk ditampilkan
        display_prediction = f"{prediction_label}: {prediction_score*100:.1f}%"
        prediction_buffer_count = PREDICTION_BUFFER_SIZE # Reset buffer tampilan

    # Tampilkan prediksi di layar
    if prediction_buffer_count > 0:
        cv2.putText(frame, display_prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        prediction_buffer_count -= 1

    # Tampilkan video
    cv2.imshow('Real-time Action Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 6. Bersihkan ---
print("Menutup program.")
cap.release()
cv2.destroyAllWindows()
