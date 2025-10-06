import cv2
import torch
import timm
import requests
import psutil
import time
from PIL import Image
from timm.data import resolve_data_config, create_transform

# --- 1. Konfigurasi ---
# Ganti angka ini jika kamera utama Anda bukan 0 (misalnya di Mac seringkali 1)
CAMERA_INDEX = 0 


# --- 2. Deteksi Perangkat Keras Terbaik ---
# Prioritas: MPS (Apple Silicon) -> CUDA (NVIDIA Jetson/GPU) -> CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Menggunakan perangkat: {device}")


# --- 3. Load Model & Siapkan Transformasi ---
print("Memuat model MobileViT (mobilevit_s)...")
model = timm.create_model('mobilevit_s', pretrained=True)
model = model.to(device)
model.eval() # Set model ke mode evaluasi/inferensi
config = resolve_data_config({}, model=model)
transform = create_transform(**config)


# --- 4. Dapatkan Label Kelas ImageNet ---
# Mengunduh daftar label dari URL jika file tidak ada
try:
    with open("imagenet_classes.txt") as f:
        categories = [s.strip() for s in f.readlines()]
except FileNotFoundError:
    print("File 'imagenet_classes.txt' tidak ditemukan. Mengunduh dari internet...")
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = requests.get(url)
    response.raise_for_status() # Pastikan unduhan berhasil
    with open("imagenet_classes.txt", "w") as f:
        f.write(response.text)
    categories = [s.strip() for s in response.text.split("\n")]
    print("Daftar kelas berhasil diunduh dan disimpan.")


# --- 5. Inisialisasi untuk Statistik Jaringan & Loop ---
last_time = time.time()
last_net_usage = psutil.net_io_counters()
print("Membuka kamera... Tekan 'q' untuk keluar.")
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"Error: Kamera dengan indeks {CAMERA_INDEX} tidak dapat dibuka.")
    exit()

while True:
    # Ambil frame dari kamera
    ret, frame = cap.read()
    if not ret:
        print("Error: Tidak bisa menerima frame. Keluar...")
        break

    # --- Blok Inferensi Model ---
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    top1_prob, top1_catid = torch.topk(probabilities, 1)
    prediction_label = categories[top1_catid[0].item()]
    prediction_score = top1_prob[0].item()
    text_result = f"{prediction_label}: {prediction_score*100:.2f}%"
    cv2.putText(frame, text_result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # --- Blok Statistik Sistem ---
    cpu_usage = psutil.cpu_percent()
    ram = psutil.virtual_memory()
    ram_percent = ram.percent
    current_time = time.time()
    current_net_usage = psutil.net_io_counters()
    time_diff = current_time - last_time
    
    if time_diff > 0:
        # Hitung kecepatan dalam KB/s
        sent_speed = (current_net_usage.bytes_sent - last_net_usage.bytes_sent) / time_diff / 1024
        recv_speed = (current_net_usage.bytes_recv - last_net_usage.bytes_recv) / time_diff / 1024
    else:
        sent_speed, recv_speed = 0, 0
        
    last_time = current_time
    last_net_usage = current_net_usage

    # Format teks untuk ditampilkan
    cpu_text = f"CPU: {cpu_usage:.1f}%"
    ram_text = f"RAM: {ram_percent:.1f}% ({ram.used/1e9:.2f}/{ram.total/1e9:.2f} GB)"
    net_text = f"Down: {recv_speed:.1f} KB/s | Up: {sent_speed:.1f} KB/s"

    # Tampilkan teks statistik di bagian bawah frame
    cv2.putText(frame, cpu_text, (10, frame.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, ram_text, (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, net_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Tampilkan video
    cv2.imshow('MobileViT Real-time Classification', frame)

    # Keluar dari loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 6. Bersihkan ---
print("Menutup program.")
cap.release()
cv2.destroyAllWindows()
