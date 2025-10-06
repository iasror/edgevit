import cv2
import torch
import timm
import requests
import psutil
import time
from PIL import Image
from timm.data import resolve_data_config, create_transform

# --- 1. Konfigurasi ---
CAMERA_INDEX = 1 # Ganti ke indeks kamera Anda (misal: 0 atau 1)

# --- 2. Deteksi Perangkat Keras & Muat Model ---
# (Bagian ini sama seperti sebelumnya, tidak ada perubahan)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Menggunakan perangkat: {device}")
print("Memuat model MobileViT (mobilevit_s)...")
model = timm.create_model('mobilevit_s', pretrained=True)
model = model.to(device)
model.eval()
config = resolve_data_config({}, model=model)
transform = create_transform(**config)

# --- 3. Dapatkan Label Kelas ImageNet ---
# (Bagian ini sama seperti sebelumnya, tidak ada perubahan)
try:
    with open("imagenet_classes.txt") as f:
        categories = [s.strip() for s in f.readlines()]
except FileNotFoundError:
    print("Mengunduh daftar kelas...")
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = requests.get(url)
    with open("imagenet_classes.txt", "w") as f: f.write(response.text)
    categories = [s.strip() for s in response.text.split("\n")]

# --- 4. Inisialisasi & Loop Utama ---
last_time = time.time()
last_net_usage = psutil.net_io_counters()
print("Membuka kamera... Tekan 'q' untuk keluar.")
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"Error: Kamera dengan indeks {CAMERA_INDEX} tidak dapat dibuka.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Blok Inferensi & Statistik (Sama seperti sebelumnya) ---
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
    
    cpu_usage = psutil.cpu_percent()
    ram = psutil.virtual_memory()
    ram_percent = ram.percent
    current_time = time.time()
    current_net_usage = psutil.net_io_counters()
    time_diff = current_time - last_time
    if time_diff > 0:
        sent_speed = (current_net_usage.bytes_sent - last_net_usage.bytes_sent) / time_diff / 1024
        recv_speed = (current_net_usage.bytes_recv - last_net_usage.bytes_recv) / time_diff / 1024
    else:
        sent_speed, recv_speed = 0, 0
    last_time = current_time
    last_net_usage = current_net_usage
    cpu_text = f"CPU: {cpu_usage:.1f}%"
    ram_text = f"RAM: {ram_percent:.1f}% ({ram.used/1e9:.2f}/{ram.total/1e9:.2f} GB)"
    net_text = f"Down: {recv_speed:.1f} KB/s | Up: {sent_speed:.1f} KB/s"
    cv2.putText(frame, cpu_text, (10, frame.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, ram_text, (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, net_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # -----------------------------------------------------------------
    # --- BLOK BARU: Tambahkan Border di Sekeliling Gambar ---
    # -----------------------------------------------------------------
    border_thickness = 10
    border_color = (0, 255, 0) # Warna hijau dalam format BGR
    
    height, width, _ = frame.shape
    start_point = (0, 0)
    end_point = (width, height)
    
    cv2.rectangle(frame, start_point, end_point, border_color, border_thickness)
    # -----------------------------------------------------------------
    
    # Tampilkan video
    cv2.imshow('MobileViT Real-time Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 6. Bersihkan ---
print("Menutup program.")
cap.release()
cv2.destroyAllWindows()
