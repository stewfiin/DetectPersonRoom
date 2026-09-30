# Generator Room Two-Person Rule - YOLO Presence Detector

Sistem monitoring cerdas berbasis **Computer Vision (YOLOv8)** dan **IoT (MQTT + Node-RED)** untuk menegakkan aturan keselamatan kerja **Two-Person Rule** di ruang generator (genset). 

Sistem mendeteksi keberadaan orang secara real-time dan mengirimkan payload occupancy count serta foto bukti snapshot ke MQTT Broker dan Telegram Bot jika terdeteksi pelanggaran.

---

## 🌟 Fitur Utama
- **Real-time Person Detection**: Menggunakan YOLOv8 Nano (`yolov8n.pt`) yang ringan dan cepat untuk deteksi manusia (`class 0: person`).
- **Two-Person Rule Enforcement**: Memantau apakah jumlah orang di ruang berisiko tinggi memenuhi standar minimal (2 orang).
- **Throttling & Heartbeat**: Hanya mem-publish data saat status jumlah orang berubah atau setiap 1 detik sebagai heartbeat.
- **Lightweight Snapshot Capture**: Mengambil snapshot bukti pelanggaran berukuran kecil (~20-30 KB) dan mengirimkannya ke payload MQTT.
- **Node-RED & Telegram Alerting**: Integrasi mudah dengan Node-RED untuk evaluasi aturan, cooldown anti-spam, dan pengiriman foto bukti ke Telegram Bot.
- **Heads-Up Display (HUD)**: Menampilkan status koneksi MQTT, jumlah okupansi, dan FPS secara real-time pada tampilan video.

---

## 🏗️ Tech Stack
- **AI & Vision**: Ultralytics YOLOv8, PyTorch, OpenCV
- **Messaging Protocol**: MQTT (Eclipse Mosquitto)
- **Automation / Workflow**: Node-RED
- **Notification**: Telegram Bot API
- **Containerization**: Docker & Docker Compose

---

## 🚀 Panduan Instalasi & Menjalankan

### 1. Prasyarat
- Python 3.10+
- Docker & Docker Compose

### 2. Jalankan MQTT Broker & Node-RED
```bash
docker compose up -d
```
- Mosquitto MQTT Broker berjalan di port `1883`
- Node-RED UI dapat diakses di `http://localhost:1880`

### 3. Siapkan Virtual Environment & Install Dependensi
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1   # (Windows PowerShell)
pip install -r requirements.txt
```

### 4. Jalankan Detektor
- **Menggunakan Webcam Default (Device 0)**:
  ```bash
  python detector.py
  # atau gunakan shortcut:
  .\run.bat
  ```

- **Menggunakan File Video (.mp4)**:
  ```bash
  python detector.py --source "path/to/video.mp4" --loop
  ```

- **Kustomisasi Parameter**:
  ```bash
  python detector.py --conf 0.4 --snap-width 640 --snap-quality 55
  ```

---

## 📲 Setup Notifikasi Telegram via Node-RED
1. Buka browser ke `http://localhost:1880`.
2. Klik menu di pojok kanan atas $\rightarrow$ **Import** $\rightarrow$ pilih/paste isi file `nodered_flow.json`.
3. Buka node function **Cek Pelanggaran & Siapkan Foto Telegram**, masukkan `TELEGRAM_BOT_TOKEN` dan `TELEGRAM_CHAT_ID` Anda.
4. Klik **Deploy**.

---

## 📄 Lisensi
MIT License
