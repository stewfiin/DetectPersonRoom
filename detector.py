#!/usr/bin/env python3
"""
Generator Room Two-Person Rule - YOLO Presence Detector
Fitur:
- Deteksi 2-Person Rule dengan jeda toleransi (Grace Period 5s)
- Overlay CCTV: FPS, Nama Ruangan, Tanggal & Jam, Status Deteksi
- Snapshot yang dikirim ke Telegram memuat seluruh anotasi visual (Bounding Box & HUD)
- Penanganan clean exit (Ctrl+C atau tombol Q)
"""

import argparse
import base64
import json
import logging
import sys
import time
from datetime import datetime
import cv2
from ultralytics import YOLO

try:
    import paho.mqtt.client as mqtt
    from paho.mqtt.enums import CallbackAPIVersion
    PAHO_V2 = True
except (ImportError, AttributeError):
    import paho.mqtt.client as mqtt
    PAHO_V2 = False

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("RoomDetector")


class MQTTManager:
    def __init__(self, broker: str = "localhost", port: int = 1883, topic: str = "genset/room/occupancy"):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.is_connected = False
        self.client = None
        self._init_client()

    def _init_client(self):
        client_id = f"genset-detector-{int(time.time())}"
        if PAHO_V2:
            self.client = mqtt.Client(CallbackAPIVersion.VERSION2, client_id=client_id)
            self.client.on_connect = lambda c, u, f, rc, p=None: setattr(self, 'is_connected', True)
            self.client.on_disconnect = lambda c, u, df, rc, p=None: setattr(self, 'is_connected', False)
        else:
            self.client = mqtt.Client(client_id=client_id)
            self.client.on_connect = lambda c, u, f, rc: setattr(self, 'is_connected', True)
            self.client.on_disconnect = lambda c, u, rc: setattr(self, 'is_connected', False)

    def start(self):
        try:
            self.client.connect_async(self.broker, self.port, keepalive=60)
            self.client.loop_start()
        except Exception as e:
            logger.error(f"Error koneksi MQTT: {e}")

    def publish_violation(self, count: int, image_base64: str = None, room_name: str = ""):
        payload = {
            "alert": True,
            "status": "VIOLATION",
            "room_name": room_name,
            "count": int(count),
            "timestamp": int(time.time()),
            "image": image_base64
        }
        if self.is_connected:
            self.client.publish(self.topic, json.dumps(payload), qos=1)
            logger.info("🚨 Sinyal PELANGGARAN (>5 detik) + Foto teranotasi berhasil dikirim ke Node-RED!")

    def stop(self):
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()


def create_snapshot(frame, max_width=800, quality=70):
    """Kompresi snapshot dengan tetap menjaga keterbacaan teks dan kotak deteksi."""
    h, w = frame.shape[:2]
    if w > max_width:
        target_h = int(h * (max_width / w))
        small = cv2.resize(frame, (max_width, target_h), interpolation=cv2.INTER_AREA)
    else:
        small = frame.copy()
    ret, buf = cv2.imencode('.jpg', small, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ret else None


def draw_cctv_hud(frame, room_name: str, fps: float, count: int, status_text: str, theme_color: tuple):
    """Menggambar panel HUD gaya CCTV profesional."""
    h, w = frame.shape[:2]

    # 1. Overlay Header (Nama Ruangan, Waktu & FPS)
    panel_w = min(540, w - 20)
    panel_h = 140
    cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (20, 20, 25), -1)
    cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (70, 75, 85), 1)

    # Nama Ruangan & CCTV Watermark
    cv2.putText(frame, f"CAM: {room_name.upper()}", (22, 35),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Info FPS & Waktu Real-time
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, f"FPS: {fps:4.1f}  |  {time_str}", (22, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 180, 190), 1, cv2.LINE_AA)

    # Garis Pembatas
    cv2.line(frame, (20, 72), (10 + panel_w - 10, 72), (60, 65, 75), 1)

    # Baris 1 Status: Jumlah Orang
    count_str = f"DETEKSI: {count} ORANG"
    cv2.putText(frame, count_str, (22, 100),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, theme_color, 2, cv2.LINE_AA)

    # Baris 2 Status: Keterangan Status Visual & Countdown
    status_label = f"STATUS : {status_text}"
    cv2.putText(frame, status_label, (22, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, theme_color, 1, cv2.LINE_AA)


def run_detector(args):
    # Batasi CPU threads PyTorch agar tidak membebani semua core
    if args.threads > 0:
        try:
            import torch
            torch.set_num_threads(args.threads)
            logger.info(f"Membatasi PyTorch CPU thread ke: {args.threads}")
        except Exception as e:
            logger.warning(f"Gagal membatasi torch thread: {e}")

    mqtt_mgr = MQTTManager(broker=args.broker, port=args.port, topic=args.topic)
    mqtt_mgr.start()

    logger.info(f"Memuat model YOLO: {args.model} (imgsz: {args.imgsz})")
    model = YOLO(args.model)

    source_val = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source_val)

    if not cap.isOpened():
        logger.error(f"Gagal membuka sumber video: {source_val}")
        mqtt_mgr.stop()
        sys.exit(1)

    # 1. Minta resolusi 16:9 Widescreen HD (1280x720) jika sumber webcam
    if isinstance(source_val, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Deteksi jika sumber adalah file video (agar tidak running melebihi FPS aslinya)
    is_video_file = isinstance(source_val, str) and not source_val.isdigit()
    video_fps = cap.get(cv2.CAP_PROP_FPS) if is_video_file else 0
    if is_video_file and (video_fps <= 0 or video_fps > 120):
        video_fps = 30.0
    frame_target_time = (1.0 / video_fps) if is_video_file else 0.0

    # Logika Toleransi (Grace Period 5 Detik)
    GRACE_PERIOD = 5.0
    one_person_start_time = None
    alert_already_sent = False

    # Perhitungan FPS
    fps = 0.0
    fps_counter = 0
    fps_start = time.time()

    # Variabel Frame Skipping untuk hemat CPU
    frame_counter = 0
    last_boxes = []

    cv2.namedWindow("CCTV Monitor", cv2.WINDOW_NORMAL)

    try:
        while cap.isOpened():
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                if isinstance(source_val, str) and args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            # Hitung FPS tampilan setiap 0.5 detik
            fps_counter += 1
            now = time.time()
            if now - fps_start >= 0.5:
                fps = fps_counter / (now - fps_start)
                fps_counter = 0
                fps_start = now

            frame_counter += 1
            # Jalankan inferensi YOLO hanya setiap (skip_frames + 1) frame
            run_inference = (args.skip_frames <= 0) or (frame_counter % (args.skip_frames + 1) == 0)

            if run_inference:
                results = model.predict(source=frame, classes=[0], conf=args.conf, imgsz=args.imgsz, verbose=False)
                boxes = results[0].boxes if len(results) > 0 else []
                last_boxes = boxes
            else:
                boxes = last_boxes

            count = len(boxes)
            current_time = time.time()
            trigger_alert_now = False

            # Logika Hitung Mundur 5 - 0 Detik (Grace Period)
            if count == 1:
                if one_person_start_time is None:
                    one_person_start_time = current_time
                    alert_already_sent = False

                elapsed = current_time - one_person_start_time

                if elapsed < GRACE_PERIOD:
                    remaining = GRACE_PERIOD - elapsed
                    status_text = f"Tunggu Orang ke-2 ({remaining:.1f}s)"
                    theme_color = (0, 165, 255)  # Oranye
                else:
                    status_text = "PELANGGARAN (ALERT)"
                    theme_color = (0, 0, 255)    # Merah

                    # Kirim alert hanya sekali setelah 5 detik terlampaui
                    if not alert_already_sent:
                        trigger_alert_now = True
                        alert_already_sent = True

            elif count >= 2:
                one_person_start_time = None
                alert_already_sent = False
                status_text = f"AMAN ({count} ORANG)"
                theme_color = (0, 220, 0)  # Hijau

            else:
                one_person_start_time = None
                alert_already_sent = False
                status_text = "RUANGAN KOSONG"
                theme_color = (180, 180, 180) # Abu-abu

            # 1. Gambar Bounding Box Orang yang Terdeteksi
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), theme_color, 2)
                
                label = f"Person #{idx + 1} ({conf:.0%})"
                cv2.rectangle(frame, (x1, y1 - 22), (x1 + 130, y1), theme_color, -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

            # 2. Gambar HUD CCTV (Dengan hitung mundur 5.0 -> 0.0s)
            draw_cctv_hud(frame, args.room_name, fps, count, status_text, theme_color)

            # 3. Kirim Sinyal ALERT + Foto ke Node-RED HANYA setelah lewat 5 detik
            if trigger_alert_now:
                snap_bytes = create_snapshot(frame)
                img_b64 = base64.b64encode(snap_bytes).decode('utf-8') if snap_bytes else None
                mqtt_mgr.publish_violation(count=1, image_base64=img_b64, room_name=args.room_name)

            # 4. Tampilkan pada jendela (otomatis Full Window tanpa area abu-abu kosong saat dimaksimalkan)
            try:
                _, _, win_w, win_h = cv2.getWindowImageRect("CCTV Monitor")
                if win_w > 100 and win_h > 100 and (win_w != frame.shape[1] or win_h != frame.shape[0]):
                    display_frame = cv2.resize(frame, (win_w, win_h), interpolation=cv2.INTER_LINEAR)
                else:
                    display_frame = frame
            except Exception:
                display_frame = frame

            cv2.imshow("CCTV Monitor", display_frame)

            # Pengaturan jeda waktu / FPS pacing untuk menghemat CPU
            if is_video_file:
                elapsed = time.time() - loop_start
                delay_ms = max(1, int((frame_target_time - elapsed) * 1000))
            else:
                delay_ms = 1

            # Tombol Q untuk keluar
            if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
                logger.info("Program dihentikan oleh user via tombol Q.")
                break

    except KeyboardInterrupt:
        logger.info("Program dihentikan oleh user (Ctrl + C).")
    finally:
        logger.info("Membersihkan resource kamera & koneksi...")
        cap.release()
        cv2.destroyAllWindows()
        mqtt_mgr.stop()
        logger.info("Selesai.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="Camera index (0) atau file video (.mp4)")
    parser.add_argument("--room-name", type=str, default="Ruang Generator Utama", help="Nama ruangan yang ditampilkan di CCTV")
    parser.add_argument("--broker", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--topic", type=str, default="genset/room/occupancy")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--loop", action="store_true")
    # Parameter penghemat CPU
    parser.add_argument("--skip-frames", type=int, default=2, help="Lompati deteksi sebanyak N frame untuk menghemat CPU (default: 2, 0 untuk tiap frame)")
    parser.add_argument("--imgsz", type=int, default=480, help="Resolusi inferensi YOLO (default: 480, standar: 640, sangat ringan: 320)")
    parser.add_argument("--threads", type=int, default=2, help="Batas jumlah CPU thread PyTorch (default: 2)")
    args = parser.parse_args()
    run_detector(args)