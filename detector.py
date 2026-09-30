#!/usr/bin/env python3
"""
Generator Room Two-Person Rule - YOLO Presence Detector (Edge AI Sensor)
Detects occupancy count in generator rooms using YOLOv8 and transmits
payloads with lightweight snapshots to an MQTT Broker (Mosquitto/Node-RED).
All alerting logic, grace period, and Telegram notifications are handled by Node-RED.
"""

import argparse
import base64
import json
import logging
import sys
import time
import cv2
from ultralytics import YOLO

try:
    import paho.mqtt.client as mqtt
    from paho.mqtt.enums import CallbackAPIVersion
    PAHO_V2 = True
except (ImportError, AttributeError):
    import paho.mqtt.client as mqtt
    PAHO_V2 = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("RoomDetector")


class MQTTManager:
    """Manages MQTT connection, reconnection, and publishing."""

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
            self.client.on_connect = self._on_connect_v2
            self.client.on_disconnect = self._on_disconnect_v2
        else:
            self.client = mqtt.Client(client_id=client_id)
            self.client.on_connect = self._on_connect_v1
            self.client.on_disconnect = self._on_disconnect_v1

        self.client.reconnect_delay_set(min_delay=1, max_delay=30)

    def _on_connect_v2(self, client, userdata, flags, reason_code, properties=None):
        if reason_code == 0 or (hasattr(reason_code, "is_failure") and not reason_code.is_failure):
            self.is_connected = True
            logger.info(f"Connected to MQTT Broker at {self.broker}:{self.port}")
        else:
            self.is_connected = False
            logger.warning(f"Failed to connect to MQTT Broker, return code: {reason_code}")

    def _on_disconnect_v2(self, client, userdata, disconnect_flags, reason_code, properties=None):
        self.is_connected = False
        logger.warning(f"Disconnected from MQTT Broker: {reason_code}. Will attempt auto-reconnect...")

    def _on_connect_v1(self, client, userdata, flags, rc):
        if rc == 0:
            self.is_connected = True
            logger.info(f"Connected to MQTT Broker at {self.broker}:{self.port}")
        else:
            self.is_connected = False
            logger.warning(f"Failed to connect to MQTT Broker, rc: {rc}")

    def _on_disconnect_v1(self, client, userdata, rc):
        self.is_connected = False
        logger.warning(f"Disconnected from MQTT Broker, rc: {rc}. Auto-reconnecting...")

    def start(self):
        logger.info(f"Connecting to MQTT Broker {self.broker}:{self.port}...")
        try:
            self.client.connect_async(self.broker, self.port, keepalive=60)
            self.client.loop_start()
        except Exception as e:
            logger.error(f"Error initiating MQTT connection: {e}")
            try:
                self.client.loop_start()
            except Exception:
                pass

    def publish_occupancy(self, count: int, image_base64: str = None) -> bool:
        payload_dict = {
            "count": int(count),
            "timestamp": int(time.time())
        }
        if image_base64:
            payload_dict["image"] = image_base64

        payload_json = json.dumps(payload_dict)

        if not self.is_connected:
            logger.debug(f"[MQTT Offline] Suppressed publish (count={count})")
            return False

        try:
            res = self.client.publish(self.topic, payload_json, qos=1)
            img_info = f" [Included Snapshot: {len(image_base64)} chars]" if image_base64 else ""
            logger.info(f"Published to '{self.topic}': count={count}{img_info} (mid={res.mid})")
            return True
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            return False

    def stop(self):
        try:
            if self.client:
                self.client.loop_stop()
                self.client.disconnect()
                logger.info("MQTT Client disconnected.")
        except Exception as e:
            logger.error(f"Error disconnecting MQTT client: {e}")


def create_small_snapshot(frame, max_width=640, quality=55):
    """
    Downsamples and compresses a frame to a lightweight JPEG byte array (~20-30 KB).
    """
    h, w = frame.shape[:2]
    if w > max_width:
        target_h = int(h * (max_width / w))
        small = cv2.resize(frame, (max_width, target_h), interpolation=cv2.INTER_AREA)
    else:
        small = frame.copy()

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    ret, buf = cv2.imencode('.jpg', small, encode_params)
    if ret:
        return buf.tobytes()
    return None


def draw_hud(frame, count: int, mqtt_connected: bool, fps: float, broker_info: str):
    """Draws an informative HUD overlay on the video frame."""
    h, w = frame.shape[:2]

    hud_w = min(430, w - 20)
    hud_h = 135
    hud_x, hud_y = 15, 15

    # Create semi-transparent background for overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (20, 24, 30), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (60, 70, 85), 1)

    # 1. Header & MQTT Status
    cv2.putText(frame, "GENSET ROOM MONITOR", (hud_x + 12, hud_y + 24),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)

    mqtt_status_text = "CONNECTED" if mqtt_connected else "DISCONNECTED"
    mqtt_color = (0, 220, 0) if mqtt_connected else (0, 70, 240)
    status_x = hud_x + hud_w - 140
    cv2.circle(frame, (status_x - 8, hud_y + 20), 5, mqtt_color, -1)
    cv2.putText(frame, mqtt_status_text, (status_x, hud_y + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, mqtt_color, 1, cv2.LINE_AA)

    # Divider line
    cv2.line(frame, (hud_x + 10, hud_y + 34), (hud_x + hud_w - 10, hud_y + 34), (70, 80, 95), 1)

    # 2. Occupancy Count & Two-Person Rule State
    if count == 0:
        occupancy_color = (200, 200, 200)  # White / Grey
        rule_text = "STATUS: ROOM EMPTY"
    elif count == 1:
        occupancy_color = (0, 50, 255)    # Red (BGR)
        rule_text = "ALERT: 1 PERSON DETECTED (Sent to Node-RED)"
    else:
        occupancy_color = (0, 230, 0)    # Green (BGR)
        rule_text = f"COMPLIANT: 2-PERSON RULE ({count} Persons)"

    count_str = f"OCCUPANCY: {count} PERSON{'S' if count != 1 else ''}"
    cv2.putText(frame, count_str, (hud_x + 12, hud_y + 65),
                cv2.FONT_HERSHEY_DUPLEX, 0.72, occupancy_color, 2, cv2.LINE_AA)

    cv2.putText(frame, rule_text, (hud_x + 12, hud_y + 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, occupancy_color, 1, cv2.LINE_AA)

    # 3. Footer info (FPS and Broker info)
    footer_text = f"FPS: {fps:4.1f} | Broker: {broker_info}"
    cv2.putText(frame, footer_text, (hud_x + 12, hud_y + 118),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 170, 180), 1, cv2.LINE_AA)


def draw_bounding_boxes(frame, boxes, count: int):
    """Draws bounding boxes and labels on detected persons."""
    if count == 1:
        box_color = (0, 50, 255)   # Red
    elif count >= 2:
        box_color = (0, 220, 0)   # Green
    else:
        box_color = (200, 200, 200)

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        # Label background & text
        label = f"Person #{idx + 1} ({conf:.0%})"
        (lbl_w, lbl_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top_lbl_y = max(y1, lbl_h + 10)
        cv2.rectangle(frame, (x1, top_lbl_y - lbl_h - 6), (x1 + lbl_w + 8, top_lbl_y + 2), box_color, -1)
        cv2.putText(frame, label, (x1 + 4, top_lbl_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def run_detector(args):
    """Main detector loop."""
    # 1. Initialize MQTT
    mqtt_mgr = MQTTManager(broker=args.broker, port=args.port, topic=args.topic)
    mqtt_mgr.start()

    # 2. Load YOLO Model
    logger.info(f"Loading YOLO model '{args.model}'...")
    try:
        model = YOLO(args.model)
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        mqtt_mgr.stop()
        sys.exit(1)

    # 3. Open Video Source
    try:
        source_val = int(args.source)
    except ValueError:
        source_val = args.source

    logger.info(f"Opening video source: {source_val}")
    cap = cv2.VideoCapture(source_val)

    if not cap.isOpened():
        logger.error(f"Could not open video source '{source_val}'.")
        mqtt_mgr.stop()
        sys.exit(1)

    if isinstance(source_val, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 4. State variables for Rate Limiting & Heartbeat
    last_published_count = -1
    last_published_time = 0.0
    heartbeat_interval = args.heartbeat

    fps = 0.0
    fps_counter = 0
    fps_start_time = time.time()

    window_name = "Generator Room Occupancy Detector (Two-Person Rule)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    logger.info("Starting detection loop. Press 'q' to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if isinstance(source_val, str) and args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    logger.info("Video feed ended or frame read failed.")
                    break

            # Run YOLO inference
            results = model.predict(
                source=frame,
                classes=[0],
                conf=args.conf,
                verbose=False
            )

            boxes = results[0].boxes if len(results) > 0 else []
            current_count = len(boxes)
            current_time = time.time()

            # Draw visual annotations on the frame
            draw_bounding_boxes(frame, boxes, current_count)
            draw_hud(frame, current_count, mqtt_mgr.is_connected, fps, f"{args.broker}:{args.port}")

            # Rate Limiter & Heartbeat for MQTT
            count_changed = (current_count != last_published_count)
            heartbeat_due = (current_time - last_published_time >= heartbeat_interval)

            if count_changed or heartbeat_due:
                image_b64 = None
                # Attach snapshot whenever someone is in the room
                if current_count > 0:
                    snap_bytes = create_small_snapshot(frame, max_width=args.snap_width, quality=args.snap_quality)
                    if snap_bytes:
                        image_b64 = base64.b64encode(snap_bytes).decode('utf-8')

                mqtt_mgr.publish_occupancy(
                    count=current_count,
                    image_base64=image_b64
                )
                last_published_count = current_count
                last_published_time = current_time

            # Calculate FPS
            fps_counter += 1
            if current_time - fps_start_time >= 0.5:
                fps = fps_counter / (current_time - fps_start_time)
                fps_counter = 0
                fps_start_time = current_time

            cv2.imshow(window_name, frame)

            # Keyboard handler: 'q' or ESC (27) to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                logger.info("Exit requested by user.")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        logger.info("Cleaning up resources...")
        cap.release()
        cv2.destroyAllWindows()
        mqtt_mgr.stop()
        logger.info("Shutdown complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Generator Room Two-Person Rule - Edge AI Detector")
    parser.add_argument("--source", type=str, default="0",
                        help="Video source: camera index (0) or path to video file (.mp4)")
    parser.add_argument("--broker", type=str, default="localhost",
                        help="MQTT Broker host (default: localhost)")
    parser.add_argument("--port", type=int, default=1883,
                        help="MQTT Broker port (default: 1883)")
    parser.add_argument("--topic", type=str, default="genset/room/occupancy",
                        help="MQTT topic to publish occupancy count (default: genset/room/occupancy)")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="YOLO model path or name (default: yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Confidence threshold for person detection (default: 0.5)")
    parser.add_argument("--heartbeat", type=float, default=1.0,
                        help="Heartbeat publish interval in seconds (default: 1.0)")
    parser.add_argument("--snap-width", type=int, default=640,
                        help="Max width in pixels for small snapshot (default: 640)")
    parser.add_argument("--snap-quality", type=int, default=55,
                        help="JPEG compression quality (1-100) for snapshot (default: 55)")
    parser.add_argument("--loop", action="store_true",
                        help="Loop video file if source is a video file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_detector(args)
