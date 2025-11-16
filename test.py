import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import os

# ================================
# 1. LOAD YOLO MODEL
# ================================
model = YOLO('yolov8n.pt')
print("Đã tải xong model YOLOv8.")

# --- Chỉ phát hiện class 'cup' ---
TARGET_CLASS = 'cup'
cup_class_id = None

for cls_id, name in model.names.items():
    if name == TARGET_CLASS:
        cup_class_id = cls_id
        break

if cup_class_id is None:
    print(f"❌ Không tìm thấy class '{TARGET_CLASS}' trong YOLO model!")
    exit()
else:
    print(f"✔ Chỉ detect class: '{TARGET_CLASS}' (ID={cup_class_id})")


# ================================
# 2. CONFIG PIPELINE - RGB ONLY
# ================================
pipeline = rs.pipeline()
config = rs.config()

bag_file = "20251112_135756.bag"

if not os.path.exists(bag_file):
    print(f"❌ Không tìm thấy file: {bag_file}")
    exit()

# Cho phép pipeline đọc file .bag
rs.config.enable_device_from_file(config, bag_file)

# KHÔNG enable stream depth
# Chỉ enable color stream
config.enable_stream(rs.stream.color)

# ================================
# 3. START PIPELINE
# ================================
profile = pipeline.start(config)

print("✔ Pipeline STARTED (RGB only). Nhấn 'q' để thoát.")

try:
    while True:
        # Lấy frame
        frames = pipeline.wait_for_frames()
        if not frames:
            print("❌ Hết frame.")
            break

        # Lấy COLOR frame
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Chuyển sang numpy
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        # ================================
        # 4. CHẠY YOLO (ONLY CUP)
        # ================================
        results = model.predict(
            color_image,
            classes=[cup_class_id],
            verbose=False
        )

        annotated_image = results[0].plot()
        cv2.imshow("YOLOv8 - RGB Only - Detect CUP", annotated_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("✔ Đã dừng pipeline.")
