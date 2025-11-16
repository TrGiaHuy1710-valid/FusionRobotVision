import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import os
# --- 1. Tải model YOLOv8 ---
# Tải model một lần duy nhất
model = YOLO('yolov8n.pt')
print("Đã tải xong model YOLOv8.")

# --- 2. Cấu hình pipeline để đọc từ file .bag ---
pipeline = rs.pipeline()
config = rs.config()

# !!! QUAN TRỌNG: Thay đổi đường dẫn này
bag_file = "20251112_135756.bag"

if not os.path.exists(bag_file):
    print(f"Lỗi: Không tìm thấy file {bag_file}")
    print("Vui lòng kiểm tra lại đường dẫn file .bag!")
    exit()

rs.config.enable_device_from_file(config, bag_file)


# --- 3. Bắt đầu pipeline VÀ tạo đối tượng Align ---
profile = pipeline.start(config)

# Lấy cảm biến độ sâu để lấy thang đo (scale)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"Depth Scale (tỉ lệ) là: {depth_scale}")  # Thường là 0.001

# QUAN TRỌNG: Tạo đối tượng Align
# Chúng ta align (căn chỉnh) ảnh depth theo ảnh color
align_to = rs.stream.color
align = rs.align(align_to)

print("Đã bắt đầu pipeline và Align. Nhấn 'q' để thoát...")

try:
    while True:
        # Chờ một cặp frame đã được căn chỉnh
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)  # Căn chỉnh tại đây

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # --- 4. Chuyển đổi sang mảng NumPy ---

        # Ảnh màu (BGR, sẵn sàng cho YOLOv8 và OpenCV)
        color_image = np.asanyarray(color_frame.get_data())

        # Ảnh depth (16-bit, đã được căn chỉnh với ảnh màu)
        depth_image = np.asanyarray(depth_frame.get_data())

        # --- 5. Chạy YOLOv8 ---
        # Chạy detection trên ảnh MÀU
        results = model.predict(color_image, verbose=False)

        # Lấy ảnh đã được vẽ box (cho tiện)
        annotated_image = results[0].plot()

        # --- 6. Lấy khoảng cách cho từng vật thể ---
        for box in results[0].boxes:
            # Lấy tọa độ bounding box
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]

            # Lấy class name
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # Tính điểm trung tâm của box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Lấy giá trị độ sâu (16-bit) tại điểm trung tâm từ ảnh DEPTH
            # depth_image[cy, cx] trả về giá trị số nguyên (ví dụ: 2500)
            depth_value_in_mm = depth_image[cy, cx]

            # Chuyển đổi sang mét
            distance_in_meters = depth_value_in_mm * depth_scale

            # Tạo text để hiển thị
            text = f"{label}: {distance_in_meters:.2f} m"

            # Vẽ text lên ảnh đã được annotate
            cv2.putText(annotated_image, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Vẽ 1 vòng tròn ở điểm trung tâm (để debug)
            cv2.circle(annotated_image, (cx, cy), 5, (0, 0, 255), -1)

        # --- 7. Hiển thị ---
        cv2.imshow('YOLOv8 voi Khoang cach (RGBD)', annotated_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Đã dừng pipeline.")