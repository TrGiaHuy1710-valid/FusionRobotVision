import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import os  # Thêm thư viện os để kiểm tra file

# --- 1. Tải model YOLOv8 ---
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

# --- THAY ĐỔI LỚN Ở ĐÂY ---
# Chúng ta không ép 640x480 hay BGR8 nữa.
# Chúng ta chỉ yêu cầu "luồng màu" (color stream)
# pyrealsense2 sẽ tự động tìm luồng màu có trong file .bag
try:
    config.enable_stream(rs.stream.color)
    print("Đã yêu cầu tự động kích hoạt luồng màu (color stream).")
except Exception as e:
    print(f"Lỗi: Không thể kích hoạt luồng màu. Có thể file .bag không có luồng màu? Lỗi: {e}")
    exit()

# --- 3. Bắt đầu pipeline ---
try:
    profile = pipeline.start(config)
    print("Đã bắt đầu pipeline từ file .bag.")
except Exception as e:
    # Nếu vẫn lỗi ở đây, file .bag của bạn gần như chắc chắn có vấn đề
    print(f"Không thể bắt đầu pipeline. Lỗi: {e}")
    print("Vui lòng kiểm tra file .bag bằng realsense-viewer.")
    exit()

playback = profile.get_device().as_playback()
playback.set_real_time(False)
print("Đã tắt chế độ real-time. Sẽ xử lý từng frame.")

window_name = "YOLOv8 Detection from .bag (Nhan 'q' de thoat)"
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

try:
    while True:
        try:
            frames = pipeline.wait_for_frames(5000)
        except RuntimeError:
            # Khi hết frame, wait_for_frames sẽ báo lỗi RuntimeError
            print("Đã xử lý hết frame trong file .bag.")
            break

        if not frames:
            continue

        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # --- 4. Chuyển đổi sang định dạng OpenCV (BGR) ---

        # Lấy thông tin định dạng của luồng màu
        color_profile = color_frame.get_profile().as_video_stream_profile()
        color_format = color_profile.format()

        # Chuyển frame sang mảng numpy
        color_image = np.asanyarray(color_frame.get_data())

        # KIỂM TRA VÀ CHUYỂN ĐỔI MÀU NẾU CẦN
        # YOLOv8 và OpenCV mặc định làm việc tốt nhất với BGR
        if color_format == rs.format.rgb8:
            # Nếu file .bag lưu là RGB, chuyển sang BGR
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        elif color_format == rs.format.yuyv:
            # Nếu file .bag lưu là YUYV, chuyển sang BGR
            color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGR_YUYV)
        elif color_format != rs.format.bgr8:
            # Nếu là định dạng lạ, báo cảnh báo
            print(f"Cảnh báo: Định dạng màu là {color_format}, không phải BGR8/RGB8. Đang cố gắng xử lý...")
            # Bạn có thể cần thêm logic chuyển đổi ở đây nếu gặp lỗi

        # Nếu định dạng là BGR8, không cần làm gì cả.

        # --- 5. Chạy YOLOv8 Detection ---
        # Bây giờ color_image luôn ở định dạng BGR (hoặc đã được cố gắng chuyển đổi)
        results = model.predict(color_image, verbose=False)

        # --- 6. Vẽ kết quả và hiển thị ---
        annotated_image = results[0].plot()
        cv2.imshow(window_name, annotated_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Đã nhấn 'q'. Đang thoát...")
            break

finally:
    print("Đang dừng pipeline và đóng cửa sổ.")
    pipeline.stop()
    cv2.destroyAllWindows()