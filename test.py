import pyrealsense2 as rs
import numpy as np
import cv2
import os

BAG_FILE = '20251112_135756.bag'


if not os.path.exists(BAG_FILE):
    print(f"Lỗi: Không tìm thấy file {BAG_FILE}")
    exit()

pipe = rs.pipeline()
config = rs.config()

try:
    config.enable_device_from_file(BAG_FILE)
    print(f"Đang đọc từ file: {BAG_FILE}")
except Exception as e:
    print(f"Lỗi khi tải file .bag: {e}")
    exit()
try:
    profile = pipe.start(config)
except RuntimeError as e:
    print(f"Không thể bắt đầu pipeline. File .bag có thể bị hỏng hoặc chứa định dạng không hỗ trợ.")
    print(f"Lỗi chi tiết: {e}")
    exit()

playback = profile.get_device().as_playback()
playback.set_real_time(False)
colorizer = rs.colorizer()

# Chúng ta muốn căn chỉnh (align) mọi thứ VÀO luồng màu (color stream)
align_to = rs.stream.color
align = rs.align(align_to)
# ----------------------------------------

try:
    while True:
        frames = pipe.wait_for_frames()
        # align.process() sẽ trả về một bộ frame đã được căn chỉnh
        aligned_frames = align.process(frames)

        # Lấy frame từ BỘ ĐÃ CĂN CHỈNH
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()  # Đây là frame độ sâu đã được căn chỉnh

        if not depth_frame or not color_frame:
            continue
        # Chuyển đổi frame sang dạng mảng numpy
        color_image = np.asanyarray(color_frame.get_data())
        colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

        images = np.hstack((color_image, colorized_depth))

        # Hiển thị
        cv2.namedWindow('RealSense RBGD', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense RBGD', images)

        key = cv2.waitKey(1)
        # Nhấn 'q' hoặc ESC để thoát
        if key == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

except RuntimeError:
    print("Đã phát hết file .bag.")
finally:
    pipe.stop()
    cv2.destroyAllWindows()
    print("Đã dừng pipeline.")