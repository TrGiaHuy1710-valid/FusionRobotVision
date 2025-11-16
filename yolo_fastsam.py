import cv2
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
from ultralytics import YOLO, FastSAM
import os

# --- 1. Tải Model ---
yolo_model = YOLO('yolov8n.pt')
fastsam_model = FastSAM('fastsam-s.pt')  # Đảm bảo tên file 'fastsam-s.pt' là chính xác
print("Đã tải xong model YOLOv8 và FastSAM.")

CLASS_ID_CUP = 39
bag_file = "20251112_135756.bag"

if not os.path.exists(bag_file):
    print(f"Lỗi: Không tìm thấy file {bag_file}")
    exit()

# --- 2. Cấu hình RealSense ---
pipeline = rs.pipeline()
config = rs.config()
rs.config.enable_device_from_file(config, bag_file)

try:
    config.enable_stream(rs.stream.depth)
    config.enable_stream(rs.stream.color)
    print("Đã yêu cầu tự động kích hoạt luồng depth và color.")
except RuntimeError as e:
    print(f"Lỗi: File .bag này có thể thiếu 1 trong 2 luồng (depth hoặc color). {e}")
    exit()

try:
    profile = pipeline.start(config)
    print("Đã khởi động pipeline thành công.")
except RuntimeError as e:
    print(f"Lỗi khi khởi động pipeline (sau khi đã enable stream): {e}")
    print("Vui lòng kiểm tra file .bag bằng realsense-viewer.")
    exit()

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"Đã phát hiện Depth Scale: {depth_scale}")

color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
color_format = color_profile.format()
print(f"Đã phát hiện Color Format: {color_format}")

align_to = rs.stream.color
align = rs.align(align_to)

intr = color_profile.get_intrinsics()
o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
    intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy
)
print(f"Đã lấy Intrinsics cho {intr.width}x{intr.height}.")  # Sẽ in ra 1280x720

# --- 3. Cấu hình Open3D Visualizer ---
vis = o3d.visualization.Visualizer()
vis.create_window("3D Visualizer - Chi 'cup'", width=800, height=600)
pcd = o3d.geometry.PointCloud()
is_first_frame = True

opt = vis.get_render_option()
opt.background_color = np.asarray([0.1, 0.1, 0.1])
opt.point_size = 2.0
opt.show_coordinate_frame = True

playback = profile.get_device().as_playback()
playback.set_real_time(False)
print("Đã tắt chế độ real-time. Sẽ xử lý từng frame.")

try:
    while True:
        try:
            frames = pipeline.wait_for_frames(5000)
        except RuntimeError:
            print("Đã xử lý hết file .bag.")
            break

        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            print("Bỏ qua frame (thiếu depth hoặc color)")
            continue

        # --- 5. Chuyển đổi định dạng ---
        color_image_raw_np = np.asanyarray(color_frame.get_data())

        if color_format == rs.format.bgr8:
            color_image_bgr_np = color_image_raw_np
            color_image_rgb_np = cv2.cvtColor(color_image_bgr_np, cv2.COLOR_BGR2RGB)
        elif color_format == rs.format.rgb8:
            color_image_rgb_np = color_image_raw_np
            color_image_bgr_np = cv2.cvtColor(color_image_rgb_np, cv2.COLOR_RGB2BGR)
        else:
            print(f"Định dạng màu {color_format} chưa được xử lý! Tạm coi là BGR.")
            color_image_bgr_np = color_image_raw_np
            color_image_rgb_np = cv2.cvtColor(color_image_bgr_np, cv2.COLOR_BGR2RGB)

        depth_image_np = np.asanyarray(depth_frame.get_data())
        o3d_color_img = o3d.geometry.Image(color_image_rgb_np)
        o3d_depth_img = o3d.geometry.Image(depth_image_np)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color_img,
            o3d_depth_img,
            depth_scale=depth_scale,
            depth_trunc=3.0,
            convert_rgb_to_intensity=False
        )

        # --- 6. Bước 1: YOLO Detection (chỉ lấy 'cup') ---
        yolo_results = yolo_model.predict(
            color_image_bgr_np,
            verbose=False,
            classes=[CLASS_ID_CUP]
        )

        # Chuẩn bị mặt nạ tổng (sẽ có shape (720, 1280))
        final_cup_mask = np.zeros((intr.height, intr.width), dtype=bool)
        bboxes_for_fastsam = yolo_results[0].boxes.xyxy.cpu().numpy()

        if len(bboxes_for_fastsam) > 0:
            # --- 7. Bước 2: FastSAM Application ---
            fastsam_results = fastsam_model.predict(
                color_image_rgb_np,
                bboxes=bboxes_for_fastsam,
                verbose=False
            )

            if fastsam_results[0].masks:
                # Lấy tensor mask thô
                all_masks = fastsam_results[0].masks.data.cpu().numpy()

                # <<< PHẦN SỬA LỖI BẮT ĐẦU >>>

                # Lấy kích thước target (W, H) cho cv2.resize
                # (intr.width, intr.height) sẽ là (1280, 720)
                target_shape_cv2 = (intr.width, intr.height)

                for mask in all_masks:
                    # 'mask' đang có shape (384, 640)

                    # Chuyển sang uint8 (0 hoặc 1) để cv2.resize có thể xử lý
                    mask_uint8 = mask.astype(np.uint8)

                    # Resize mask về đúng kích thước của ảnh màu (1280, 720)
                    # Dùng INTER_NEAREST để giữ nguyên giá trị 0 hoặc 1, không làm mờ mask
                    resized_mask_uint8 = cv2.resize(
                        mask_uint8,
                        target_shape_cv2,  # (width, height)
                        interpolation=cv2.INTER_NEAREST
                    )

                    # Chuyển lại thành boolean (True/False)
                    resized_mask_bool = resized_mask_uint8.astype(bool)

                    # Giờ 2 shape đã khớp: (720, 1280) và (720, 1280)
                    final_cup_mask = np.logical_or(final_cup_mask, resized_mask_bool)

                # <<< PHẦN SỬA LỖI KẾT THÚC >>>

        # --- 8. Bước 3: 3D Reconstruction (Lọc bằng Mask) ---
        temp_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, o3d_intrinsics
        )
        final_mask_flat = final_cup_mask.flatten()
        indices_to_keep = np.where(final_mask_flat)[0]
        cup_pcd = temp_pcd.select_by_index(indices_to_keep)

        # --- 9. Bước 4: Hậu xử lý (Downsample & Denoise) ---
        if cup_pcd.has_points():
            cup_pcd_down = cup_pcd.voxel_down_sample(voxel_size=0.005)
            cup_pcd_denoised, _ = cup_pcd_down.remove_statistical_outlier(
                nb_neighbors=30,
                std_ratio=2.0
            )
        else:
            cup_pcd_denoised = cup_pcd

        # --- 10. Cập nhật cửa sổ 3D ---
        pcd.points = cup_pcd_denoised.points
        pcd.colors = cup_pcd_denoised.colors

        if is_first_frame:
            vis.add_geometry(pcd)
            is_first_frame = False
        else:
            vis.update_geometry(pcd)

        if not vis.poll_events():
            break
        vis.update_renderer()

        # --- 11. Hiển thị cửa sổ 2D (Debug) ---
        annotated_2d_image = yolo_results[0].plot()
        cv2.imshow("2D YOLO Detection (Nhan 'q' de thoat)", annotated_2d_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Dọn dẹp
    print("Đang dừng pipeline và đóng cửa sổ.")
    pipeline.stop()
    vis.destroy_window()
    cv2.destroyAllWindows()