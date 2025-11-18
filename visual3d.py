import cv2
import open3d as o3d
import numpy as np
import torch

# Import 2 class
from RealsenYolo import RealsenseYOLO
from RGBDFusion import RGBDFusion

# --- 1. Cấu hình ---
YOLO_MODEL = 'yolov8n.pt'
FASTSAM_MODEL = 'fastsam-s.pt'
BAG_FILE = "20251112_135756.bag"
DETECT_CLASSES = ["cup", "laptop"]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Đang sử dụng thiết bị: {DEVICE}")
try:
    realsense_yolo = RealsenseYOLO(
        model_path=YOLO_MODEL,
        bag_path=BAG_FILE,
        detect_classes=DETECT_CLASSES
    )
    print("Khởi tạo RealsenseYOLO thành công.")

    rgbd_fusion = RGBDFusion(
        fastsam_ckpt=FASTSAM_MODEL,
        device=DEVICE
    )
    print("Khởi tạo RGBDFusion (FastSAM) thành công.")

except RuntimeError as e:
    print(f"Lỗi khởi tạo: {e}")
    exit()
vis = o3d.visualization.Visualizer()
vis.create_window("Fusion Vision 3D Display", width=1280, height=720)
master_pcd = o3d.geometry.PointCloud()
is_first_frame = True

opt = vis.get_render_option()
opt.background_color = np.asarray([0.15, 0.15, 0.15])
opt.point_size = 3.0
opt.show_coordinate_frame = True

print("Đang bắt đầu vòng lặp chính... Nhấn 'q' trên cửa sổ 2D để thoát.")

try:
    while True:
        try:
            color_bgr, depth, dets, yolo_results = \
                realsense_yolo.get_rgbd_and_detections()
        except RuntimeError:
            print("Đã xử lý hết file .bag.")
            break

        if color_bgr is None:
            continue

        all_frame_pcds = []
        for det in dets:
            bbox = det["bbox"]

            mask = rgbd_fusion.get_mask_from_bbox(color_bgr, bbox)

            if mask is not None:
                pc = rgbd_fusion.mask_to_pointcloud(
                    mask=mask,
                    depth=depth,
                    color_bgr=color_bgr,
                    intr=realsense_yolo.color_intr,
                    depth_scale=realsense_yolo.depth_scale
                )

                if pc is not None:
                    all_frame_pcds.append(pc)

        # --- 7. Hậu xử lý (Downsample & Denoise) ---
        if not all_frame_pcds:
            final_pcd_processed = o3d.geometry.PointCloud()
        else:
            # Gộp tất cả pcd trong frame lại
            final_pcd_raw = all_frame_pcds[0]
            for i in range(1, len(all_frame_pcds)):
                final_pcd_raw += all_frame_pcds[i]


            final_pcd_processed = rgbd_fusion.post_process_pointcloud(
                final_pcd_raw,
                voxel_size=0.005,  # 5mm
                nb_neighbors=30,
                std_ratio=2.0
            )

        # --- 8. Bước 4 (Display 3D Object): Cập nhật cửa sổ 3D ---
        master_pcd.points = final_pcd_processed.points
        master_pcd.colors = final_pcd_processed.colors

        if is_first_frame:
            vis.add_geometry(master_pcd)
            is_first_frame = False
        else:
            vis.update_geometry(master_pcd)

        if not vis.poll_events():
            break
        vis.update_renderer()

        # --- 9. Hiển thị cửa sổ 2D (Debug) ---
        annotated_2d_image = yolo_results.plot()
        cv2.imshow("2D YOLO Detection (Nhan 'q' de thoat)", annotated_2d_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Dọn dẹp
    print("Đang dừng pipeline và đóng cửa sổ.")
    vis.destroy_window()
    cv2.destroyAllWindows()