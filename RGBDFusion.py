import cv2
import numpy as np
import open3d as o3d
from ultralytics import FastSAM
import torch


class RGBDFusion:
    def __init__(self, fastsam_ckpt="FastSAM-s.pt", device="cuda"):
        self.model = FastSAM(fastsam_ckpt)
        self.device = device
        # Bạn có thể thay đổi imgsz nếu cần
        self.imgsz = 640

    def get_mask_from_bbox(self, img_bgr, bbox):
        """
        bbox = [x1, y1, x2, y2]
        FastSAM dùng RGB.

        *** CẢI TIẾN ***: 
        1. Gộp tất cả các mask con do FastSAM trả về.
        2. Tự động resize mask về kích thước ảnh gốc.
        """
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Lấy kích thước ảnh gốc
        orig_h, orig_w = img_rgb.shape[:2]

        results = self.model(
            img_rgb,
            bboxes=[bbox],
            device=self.device,
            retina_masks=True,
            imgsz=self.imgsz,
            verbose=False
        )

        if results[0].masks is None:
            return None

        masks_raw = results[0].masks.data.cpu().numpy()  # (N, H_model, W_model)

        if len(masks_raw) == 0:
            return None

        # --- CẢI THIỆN BINARY MASK ---
        # 1. Tạo mask rỗng với kích thước xử lý của model
        model_h, model_w = masks_raw.shape[1], masks_raw.shape[2]
        combined_mask_model_size = np.zeros((model_h, model_w), dtype=bool)

        # 2. Gộp tất cả các mask con lại (thường N=1, nhưng đây là cách an toàn)
        for mask in masks_raw:
            combined_mask_model_size = np.logical_or(combined_mask_model_size, mask.astype(bool))

        # 3. Resize mask về kích thước ảnh gốc
        # Chuyển sang uint8 (0, 1)
        combined_mask_uint8 = combined_mask_model_size.astype(np.uint8)

        # Resize bằng INTER_NEAREST (quan trọng cho mask)
        final_mask_uint8 = cv2.resize(
            combined_mask_uint8,
            (orig_w, orig_h),  # (width, height)
            interpolation=cv2.INTER_NEAREST
        )

        # 4. Chuyển lại thành boolean và trả về
        return final_mask_uint8.astype(bool)

    def mask_to_pointcloud(self, mask, depth, color_bgr, intr, depth_scale):
        """
        Tạo point cloud thô từ mask, depth và color.
        """
        v_idx, u_idx = np.where(mask)
        if len(u_idx) == 0:
            return None

        # Lấy giá trị depth và chuyển sang mét
        z = depth[v_idx, u_idx].astype(np.float32) * depth_scale

        # Chỉ giữ lại các điểm có độ sâu hợp lệ ( > 0)
        valid = z > 0
        if not np.any(valid):
            return None

        z = z[valid]
        u = u_idx[valid].astype(np.float32)
        v = v_idx[valid].astype(np.float32)

        # Chiếu 3D (Un-project)
        x = (u - intr.ppx) / intr.fx * z
        y = (v - intr.ppy) / intr.fy * z

        pts = np.vstack((x, y, z)).T

        # Lấy màu BGR và chuẩn hóa (0-1)
        colors_bgr = color_bgr[v_idx[valid], u_idx[valid]].astype(np.float32) / 255.0

        # Chuyển BGR (OpenCV) sang RGB (Open3D)
        colors_rgb = colors_bgr[:, ::-1]

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts)
        pc.colors = o3d.utility.Vector3dVector(colors_rgb)

        return pc

    # --- HÀM MỚI THEO YÊU CẦU ---
    def post_process_pointcloud(self, pc, voxel_size=0.005, nb_neighbors=30, std_ratio=2.0):
        """
        Áp dụng down sampling và denoising cho point cloud
        theo pipeline "Fusion Vision".
        """
        if not pc.has_points():
            return pc

        # 1. Downsampling (lọc thô) 
        # Giảm mật độ điểm, làm cho PC nhẹ hơn
        pc_down = pc.voxel_down_sample(voxel_size=voxel_size)

        # 2. Denoising (lọc nhiễu) 
        # Loại bỏ các điểm bay lơ lửng (outliers)
        pc_denoised, _ = pc_down.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        return pc_denoised


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fusion = RGBDFusion(fastsam_ckpt="FastSAM-s.pt", device=device)

    print(f"[INFO] RGBDFusion initialized on {device}")

    # ============================
    # DEMO: Load test image + depth
    # ============================
    color = cv2.imread("test.jpg")    # ảnh RGB
    depth = cv2.imread("test_depth.png", cv2.IMREAD_UNCHANGED)  # file depth
    if color is None or depth is None:
        print("[ERR] Thiếu test.jpg hoặc test_depth.png!")
        quit()

    # Depth scale (Realsense thường là 0.001)
    depth_scale = 0.001

    # Fake intrinsic (thay bằng intr của bạn)
    class Intr:
        fx = 600
        fy = 600
        ppx = color.shape[1] / 2
        ppy = color.shape[0] / 2

    intr = Intr()

    # ============================
    # BBOX đối tượng muốn segment
    # ============================
    bbox = [200, 150, 420, 400]  # [x1,y1,x2,y2]

    # ============================
    # 1. Lấy mask từ FastSAM
    # ============================
    mask = fusion.get_mask_from_bbox(color, bbox)

    if mask is None:
        print("[ERR] Không tìm thấy mask!")
        quit()

    # Hiển thị mask gốc
    cv2.imshow("Mask (raw)", mask.astype(np.uint8) * 255)

    # ============================
    # 2. Tạo point cloud thô
    # ============================
    pc_raw = fusion.mask_to_pointcloud(mask, depth, color, intr, depth_scale)
    if pc_raw is None:
        print("[ERR] PC raw empty!")
        quit()

    print(f"[INFO] Raw point cloud size: {len(pc_raw.points)} points")

    # ============================
    # 3. Post-process (Downsampling + Denoise)
    # ============================
    pc_clean = fusion.post_process_pointcloud(
        pc_raw,
        voxel_size=0.005,
        nb_neighbors=20,
        std_ratio=2.0
    )

    print(f"[INFO] Cleaned point cloud size: {len(pc_clean.points)} points")

    # ============================
    # 4. Tạo mask đã lọc nhiễu (optional)
    # ============================
    # render mask đã lọc bằng cách mark lại những pixel xuất hiện trong pc_clean
    mask_clean = np.zeros_like(mask, dtype=np.uint8)

    pts = np.asarray(pc_clean.points)
    colors = np.asarray(pc_clean.colors)

    # Convert (X,Y,Z) → pixel coordinate (u,v)
    u = (pts[:, 0] / pts[:, 2]) * intr.fx + intr.ppx
    v = (pts[:, 1] / pts[:, 2]) * intr.fy + intr.ppy

    u = u.astype(int)
    v = v.astype(int)

    # Ghi lại vào mask
    for ui, vi in zip(u, v):
        if 0 <= ui < mask_clean.shape[1] and 0 <= vi < mask_clean.shape[0]:
            mask_clean[vi, ui] = 255

    cv2.imshow("Mask After Denoise", mask_clean)

    # ============================
    # 5. Show point cloud
    # ============================
    o3d.visualization.draw_geometries([pc_clean])

    cv2.waitKey(0)
    cv2.destroyAllWindows()