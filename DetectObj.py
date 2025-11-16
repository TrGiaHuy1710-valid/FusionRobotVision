# import pyrealsense2 as rs
# import numpy as np
# import cv2
# from ultralytics import YOLO
# import os
#
# class RealsenseYOLO:
#     def __init__(self, model_path, bag_path, detect_classes):
#         self.model = YOLO(model_path)
#         print("✔ YOLO model loaded.")
#
#         self.bag_path = bag_path
#         self.detect_classes = detect_classes
#
#         # Map class_name → class_id
#         self.class_ids = self._resolve_class_ids(detect_classes)
#
#         # Setup RealSense
#         self.pipeline = rs.pipeline()
#         self.config = rs.config()
#
#         if not os.path.exists(bag_path):
#             raise FileNotFoundError(f"Không tìm thấy file .bag: {bag_path}")
#
#         rs.config.enable_device_from_file(self.config, bag_path)
#         self.config.enable_stream(rs.stream.color)
#
#         self.profile = self.pipeline.start(self.config)
#         print("✔ RealSense pipeline started (RGB only).")
#
#     def _resolve_class_ids(self, names):
#         """Chuyển tên class sang class_id theo YOLO."""
#         ids = []
#         for target in names:
#             found = False
#             for cls_id, name in self.model.names.items():
#                 if name == target:
#                     ids.append(cls_id)
#                     found = True
#                     break
#             if not found:
#                 print(f"⚠ Warning: '{target}' không tồn tại trong YOLO model!")
#         print("✔ Detecting classes:", names)
#         return ids
#
#     def get_frame(self):
#         """Lấy 1 frame từ file .bag (RGB)."""
#         frames = self.pipeline.wait_for_frames()
#         color_frame = frames.get_color_frame()
#         if not color_frame:
#             return None
#         color_image = np.asanyarray(color_frame.get_data())
#         color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
#         return color_image
#
#     def detect(self, frame):
#         """Detect object, trả về list thông tin bbox."""
#         results = self.model.predict(frame, classes=self.class_ids, verbose=False)
#         objs = []
#
#         for box in results[0].boxes:
#             cls_id = int(box.cls[0])
#             conf = float(box.conf[0])
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#
#             objs.append({
#                 "name": self.model.names[cls_id],
#                 "confidence": conf,
#                 "bbox": [x1, y1, x2, y2]
#             })
#
#         return objs, results[0].plot()
#
#     def run(self, show=True):
#         """Chạy loop — hiển thị và trả về bbox mỗi frame."""
#         print("✔ Running detection... (press 'q' to quit)")
#
#         while True:
#             frame = self.get_frame()
#             if frame is None:
#                 print("❌ Không lấy được frame.")
#                 break
#
#             detections, annotated = self.detect(frame)
#
#             # **Trả OUTPUT ngay tại đây**
#             print("Detections:", detections)
#
#             if show:
#                 cv2.imshow("Realsense + YOLO", annotated)
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break
#
#         self.pipeline.stop()
#         cv2.destroyAllWindows()
#         print("✔ Stopped.")
#
#
# if __name__ == '__main__':
#     detector = RealsenseYOLO(
#         model_path="yolov8n.pt",
#         bag_path="20251112_135756.bag",
#         detect_classes=["cup", "bottle"]
#     )
#
#     detector.run(show=True)