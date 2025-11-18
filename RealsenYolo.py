import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO


class RealsenseYOLO:
    def __init__(self, model_path, bag_path, detect_classes):
        self.model = YOLO(model_path)

        # Resolve class IDs
        self.class_ids = []
        for t in detect_classes:
            for cid, name in self.model.names.items():
                if name == t:
                    self.class_ids.append(cid)

        # Init rs
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        rs.config.enable_device_from_file(self.config, bag_path)

        # Enable both RGB and Depth
        rs.config.enable_device_from_file(self.config, bag_path, repeat_playback=False)

        self.profile = self.pipeline.start(self.config)

        # Align depth → color
        self.align = rs.align(rs.stream.color)

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.color_intr = color_stream.get_intrinsics()

    def get_rgbd_and_detections(self):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None, None, None

        color = np.asanyarray(color_frame.get_data())
        color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        depth = np.asanyarray(depth_frame.get_data())

        results = self.model.predict(color_bgr, classes=self.class_ids, verbose=False)

        dets = []
        for box in results[0].boxes:
            cid = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            dets.append({
                "name": self.model.names[cid],
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })

        return color_bgr, depth, dets, results[0]

if __name__ == '__main__':
    model_path = "yolov8n.pt"
    bag_path = "20251112_135756.bag"
    detect_classes = ['cup', 'laptop']
    realsense = RealsenseYOLO(model_path=model_path, bag_path=bag_path, detect_classes=detect_classes)