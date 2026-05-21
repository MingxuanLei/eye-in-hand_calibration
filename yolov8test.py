import pyrealsense2 as rs
import cv2
import numpy as np
from ultralytics import YOLO

# ------------------------------
# 0. 配置参数
# ------------------------------
# 只检测以下5个类别（COCO数据集类别ID）
KEEP_CLASSES = [0, 64, 66, 67, 41]   # person, mouse, keyboard, cell phone, cup
# 可选其他ID：例如只检测人 [0]，只检测鼠标键盘 [64,66] 等

# ------------------------------
# 1. 初始化RealSense相机
# ------------------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 启动流
profile = pipeline.start(config)

# 获取深度单位和彩色相机内参
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()  # 深度值转米

color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
intrinsics = color_profile.get_intrinsics()
fx, fy = intrinsics.fx, intrinsics.fy
cx, cy = intrinsics.ppx, intrinsics.ppy

# ------------------------------
# 2. 加载YOLO模型（自动下载预训练权重）
# ------------------------------
model = YOLO('yolov8n.pt')

# ------------------------------
# 3. 主循环
# ------------------------------
try:
    while True:
        # 获取帧
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())  # 单位：毫米

        # YOLO检测，只保留指定类别
        results = model(color_image, classes=KEEP_CLASSES, verbose=False)

        # 处理检测结果
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                # 边界框坐标
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]

                # 计算物体中心像素坐标
                center_u = (x1 + x2) // 2
                center_v = (y1 + y2) // 2

                # 获取深度值（毫米）
                depth_mm = depth_image[center_v, center_u]
                if depth_mm == 0:
                    # 如果中心点无效，尝试用边界框内有效深度的中位数
                    roi_depth = depth_image[y1:y2, x1:x2]
                    valid = roi_depth[roi_depth > 0]
                    if len(valid) > 0:
                        depth_mm = np.median(valid)
                    else:
                        depth_mm = 0

                if depth_mm > 0:
                    Z = depth_mm / 1000.0          # 米
                    X = (center_u - cx) * Z / fx    # 米
                    Y = (center_v - cy) * Z / fy    # 米
                else:
                    X = Y = Z = 0.0

                # 绘制边框和信息
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{label} {conf:.2f} | X:{X:.2f} Y:{Y:.2f} Z:{Z:.2f}m"
                cv2.putText(color_image, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 显示
        cv2.imshow("YOLO + RealSense D455 (Filtered Classes)", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()