import pyrealsense2 as rs
import cv2
import numpy as np
from ultralytics import YOLO

# ------------------------------
# 1. 初始化RealSense相机
# ------------------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 启动流
profile = pipeline.start(config)

# 获取深度和彩色相机的内参
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()  # 深度单位->米

# 获取彩色相机内参 (用于像素坐标转相机坐标)
color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
intrinsics = color_profile.get_intrinsics()  # fx, fy, cx, cy
fx, fy = intrinsics.fx, intrinsics.fy
cx, cy = intrinsics.ppx, intrinsics.ppy

# ------------------------------
# 2. 加载YOLOv8模型 (自动下载预训练权重)
# ------------------------------
model = YOLO('yolov8n.pt')  # 使用nano模型，轻量快速

# ------------------------------
# 3. 主循环
# ------------------------------
try:
    while True:
        # 等待新的一帧
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        # 转为numpy数组
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())  # 单位：毫米

        # YOLO检测
        results = model(color_image, verbose=False)  # verbose=False 减少输出

        # 解析检测结果
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                # 获取边界框坐标 (xyxy格式)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]

                # 计算物体中心坐标 (像素)
                center_u = (x1 + x2) // 2
                center_v = (y1 + y2) // 2

                # 获取中心点的深度值 (毫米)
                depth_mm = depth_image[center_v, center_u]
                if depth_mm == 0:
                    # 如果中心点深度无效，尝试取边界框内有效深度的中位数
                    roi_depth = depth_image[y1:y2, x1:x2]
                    valid_depths = roi_depth[roi_depth > 0]
                    if len(valid_depths) > 0:
                        depth_mm = np.median(valid_depths)
                    else:
                        depth_mm = 0

                if depth_mm > 0:
                    # 转换为米
                    Z = depth_mm / 1000.0
                    # 像素坐标转相机坐标 (单位：米)
                    X = (center_u - cx) * Z / fx
                    Y = (center_v - cy) * Z / fy
                else:
                    X = Y = Z = 0.0

                # 绘制边界框和标签
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{label} {conf:.2f} | X:{X:.2f} Y:{Y:.2f} Z:{Z:.2f}m"
                cv2.putText(color_image, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 显示结果
        cv2.imshow("YOLO + RealSense D455", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 清理资源
    pipeline.stop()
    cv2.destroyAllWindows()