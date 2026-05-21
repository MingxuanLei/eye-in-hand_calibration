import pyrealsense2 as rs
import numpy as np
import cv2
import json
import os
from datetime import datetime

# ==================== 配置参数 ====================
CHESSBOARD_SIZE = (11, 8)          # 内角点数 (宽, 高)  例如11x8
SQUARE_SIZE = 0.02                # 每个方格边长 单位:米 (20mm)
SAVE_DIR = "calibration_data"     # 保存数据的文件夹
KEY_SAVE = ord('s')               # 保存按键
KEY_QUIT = ord('q')               # 退出按键

# ==================== 初始化RealSense ====================
pipeline = rs.pipeline()
config = rs.config()

# 启用RGB和深度流
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 启动管道
profile = pipeline.start(config)

# 获取深度和彩色流的内参
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()   # 深度图缩放因子(米/单位)

color_profile = profile.get_stream(rs.stream.color)
color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

# 创建对齐对象：深度对齐到彩色
align_to = rs.stream.color
align = rs.align(align_to)

# 创建保存目录
os.makedirs(SAVE_DIR, exist_ok=True)

# ==================== 辅助函数 ====================
def get_3d_point(u, v, depth, intrinsics):
    """
    根据像素坐标和深度值，计算相机坐标系下的三维点
    :param u: 像素横坐标
    :param v: 像素纵坐标
    :param depth: 深度值（米）
    :param intrinsics: 相机内参
    :return: (x, y, z) 相机坐标系坐标（米）
    """
    x = (u - intrinsics.ppx) * depth / intrinsics.fx
    y = (v - intrinsics.ppy) * depth / intrinsics.fy
    z = depth
    return x, y, z

def save_frame_data(corners_pixel, corners_depth, robot_pose=None):
    """
    保存当前帧的角点数据
    :param corners_pixel: 角点像素坐标列表 [(u,v), ...]
    :param corners_depth: 每个角点的深度值列表 [depth, ...] 单位米
    :param robot_pose: 机械臂末端位姿 (可选) 字典或列表
    """
    if len(corners_pixel) == 0:
        print("没有角点数据，跳过保存")
        return

    # 生成时间戳作为文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(SAVE_DIR, f"frame_{timestamp}.json")

    # 计算每个角点的三维坐标
    points_3d = []
    for (u, v), depth in zip(corners_pixel, corners_depth):
        if depth > 0 and depth < 10:  # 有效深度范围
            x, y, z = get_3d_point(u, v, depth, color_intrinsics)
            points_3d.append([x, y, z])
        else:
            points_3d.append([None, None, None])  # 无效深度

    # 组织数据
    data = {
        "timestamp": timestamp,
        "chessboard_size": CHESSBOARD_SIZE,
        "square_size_m": SQUARE_SIZE,
        "robot_pose": robot_pose,  # 可自行填充
        "corners": {
            "pixel": [[int(u), int(v)] for u, v in corners_pixel],
            "depth_m": [float(d) if d > 0 else None for d in corners_depth],
            "camera_xyz_m": points_3d
        }
    }

    # 保存为JSON文件
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"数据已保存至: {filename}")

# ==================== 主循环 ====================
def main():
    print("开始RealSense相机...")
    print("操作说明:")
    print("  - 按 's' 键保存当前帧的角点数据")
    print("  - 按 'q' 键退出程序")
    print("  - 确保棋盘格完整出现在彩色图像中，且角点清晰可见")

    try:
        while True:
            # 等待新的一帧
            frames = pipeline.wait_for_frames()
            # 对齐深度到彩色
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # 转换为numpy数组
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # 深度图像转换为米为单位（用于显示和存储）
            depth_meters = depth_image * depth_scale

            # 检测棋盘格角点（使用彩色图）
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

            corners_pixel = []   # 存储角点像素坐标
            corners_depth = []   # 存储角点深度值(米)
            if ret:
                # 亚像素细化
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_sub = cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), criteria)

                # 绘制角点
                cv2.drawChessboardCorners(color_image, CHESSBOARD_SIZE, corners_sub, ret)

                # 提取每个角点的像素坐标和深度值
                for corner in corners_sub:
                    u, v = corner.ravel()
                    # 四舍五入取整像素坐标
                    u_int, v_int = int(round(u)), int(round(v))
                    # 确保像素在图像范围内
                    if 0 <= u_int < depth_meters.shape[1] and 0 <= v_int < depth_meters.shape[0]:
                        depth_val = depth_meters[v_int, u_int]
                    else:
                        depth_val = 0.0
                    corners_pixel.append((u, v))
                    corners_depth.append(depth_val)
            else:
                # 未检测到棋盘格，清空列表
                corners_pixel = []
                corners_depth = []

            # 准备深度图像用于显示（将深度映射到0-255彩色）
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # 在深度图像上也可以绘制角点（可选，有助于观察）
            if ret and len(corners_pixel) > 0:
                for (u, v) in corners_pixel:
                    cv2.circle(depth_colormap, (int(round(u)), int(round(v))), 3, (0, 255, 0), -1)

            # 显示图像
            cv2.imshow('RGB Image with Chessboard', color_image)
            cv2.imshow('Depth Map', depth_colormap)

            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            if key == KEY_SAVE:
                if ret:
                    # 如果需要同时保存机械臂末端位姿，请在此处获取并传入
                    # 例如：robot_pose = get_robot_pose()  需要您自己实现
                    robot_pose = None   # 可以替换为实际读取的值
                    save_frame_data(corners_pixel, corners_depth, robot_pose)
                else:
                    print("当前未检测到完整棋盘格，无法保存！")
            elif key == KEY_QUIT:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("程序已退出")

if __name__ == "__main__":
    main()