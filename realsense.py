import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    # 配置 RealSense 数据流
    pipeline = rs.pipeline()
    config = rs.config()

    # 启用彩色流（分辨率 640x480，30fps，BGR 格式）
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # 启用深度流（分辨率 640x480，30fps，Z16 格式）
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # 启动相机
    try:
        pipeline.start(config)
        print("相机启动成功，按 'q' 或 ESC 键退出")
    except Exception as e:
        print(f"相机启动失败: {e}")
        return

    # 创建对齐对象：将深度图像对齐到彩色图像
    align_to = rs.stream.color
    align = rs.align(align_to)

    # 获取彩色图像的尺寸（与配置一致）
    color_profile = pipeline.get_active_profile().get_stream(rs.stream.color)
    color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
    width = color_intrinsics.width
    height = color_intrinsics.height
    center_x = width // 2
    center_y = height // 2
    print(f"彩色图像尺寸: {width}x{height}, 中心点坐标: ({center_x}, {center_y})")

    try:
        while True:
            # 等待一帧数据
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            # 对齐深度帧到彩色帧
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # 转换为 numpy 数组
            color_image = np.asanyarray(color_frame.get_data())
            # 获取对齐后的深度数据（单位：毫米，可通过 get_distance 获取）
            depth_value_mm = depth_frame.get_distance(center_x, center_y) * 1000.0

            # 在彩色图像上绘制中心点（红色圆点）
            cv2.circle(color_image, (center_x, center_y), 5, (0, 0, 255), -1)
            # 显示中心点坐标和深度值
            text = f"({center_x}, {center_y}) Depth: {depth_value_mm:.1f} mm"
            cv2.putText(color_image, text, (center_x + 10, center_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 可选：同时显示伪彩色的深度图像（单独窗口）
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # 在深度图像中心也画个标记（便于观察）
            cv2.circle(depth_colormap, (center_x, center_y), 5, (0, 0, 255), -1)

            # 显示窗口
            cv2.imshow('Color Image with Depth Annotation', color_image)
            cv2.imshow('Aligned Depth Image', depth_colormap)

            # 按键退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    except KeyboardInterrupt:
        print("用户中断")
    except Exception as e:
        print(f"运行时错误: {e}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("相机已关闭")

if __name__ == "__main__":
    main()