import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
profile = pipeline.start(config)

color_sensor = profile.get_device().first_color_sensor()
color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
print("fx, fy, cx, cy:", color_intrinsics.fx, color_intrinsics.fy, color_intrinsics.ppx, color_intrinsics.ppy)
print("畸变系数:", color_intrinsics.coeffs)
pipeline.stop()