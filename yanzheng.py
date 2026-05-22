import os
import json
import numpy as np
import pandas as pd
import cv2

# ============================================================
# 路径设置
# ============================================================

ROBOT_CSV_PATH = "robot_poses.csv"
JSON_DIR = "calibration_data"

OUTPUT_GRIPPER_CAMERA = "T_gripper_camera.txt"
OUTPUT_CAMERA_GRIPPER = "T_camera_gripper.txt"

# ============================================================
# 基础矩阵函数
# ============================================================

def make_T(R, t):
    """
    R: 3x3
    t: 3 或 3x1

    返回 4x4 齐次变换矩阵
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T

def invert_T(T):
    """
    求 4x4 齐次变换矩阵的逆
    """
    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t

    return T_inv

# ============================================================
# 机械臂 CSV 位姿读取
# ============================================================

def robot_row_to_T_base_gripper(row):
    """
    将 robot_poses.csv 中一行机械臂位姿转成 base_T_gripper.

    你的 CSV 格式：

        file_name, x, y, z, rx(deg), ry(deg), rz(deg)

    其中：
        x,y,z 单位：m
        rx,ry,rz：UR 旋转向量，单位 deg

    注意：
        UR 的 Rx,Ry,Rz 是旋转向量，不是欧拉角。
    """
    x = float(row["x"])
    y = float(row["y"])
    z = float(row["z"])

    rx_deg = float(row["rx(deg)"])
    ry_deg = float(row["ry(deg)"])
    rz_deg = float(row["rz(deg)"])

    # UR 旋转向量：deg -> rad
    rvec_rad = np.deg2rad([rx_deg, ry_deg, rz_deg]).reshape(3, 1)

    # 旋转向量 -> 旋转矩阵
    R_base_gripper, _ = cv2.Rodrigues(rvec_rad)

    t_base_gripper = np.array([x, y, z], dtype=np.float64)

    T_base_gripper = make_T(R_base_gripper, t_base_gripper)

    return T_base_gripper


# ============================================================
# JSON 棋盘格点读取与 3D-3D 配准
# ============================================================

def rigid_transform_3d(A, B):
    """
    求刚体变换 R,t，使得：

        B ≈ R @ A + t

    A: Nx3，棋盘格理论角点，target 坐标系
    B: Nx3，角点在相机坐标系下的三维坐标

    返回：
        R_target2camera, t_target2camera
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    if A.shape != B.shape:
        raise ValueError(f"A.shape != B.shape: {A.shape} vs {B.shape}")

    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # 防止出现反射矩阵
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A

    return R, t


def load_target_T_camera_from_json(json_path):
    """
    从单个 json 文件中读取棋盘格角点 camera_xyz_m，
    并和理论棋盘格角点做 3D-3D 配准，得到：

        target_T_camera

    实际矩阵含义为：

        camera_point = R_target2camera @ target_point + t_target2camera

    即：

        T_target_camera = ^camera T_target
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chessboard_size = data["chessboard_size"]
    square_size_m = float(data["square_size_m"])

    cols = int(chessboard_size[0])
    rows = int(chessboard_size[1])

    camera_xyz = np.asarray(data["corners"]["camera_xyz_m"], dtype=np.float64)

    expected_num = cols * rows
    if camera_xyz.shape[0] != expected_num:
        raise ValueError(
            f"{json_path}: 角点数量不对，"
            f"期望 {expected_num}，实际 {camera_xyz.shape[0]}"
        )

    # 构造棋盘格理论角点坐标，单位 m
    # 顺序假设和 cv2.findChessboardCorners 返回顺序一致：逐行排列
    object_points = []
    for r in range(rows):
        for c in range(cols):
            object_points.append([c * square_size_m, r * square_size_m, 0.0])

    object_points = np.asarray(object_points, dtype=np.float64)

    R_target2camera, t_target2camera = rigid_transform_3d(
        object_points,
        camera_xyz
    )

    T_target_camera = make_T(R_target2camera, t_target2camera)

    # 计算 3D 拟合误差，方便判断深度点质量
    camera_xyz_fit = (R_target2camera @ object_points.T).T + t_target2camera
    errors = np.linalg.norm(camera_xyz_fit - camera_xyz, axis=1)

    mean_error = float(np.mean(errors))
    max_error = float(np.max(errors))

    return T_target_camera, mean_error, max_error


# ============================================================
# 手眼标定结果检查
# ============================================================

def rotation_error_deg(R1, R2):
    """
    两个旋转矩阵之间的角度误差，单位 deg
    """
    R = R1 @ R2.T
    value = (np.trace(R) - 1.0) / 2.0
    value = np.clip(value, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(value)))


def evaluate_hand_eye(T_gripper_camera, T_base_gripper_list, T_target_camera_list):
    """
    简单检查手眼结果的一致性。

    对 Eye-in-Hand：

        base_T_target_i = base_T_gripper_i @ gripper_T_camera @ camera_T_target_i

    如果标定板固定不动，那么不同 i 算出来的 base_T_target 应该接近一致。
    """
    T_base_target_list = []

    for T_base_gripper, T_target_camera in zip(T_base_gripper_list, T_target_camera_list):
        T_camera_target = T_target_camera
        T_base_target = T_base_gripper @ T_gripper_camera @ T_camera_target
        T_base_target_list.append(T_base_target)

    translations = np.array([T[:3, 3] for T in T_base_target_list])

    mean_t = translations.mean(axis=0)
    trans_errors = np.linalg.norm(translations - mean_t, axis=1)

    R_ref = T_base_target_list[0][:3, :3]
    rot_errors = [
        rotation_error_deg(T[:3, :3], R_ref)
        for T in T_base_target_list
    ]

    print("\n================ 固定标定板一致性检查 ================")
    print("base_T_target 平移标准差 / m:")
    print(np.std(translations, axis=0))
    print(f"base_T_target 平移平均误差: {np.mean(trans_errors) * 1000:.3f} mm")
    print(f"base_T_target 平移最大误差: {np.max(trans_errors) * 1000:.3f} mm")
    print(f"base_T_target 姿态平均误差: {np.mean(rot_errors):.3f} deg")
    print(f"base_T_target 姿态最大误差: {np.max(rot_errors):.3f} deg")


# ============================================================
# 主程序
# ============================================================

def main():
    df = pd.read_csv(ROBOT_CSV_PATH)

    # 清理列名，防止有隐藏空格
    df.columns = [c.strip() for c in df.columns]

    required_columns = [
        "file_name",
        "x",
        "y",
        "z",
        "rx(deg)",
        "ry(deg)",
        "rz(deg)",
    ]

    for col in required_columns:
        if col not in df.columns:
            raise RuntimeError(f"robot_poses.csv 缺少列: {col}")

    R_gripper2base = []
    t_gripper2base = []

    R_target2cam = []
    t_target2cam = []

    T_base_gripper_list = []
    T_target_camera_list = []

    used_files = []

    print("Start loading calibration data...\n")

    for i, row in df.iterrows():
        file_name = str(row["file_name"]).strip()

        if not file_name:
            print(f"[Skip] row {i}: empty file_name")
            continue

        json_path = os.path.join(JSON_DIR, file_name)

        if not os.path.exists(json_path):
            print(f"[Skip] JSON not found: {json_path}")
            continue

        # 机械臂位姿：base_T_gripper
        T_base_gripper = robot_row_to_T_base_gripper(row)

        # 棋盘格位姿：target_T_camera，即 ^camera T_target
        T_target_camera, mean_err, max_err = load_target_T_camera_from_json(json_path)

        # OpenCV calibrateHandEye 需要：
        # R_gripper2base, t_gripper2base
        # 这里 base_T_gripper 本身就是 gripper -> base
        R_gripper2base.append(T_base_gripper[:3, :3])
        t_gripper2base.append(T_base_gripper[:3, 3].reshape(3, 1))

        # OpenCV calibrateHandEye 需要：
        # R_target2cam, t_target2cam
        # 这里 T_target_camera 本身就是 target -> camera
        R_target2cam.append(T_target_camera[:3, :3])
        t_target2cam.append(T_target_camera[:3, 3].reshape(3, 1))

        T_base_gripper_list.append(T_base_gripper)
        T_target_camera_list.append(T_target_camera)

        used_files.append(file_name)

        print(
            f"[Use] {file_name} | "
            f"3D fit mean error = {mean_err * 1000:.3f} mm, "
            f"max error = {max_err * 1000:.3f} mm"
        )

    sample_num = len(used_files)

    print(f"\nValid samples: {sample_num}")

    if sample_num < 5:
        raise RuntimeError(
            f"有效数据只有 {sample_num} 组，太少。"
            f"建议至少 10~15 组，最好 15~25 组。"
        )

    # ========================================================
    # 手眼标定
    # ========================================================

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base,
        t_gripper2base,
        R_target2cam,
        t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    # OpenCV 返回的是 camera -> gripper
    # 即：gripper_T_camera，也就是相机坐标系在末端坐标系下的位姿
    T_gripper_camera = make_T(R_cam2gripper, t_cam2gripper)

    # 反过来：camera_T_gripper
    T_camera_gripper = invert_T(T_gripper_camera)

    print("\n================ 手眼标定结果 ================\n")

    print("T_gripper_camera  =  末端坐标系下的相机位姿")
    print("也就是 gripper_T_camera / ^gripper T_camera:\n")
    print(T_gripper_camera)

    print("\nT_camera_gripper  =  相机坐标系下的末端位姿")
    print("也就是 camera_T_gripper / ^camera T_gripper:\n")
    print(T_camera_gripper)

    print("\n相机原点在末端坐标系下的位置，单位 m:")
    print(T_gripper_camera[:3, 3])

    np.savetxt(OUTPUT_GRIPPER_CAMERA, T_gripper_camera, fmt="%.10f")
    np.savetxt(OUTPUT_CAMERA_GRIPPER, T_camera_gripper, fmt="%.10f")

    print("\nSaved:")
    print(OUTPUT_GRIPPER_CAMERA)
    print(OUTPUT_CAMERA_GRIPPER)

    # 简单一致性检查
    evaluate_hand_eye(
        T_gripper_camera,
        T_base_gripper_list,
        T_target_camera_list
    )


if __name__ == "__main__":
    main()