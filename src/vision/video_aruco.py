import cv2
import cv2.aruco as aruco
import numpy as np

# ====== 加载相机参数 ======
calib = np.load("calibration_data.npz")
K, dist = calib["K"], calib["dist"]

# ====== ArUco 设置 ======
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# ====== 打开摄像头（ID=1） ======
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise RuntimeError("无法打开摄像头 ID=1")

marker_length = 0.0721  # m，marker 边长

# 定义 marker 在自身坐标系下的 3D 角点
obj_points = np.array([
    [-marker_length / 2,  marker_length / 2, 0],
    [ marker_length / 2,  marker_length / 2, 0],
    [ marker_length / 2, -marker_length / 2, 0],
    [-marker_length / 2, -marker_length / 2, 0]
], dtype=np.float32)

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取视频帧")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ====== 检测 marker ======
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None and len(ids) > 0:
        aruco.drawDetectedMarkers(frame, corners, ids)

        for i, id in enumerate(ids):
            img_points = corners[i].reshape(-1, 2)

            success, rvec, tvec = cv2.solvePnP(
                obj_points, img_points, K, dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if not success:
                continue

            # ====== 打印位姿矩阵 ======
            R, _ = cv2.Rodrigues(rvec)
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = R
            T[:3, 3] = tvec.ravel()
            print(f"ID {id}: 位姿矩阵 (相机坐标系下)")
            print(T)

            # ====== 在画面中绘制坐标轴 ======
            cv2.drawFrameAxes(frame, K, dist, rvec, tvec, 0.02)  # 坐标轴长 2 cm

    cv2.imshow("ArUco Pose", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == 27:  # 按 ESC 退出
        break

cap.release()
cv2.destroyAllWindows()
