import subprocess
import numpy as np
import cv2
import cv2.aruco as aruco

# ====== 加载相机参数 ======
calib = np.load("calibration_data.npz")
K, dist = calib["K"], calib["dist"]

# ====== ArUco 设置 ======
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

device_name = "HD Pro Webcam C920"
w, h = 1920, 1080

# ffmpeg 输出 BGR24 原始流
ffmpeg_cmd = [
    "ffmpeg",
    "-f", "dshow",
    "-rtbufsize", "16M",
    "-flags", "low_delay",
    "-framerate", "30",
    "-video_size", f"{w}x{h}",
    "-i", f"video={device_name}",
    "-pix_fmt", "bgr24",
    "-vcodec", "rawvideo",
    "-an", "-sn", "-f", "rawvideo", "-"
]

pipe = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, bufsize=w*h*3)

marker_length = 0.0721  # m，marker 边长

# 定义 marker 在自身坐标系下的 3D 角点
obj_points = np.array([
    [-marker_length / 2,  marker_length / 2, 0],
    [ marker_length / 2,  marker_length / 2, 0],
    [ marker_length / 2, -marker_length / 2, 0],
    [-marker_length / 2, -marker_length / 2, 0]
], dtype=np.float32)

frame_size = w * h * 3

while True:
    # 一次性多读一些，把旧帧冲掉
    raw_frame = pipe.stdout.read(frame_size * 5)
    if not raw_frame:
        continue

    # 如果读到多帧，只取最后一帧
    if len(raw_frame) >= frame_size:
        raw_frame = raw_frame[-frame_size:]

        try:
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((h, w, 3)).copy()
        except ValueError:
            continue

        # ====== 检测 marker ======
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            # 绘制检测到的 marker
            aruco.drawDetectedMarkers(frame, corners, ids)

            # 遍历每个 marker，估计位姿
            for i, marker_id in enumerate(ids):
                img_points = corners[i].reshape(-1, 2)

                success, rvec, tvec = cv2.solvePnP(
                    obj_points, img_points, K, dist,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                if not success:
                    continue

                # 位姿矩阵
                R, _ = cv2.Rodrigues(rvec)
                T = np.eye(4, dtype=np.float32)
                T[:3, :3] = R
                T[:3, 3] = tvec.ravel()

                print(f"\nID {int(marker_id)} 的位姿矩阵:")
                print(T)

                # 在图像上绘制坐标轴
                cv2.drawFrameAxes(frame, K, dist, rvec, tvec, 0.03)

                # 在 marker 上方标注 ID
                cX, cY = img_points.mean(axis=0).astype(int)
                cv2.putText(frame, f"ID {int(marker_id)}",
                            (cX, cY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)

        cv2.imshow("ArUco Multi Pose", frame)

    if cv2.waitKey(1) == 27:  # ESC退出
        break

pipe.stdout.close()
pipe.terminate()
pipe.wait()
cv2.destroyAllWindows()
