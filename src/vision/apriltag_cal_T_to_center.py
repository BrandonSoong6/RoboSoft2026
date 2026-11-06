import subprocess
import numpy as np
import cv2
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation as R

# ====== 加载相机参数 ======
calib = np.load("calibration_data.npz")
K, dist = calib["K"], calib["dist"]

# ====== Apriltag 设置 ======
detector = Detector(
    families="tag36h11",
    nthreads=16,
    quad_decimate=2.0,
    quad_sigma=0.0,
    refine_edges=True,
    decode_sharpening=0.25,
)

w, h = 1920, 1080
ffmpeg_cmd = [
    "ffmpeg",
    "-f", "v4l2",
    "-rtbufsize", "8M",
    "-input_format", "mjpeg",
    "-flags", "low_delay",
    "-framerate", "30",
    "-video_size", f"{w}x{h}",
    "-i", "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_5630B8AF-video-index0",
    "-pix_fmt", "bgr24",
    "-vcodec", "rawvideo",
    "-an", "-sn", "-f", "rawvideo", "-"
]
pipe = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, bufsize=w*h*3)

# ====== marker 定义 ======
marker_length = 0.04385  # m
frame_size = w * h * 3

# ====== 存储每个marker->重心的变换 ======
accum = {0: [], 1: [], 2: []}
max_samples = 200  # 采样多少帧后停止

def pose_to_T(Rm, tvec):
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = Rm
    T[:3, 3] = tvec.ravel()
    return T

def average_quat(quats):
    q0 = quats[0]
    for i in range(1, len(quats)):
        if np.dot(quats[i], q0) < 0:
            quats[i] = -quats[i]
    A = np.zeros((4,4), dtype=np.float64)
    for q in quats:
        A += np.outer(q, q)
    eigvals, eigvecs = np.linalg.eigh(A)
    q_mean = eigvecs[:, np.argmax(eigvals)]
    return q_mean / np.linalg.norm(q_mean)

def average_transform(T_list):
    translations = np.array([T[:3,3] for T in T_list], dtype=np.float64)
    t_avg = translations.mean(axis=0)

    quats = np.array([R.from_matrix(T[:3,:3]).as_quat() for T in T_list], dtype=np.float64)
    q_mean = average_quat(quats)
    R_avg = R.from_quat(q_mean).as_matrix()

    T_avg = np.eye(4, dtype=np.float64)
    T_avg[:3,:3] = R_avg
    T_avg[:3,3]  = t_avg
    return T_avg

while True:
    raw_frame = pipe.stdout.read(frame_size * 2)
    if not raw_frame or len(raw_frame) < frame_size:
        continue
    raw_frame = raw_frame[-frame_size:]
    try:
        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((h, w, 3)).copy()
    except ValueError:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = detector.detect(
        gray,
        estimate_tag_pose=True,   # ★ 直接估计位姿
        camera_params=[K[0,0], K[1,1], K[0,2], K[1,2]],
        tag_size=marker_length
    )

    ids = [r.tag_id for r in results]
    if all(i in ids for i in [0,1,2]):
        T_dict = {}
        for r in results:
            if r.tag_id not in [0,1,2]:
                continue
            Rm = r.pose_R  # (3,3)
            tvec = r.pose_t  # (3,1)
            T_dict[r.tag_id] = pose_to_T(Rm, tvec)

        if len(T_dict) == 3:
            T0, T1, T2 = T_dict[0], T_dict[1], T_dict[2]

            # 看相对位姿
            T01 = np.linalg.inv(T0) @ T1
            T02 = np.linalg.inv(T0) @ T2
            print("=== T01 (id1 在 id0 坐标系) ===")
            print(T01)
            print("=== T02 (id2 在 id0 坐标系) ===")
            print(T02)

            # 平面法向
            p0, p1, p2 = [T_dict[i][:3, 3] for i in [0, 1, 2]]
            n = np.cross(p1 - p0, p2 - p1)
            n /= np.linalg.norm(n)
            avg_tag_normal = sum(T_dict[i][:3, :3] @ np.array([0.,0.,1.]) for i in [0,1,2]) / 3.0
            if np.dot(n, avg_tag_normal) > 0:
                n = -n

            # 中心点（投影到平面）
            centroid = (p0 + p1 + p2) / 3.0
            center_t = centroid - np.dot(centroid - p0, n) * n

            # === 定义中心坐标系 ===
            x_tmp = p0 - centroid
            x_axis = x_tmp / np.linalg.norm(x_tmp)
            y_axis = np.cross(n, x_axis); y_axis /= np.linalg.norm(y_axis)
            z_axis = n
            center_R = np.column_stack([x_axis, y_axis, z_axis])

            T_center = np.eye(4)
            T_center[:3,:3] = center_R
            T_center[:3,3] = center_t

            # === 在局部 z 方向平移 0.1 m 并绕自身 z 轴逆时针旋转 0.3134 rad ===
            theta = -0.3134  # rad
            Rz = np.array([
                [np.cos(theta), -np.sin(theta), 0, 0],
                [np.sin(theta),  np.cos(theta), 0, 0],
                [0,              0,             1, 0],
                [0,              0,             0, 1],
            ], dtype=np.float64)

            T_offset = np.eye(4)
            T_offset[:3, 3] = [0, 0, 0.1]  # 沿局部 z 平移 0.1 m

            # 注意先旋转再平移还是先平移再旋转的问题：
            # 如果你想“先平移再旋转”，就写 T_offset = Rz @ T_offset
            # 如果想“先旋转再平移”，就写 T_offset = T_offset @ Rz
            # 这里一般期望是在中心点基础上先平移，再绕中心 z 旋转：
            T_offset = Rz @ T_offset

            # 应用到 center
            T_center = T_center @ T_offset

            # 存 marker->center
            for i in [0,1,2]:
                T_marker2center = np.linalg.inv(T_dict[i]) @ T_center
                accum[i].append(T_marker2center)

            # === 可视化: 三个marker和center的坐标轴 ===
            axis_len = 0.03  # 3 cm
            for i in [0,1,2]:
                T = T_dict[i]
                rvec, _ = cv2.Rodrigues(T[:3,:3])
                tvec = T[:3,3].reshape(3,1)
                cv2.drawFrameAxes(frame, K, dist, rvec, tvec, axis_len)

            rvec_c, _ = cv2.Rodrigues(T_center[:3,:3])
            tvec_c = T_center[:3,3].reshape(3,1)
            cv2.drawFrameAxes(frame, K, dist, rvec_c, tvec_c, axis_len*1.5)

            print(f"已采集 {len(accum[0])} 帧")
            if len(accum[0]) >= max_samples:
                break

    cv2.imshow("Calibration", frame)
    if cv2.waitKey(1) == 27:
        break

pipe.stdout.close()
pipe.terminate()
pipe.wait()
cv2.destroyAllWindows()

# ===== 求平均结果 =====
T0 = average_transform(accum[0])
T1 = average_transform(accum[1])
T2 = average_transform(accum[2])

np.savez("marker_to_center.npz", T0=T0, T1=T1, T2=T2)
print("保存完成：marker_to_center.npz")
