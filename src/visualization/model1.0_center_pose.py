import subprocess
import numpy as np
import cv2
from collections import deque
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation as R

# ========= 参数 =========
marker_length = 0.04385  # m
WINDOW = 5               # 均值滤波窗口（帧）
POS_THRESH = 0.02        # 2 cm 偏差剔除（与上次滤波输出比较）
MAX_BAD = 5              # 连续超阈 N 次则清空队列
DRAW_AXIS_LEN = 0.05     # 画轴长度（m）
SHOW = True

# ========= 加载相机内参 =========
calib = np.load("calibration_data.npz")
K, dist = calib["K"], calib["dist"]

# ========= 加载 marker->center 变换 =========
mtc = np.load("marker_to_center.npz")
T_marker2center = {
    0: mtc["T0"].astype(np.float32),
    1: mtc["T1"].astype(np.float32),
    2: mtc["T2"].astype(np.float32),
}

# ========= Apriltag 检测器 =========
detector = Detector(
    families="tag36h11",
    nthreads=8,
    quad_decimate=2.0,
    quad_sigma=0.0,
    refine_edges=True,
    decode_sharpening=0.25,
)

# ========= 摄像头输入（ffmpeg） =========
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
frame_size = w * h * 3

# ========= 几何基元 =========
# 四个棋盘角点在 marker 坐标系下（与 r.corners 顺序匹配：TL, TR, BR, BL）
obj_points = np.array([
    [-marker_length/2,  marker_length/2, 0.0],
    [ marker_length/2,  marker_length/2, 0.0],
    [ marker_length/2, -marker_length/2, 0.0],
    [-marker_length/2, -marker_length/2, 0.0]
], dtype=np.float32)

def pose_to_T(rvec, tvec):
    Rm, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = Rm
    T[:3, 3] = tvec.reshape(3)
    return T

def average_quaternions(qs):
    """qs: (N,4) in xyzw. 半球对齐后做算术平均并归一化。"""
    qs = np.asarray(qs, dtype=np.float64)
    if qs.shape[0] == 1:
        q = qs[0]
        return (q / np.linalg.norm(q)).astype(np.float64)
    ref = qs[0]
    for i in range(1, qs.shape[0]):
        if np.dot(ref, qs[i]) < 0.0:
            qs[i] = -qs[i]
    q_mean = qs.mean(axis=0)
    q_mean /= np.linalg.norm(q_mean) + 1e-9
    return q_mean

def average_transforms(T_list):
    """同帧内平均：位移算术平均 + 姿态四元数平均"""
    if len(T_list) == 1:
        return T_list[0]
    ts = np.array([T[:3, 3] for T in T_list], dtype=np.float64)
    qs = np.array([R.from_matrix(T[:3, :3]).as_quat() for T in T_list], dtype=np.float64)  # xyzw
    t_avg = ts.mean(axis=0)
    q_avg = average_quaternions(qs)
    T_avg = np.eye(4, dtype=np.float64)
    T_avg[:3, :3] = R.from_quat(q_avg).as_matrix()
    T_avg[:3, 3] = t_avg
    return T_avg.astype(np.float32)

# ========= 跨帧滑动均值滤波缓存 =========
t_buf = deque(maxlen=WINDOW)
q_buf = deque(maxlen=WINDOW)
bad_count = 0
last_filtered_t = None  # 用于和当前帧“瞬时估计”比 2 cm

def push_filter(T):
    t = T[:3, 3].astype(np.float64)
    q = R.from_matrix(T[:3, :3]).as_quat().astype(np.float64)  # xyzw
    # 与缓冲区第一个四元数半球对齐，避免平均被反号抵消
    if len(q_buf) > 0 and np.dot(q, q_buf[0]) < 0:
        q = -q
    t_buf.append(t)
    q_buf.append(q)

def get_filtered_T():
    ts = np.stack(t_buf, axis=0)
    qs = np.stack(q_buf, axis=0)
    t_avg = ts.mean(axis=0)
    q_avg = average_quaternions(qs)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.from_quat(q_avg).as_matrix().astype(np.float32)
    T[:3, 3] = t_avg.astype(np.float32)
    return T

def draw_axes(image, T_cam_obj, K, dist, axis_len=DRAW_AXIS_LEN):
    Rm = T_cam_obj[:3, :3]
    tv = T_cam_obj[:3, 3].reshape(3, 1)
    rvec, _ = cv2.Rodrigues(Rm.astype(np.float64))
    # 坐标轴：原点和三个轴端点
    axis_pts_obj = np.array([[0,0,0],
                             [axis_len,0,0],
                             [0,axis_len,0],
                             [0,0,axis_len]], dtype=np.float32)
    imgpts, _ = cv2.projectPoints(axis_pts_obj, rvec, tv, K, dist)
    imgpts = imgpts.reshape(-1, 2).astype(int)
    O, X, Y, Z = imgpts
    cv2.line(image, O, X, (0, 0, 255), 2)  # X 红
    cv2.line(image, O, Y, (0, 255, 0), 2)  # Y 绿
    cv2.line(image, O, Z, (255, 0, 0), 2)  # Z 蓝
    cv2.circle(image, O, 3, (255, 255, 255), -1)

try:
    while True:
        raw = pipe.stdout.read(frame_size * 2)
        if not raw or len(raw) < frame_size:
            continue
        raw = raw[-frame_size:]
        try:
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3)).copy()
        except ValueError:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = detector.detect(
            gray,
            estimate_tag_pose=False,
            camera_params=[K[0,0], K[1,1], K[0,2], K[1,2]],
            tag_size=marker_length
        )

        # === 生成本帧的重心候选 ===
        T_candidates = []
        for d in dets:
            tid = int(d.tag_id)
            if tid not in (0, 1, 2):
                continue
            corners = np.array(d.corners, dtype=np.float32).reshape(-1, 2)
            ok, rvec, tvec = cv2.solvePnP(
                obj_points, corners, K, dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if not ok:
                continue
            T_cam_marker = pose_to_T(rvec, tvec)              # 相机 <- marker
            T_m2c = T_marker2center[tid]                      # marker <- center? 这里是 marker->center
            T_cam_center = (T_cam_marker @ T_m2c).astype(np.float32)  # 相机 <- center
            T_candidates.append(T_cam_center)

        updated = False
        if len(T_candidates) > 0:
            # 同帧融合（1-3 个候选）
            T_fuse = average_transforms(T_candidates)

            # 与上一次“滤波输出”比较平移差（2 cm 阈值）
            accept = True
            if last_filtered_t is not None:
                delta = np.linalg.norm(T_fuse[:3, 3] - last_filtered_t)
                accept = (delta <= POS_THRESH)

            if accept:
                push_filter(T_fuse)
                bad_count = 0
                if len(t_buf) > 0:
                    T_out = get_filtered_T()
                    last_filtered_t = T_out[:3, 3].copy()
                    updated = True
            else:
                bad_count += 1
                if bad_count >= MAX_BAD:
                    # 连续多次超阈：清空重来
                    t_buf.clear()
                    q_buf.clear()
                    last_filtered_t = None
                    bad_count = 0
                    print("[WARN] 超阈过多，清空滤波队列。")

        # 可视化：画当前“滤波输出”坐标轴
        if SHOW:
            if len(t_buf) > 0:
                T_vis = get_filtered_T()
                draw_axes(frame, T_vis, K, dist, DRAW_AXIS_LEN)
                t = T_vis[:3, 3]
                cv2.putText(frame, f"Center t: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 50), 2)
            cv2.imshow("Center Pose (avg + MA)", frame)
            if cv2.waitKey(1) == 27:  # ESC
                break

        # 打印更新节奏（可选）
        if updated:
            t = last_filtered_t
            print(f"[OK] 更新滤波输出 t = {t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f} (m) | buf={len(t_buf)}")

finally:
    if pipe and pipe.stdout:
        pipe.stdout.close()
    if pipe:
        pipe.terminate()
        pipe.wait()
    cv2.destroyAllWindows()
