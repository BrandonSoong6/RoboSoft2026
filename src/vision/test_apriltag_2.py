import subprocess
import numpy as np
import cv2
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation as R
import time
from collections import deque

# ====== 历史合法解缓存 ======
history = deque(maxlen=5)  # 存储 (frame_idx, 平移, 四元数)

# ====== 时间滤波队列 ======
window = deque(maxlen=1)   # 最近合法的中心位姿
pos_thresh = 0.02           # 平移阈值 (m)
rot_thresh = np.deg2rad(10) # 旋转阈值 (rad)
frame_idx = 0

# ====== 零位缓存（★ 第 N 个入队的真实解） ======
zero_target_enqueue = 50    # ★ 想改第几个入队就改这里
enqueue_count = 0           # ★ 入队计数（只在 valid→append 时自增）
zero_saved = False
T0 = None
zero_file = "center_zero_pose.npz"

# ====== 加载相机参数 ======
calib = np.load("calibration_data.npz")
K, dist = calib["K"], calib["dist"]

# ====== 加载 marker->center 变换 ======
marker2center = np.load("marker_to_center.npz")
T_marker2center = {
    0: marker2center["T0"],
    1: marker2center["T1"],
    2: marker2center["T2"],
}
print("已加载 marker_to_center.npz")

# ====== Apriltag 设置 ======
detector = Detector(
    families="tag36h11",
    nthreads=16,
    quad_decimate=2.0,
    quad_sigma=0.0,
    refine_edges=True,
    decode_sharpening=0.25,
)

# ====== 打开摄像头 (1080p, 30fps) ======
subprocess.run([
    "v4l2-ctl",
    "--device=/dev/video2",
    "--set-fmt-video=width=1920,height=1080,pixelformat=MJPG"
], check=True)

cap = cv2.VideoCapture(2, cv2.CAP_V4L2)

# 设置 MJPG + 分辨率 + 帧率
w, h = 1920, 1080
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
cap.set(cv2.CAP_PROP_FPS, 30)
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
print("当前 FOURCC:", "".join([chr((fourcc >> 8*i) & 0xFF) for i in range(4)]))
print("实际分辨率:", cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("实际 FPS:", cap.get(cv2.CAP_PROP_FPS))

if not cap.isOpened():
    raise RuntimeError("无法打开摄像头")

marker_length = 0.04385  # m
frame_size = w * h * 3

# ====== 工具函数 ======
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

def compute_reproj_error(Rm, tvec, corners, K, dist, marker_length):
    half = marker_length / 2.0
    objp = np.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0]
    ], dtype=np.float32)
    rvec, _ = cv2.Rodrigues(Rm)
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    proj = proj.reshape(-1,2)
    err = np.linalg.norm(proj - corners.reshape(-1,2), axis=1).mean()
    return err

def push_history(frame_idx, T):
    t = T[:3,3]
    q = R.from_matrix(T[:3,:3]).as_quat()
    history.append((frame_idx, t, q))

def extrapolate_center(frame_idx):
    if len(history) < 2:
        return None
    frames = np.array([h[0] for h in history])
    positions = np.array([h[1] for h in history])
    t_est = []
    for i in range(3):
        coeffs = np.polyfit(frames, positions[:,i], 2)
        t_est.append(np.polyval(coeffs, frame_idx))
    t_est = np.array(t_est)
    q_est = history[-1][2]
    R_est = R.from_quat(q_est).as_matrix()
    T_est = np.eye(4, dtype=np.float64)
    T_est[:3,:3] = R_est
    T_est[:3,3] = t_est
    return T_est

def inv_T(T):
    Rinvt = T[:3,:3].T
    tinv = -Rinvt @ T[:3,3]
    Tout = np.eye(4, dtype=T.dtype)
    Tout[:3,:3] = Rinvt
    Tout[:3,3]  = tinv
    return Tout

def rotvec_deg_from_T(T):
    rv = R.from_matrix(T[:3,:3]).as_rotvec()
    return np.rad2deg(rv)

def push_candidate(T_candidates, window):
    """返回当前平均位姿（可能是外推），并在 valid→append 时统计入队与保存零位。"""
    global frame_idx, enqueue_count, zero_saved, T0

    frame_idx += 1

    # 队列均值（若有）
    if window:
        T_avg = average_transform(list(window))
        t_avg = T_avg[:3,3]
        q_avg = R.from_matrix(T_avg[:3,:3]).as_quat()
    else:
        T_avg, t_avg, q_avg = None, None, None

    # 合法性检查
    valid = []
    for T in T_candidates:
        if T_avg is None:
            valid.append(T)
            continue
        dp = np.linalg.norm(T[:3,3] - t_avg)
        dR = R.from_quat(q_avg).inv() * R.from_matrix(T[:3,:3])
        d_angle = dR.magnitude()
        if dp < pos_thresh and d_angle < rot_thresh:
            valid.append(T)

    # 更新队列（只有这里才算“入队”的真实解）
    if valid:
        T_new = average_transform(valid)
        window.append(T_new)

        # ★ 零位保存：入队计数自增与触发保存
        enqueue_count += 1
        if (not zero_saved) and (enqueue_count == zero_target_enqueue):
            T0 = T_new.copy()

            # ===== 沿着自身 -Z 方向平移 125 mm =====
            offset_local = np.array([0.0, 0.0, -125.0])  # 单位 mm
            # 如果你的 T_new 是米为单位，就写 -0.125
            offset_global = T0[:3, :3] @ offset_local
            T0[:3, 3] += offset_global

            np.savez(zero_file,
                     T0=T0,
                     enqueue_index=enqueue_count,
                     frame_idx_when_saved=frame_idx,
                     saved_time=time.time())
            zero_saved = True
            print(f"✅ 已保存零位并下移125mm：第 {enqueue_count} 个入队的真实解 → {zero_file}")

        push_history(frame_idx, T_new)

    else:
        # 没有合法解时，每两帧才 pop 一次
        if frame_idx % 2 == 0 and window:
            window.popleft()

    # 如果窗口为空，尝试用历史外推（外推不计入“入队”）
    if not window:
        T_est = extrapolate_center(frame_idx)
        if T_est is not None:
            print(f"[Frame {frame_idx}] window empty → 使用外推")
            return T_est

    print(f"[Frame {frame_idx}] window_len={len(window)}, valid={len(valid)}, enqueued={enqueue_count}")

    # 返回当前平均
    return average_transform(list(window)) if window else None

frame_count = 0
last_time = time.time()

# ====== 主循环 ======
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=[K[0,0], K[1,1], K[0,2], K[1,2]],
        tag_size=marker_length
    )

    T_centers = []
    for r in results:
        if r.tag_id not in [0,1,2]:
            continue
        if r.hamming != 0:
            continue
        err = compute_reproj_error(r.pose_R, r.pose_t, r.corners, K, dist, marker_length)
        if err > 2.0:
            continue

        T_cam_marker = pose_to_T(r.pose_R, r.pose_t)
        T_cam_center = T_cam_marker @ T_marker2center[r.tag_id]
        T_centers.append(T_cam_center)

        # 可视化 marker 坐标系
        rvec, _ = cv2.Rodrigues(r.pose_R)
        tvec = r.pose_t.reshape(3,1)
        cv2.drawFrameAxes(frame, K, dist, rvec, tvec, 0.03)

    # 经过窗口滤波后的 center 位姿（相机坐标）
    T_center_avg = push_candidate(T_centers, window)

    # 画当前 center 及相对零位
    if T_center_avg is not None:
        rvec_c, _ = cv2.Rodrigues(T_center_avg[:3,:3])
        tvec_c = T_center_avg[:3,3].reshape(3,1)
        cv2.drawFrameAxes(frame, K, dist, rvec_c, tvec_c, 0.05)

        if T0 is not None:
            T_rel = inv_T(T0) @ T_center_avg
            rv_deg = rotvec_deg_from_T(T_rel)
            text = f"Enq:{enqueue_count} | Rel t(mm): {T_rel[0,3]*1000:.1f}, {T_rel[1,3]*1000:.1f}, {T_rel[2,3]*1000:.1f} | rot(deg): {np.linalg.norm(rv_deg):.2f}"
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        print("过滤后的平均 Center 位姿:\n", T_center_avg)

    # ====== 帧率检测 ======
    frame_count += 1
    now = time.time()
    if now - last_time >= 1.0:
        fps = frame_count / (now - last_time)
        print(f"FPS: {fps:.2f}")
        frame_count = 0
        last_time = now

    # 叠字显示入队计数，方便确认何时达到 50
    cv2.putText(frame, f"Enqueued: {enqueue_count}/{zero_target_enqueue}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2, cv2.LINE_AA)

    cv2.imshow("Center Pose", frame)
    if cv2.waitKey(1) == 27:  # Esc 退出
        break

cap.release()
cv2.destroyAllWindows()
# ====== 结束 ======
