import os
import time
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from collections import deque

# ---------- Imports from your project ----------
try:
    from serial_commander import ArduinoComm
except Exception as e:
    print("[WARN] serial_commander.ArduinoComm 未加载：", e)
    ArduinoComm = None

# ----------------- 数据加载 -----------------
def load_fk_npz(fk_npz):
    d = np.load(fk_npz, allow_pickle=True)
    arr = d["arr"]
    q1_range, q2_range, q3_range = d["q1_range"], d["q2_range"], d["q3_range"]
    step = float(d["step"])
    return arr, q1_range, q2_range, q3_range, step

def fk_lookup_interp(q, arr, q1_range, q2_range, q3_range, step):
    q1, q2, q3 = q
    def idx_frac(qv, q_range):
        i = (qv - q_range[0]) / step
        j = int(np.floor(i))
        t = i - j
        j = np.clip(j, 0, len(q_range)-2)
        return j, t
    i1,t1 = idx_frac(q1, q1_range)
    i2,t2 = idx_frac(q2, q2_range)
    i3,t3 = idx_frac(q3, q3_range)

    vals_pos = arr[i1:i1+2, i2:i2+2, i3:i3+2, 9:12]   # (Cx,Cy,Cz)
    vals_eul = arr[i1:i1+2, i2:i2+2, i3:i3+2, 6:9]    # (z,y,x)

    def trilerp(vals):
        return (
            vals[0,0,0]*(1-t1)*(1-t2)*(1-t3) +
            vals[1,0,0]*t1*(1-t2)*(1-t3) +
            vals[0,1,0]*(1-t1)*t2*(1-t3) +
            vals[0,0,1]*(1-t1)*(1-t2)*t3 +
            vals[1,1,0]*t1*t2*(1-t3) +
            vals[1,0,1]*t1*(1-t2)*t3 +
            vals[0,1,1]*(1-t1)*t2*t3 +
            vals[1,1,1]*t1*t2*t3
        )
    pos = trilerp(vals_pos)
    eul = trilerp(vals_eul)
    quat = R.from_euler("zyx", eul, degrees=True).as_quat()
    return pos, quat

# 配置
FK_NPZ_PATH = "canfield_lookup_pose_0p4deg_full.npz"
SERIAL_PORT = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"
SERIAL_BAUD = 115200
CAM_INDEX = 2
TAG_SIZE = 0.04385
WORLD_INIT_FRAME = 50

# 常量
DOWN_28MM = -0.130
UP_VISUAL = 0.000
UP_MODEL = 0.102

def pose_to_T(Rm, tvec):
    T = np.eye(4)
    T[:3,:3] = Rm
    T[:3,3] = tvec.reshape(3)
    return T

def inv_T(T):
    Rm, t = T[:3,:3], T[:3,3]
    Ti = np.eye(4)
    Ti[:3,:3] = Rm.T
    Ti[:3,3] = -Rm.T @ t
    return Ti

def trans_z(d):
    T = np.eye(4)
    T[2,3] = d
    return T

def draw_axes(frame, K, dist, T_cam_obj, axis_len=0.06, thickness=2):
    Rm = T_cam_obj[:3,:3].astype(np.float32)
    t = T_cam_obj[:3,3].reshape(3,1).astype(np.float32)
    rvec,_ = cv2.Rodrigues(Rm)
    cv2.drawFrameAxes(frame,K,dist,rvec,t,axis_len)

def fmt_fixed(vec_m, scale=1000.0, width=10, prec=2):
    def f(v): return f"{v*scale:+0{width}.{prec}f}"
    return f"[{f(vec_m[0])} {f(vec_m[1])} {f(vec_m[2])}]"

def average_quat(quats):
    q0 = quats[0]
    qq = quats.copy()
    for i in range(1,len(qq)):
        if np.dot(qq[i],q0) < 0: qq[i] = -qq[i]
    A = np.zeros((4,4))
    for q in qq: A += np.outer(q,q)
    w,v = np.linalg.eigh(A)
    q_mean = v[:,np.argmax(w)]
    return q_mean/np.linalg.norm(q_mean)

def average_transform(Ts):
    if len(Ts)==0: return None
    t_avg = np.mean([T[:3,3] for T in Ts],axis=0)
    qs = np.array([R.from_matrix(T[:3,:3]).as_quat() for T in Ts])
    q_mean = average_quat(qs)
    Rm = R.from_quat(q_mean).as_matrix()
    T = np.eye(4); T[:3,:3]=Rm; T[:3,3]=t_avg
    return T

def setup_camera_and_detector():
    from pupil_apriltags import Detector
    calib = np.load("calibration_data.npz")
    K,dist = calib["K"].astype(np.float32), calib["dist"].astype(np.float32)
    marker2center = np.load("marker_to_center.npz")
    T_marker2center = {0:marker2center["T0"],1:marker2center["T1"],2:marker2center["T2"]}
    det = Detector(families="tag36h11",nthreads=16,quad_decimate=2.0)
    cap = cv2.VideoCapture(CAM_INDEX,cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
    return cap,det,K,dist,T_marker2center


# ---------- 视觉增强工具函数 ----------

history = deque(maxlen=5)   # 历史缓存 (frame_idx, 平移, 四元数)
window  = deque(maxlen=1)   # 最近合法位姿
pos_thresh = 0.02           # 平移阈值 (m)
rot_thresh = np.deg2rad(10) # 旋转阈值 (rad)
frame_idx = 0

zero_target_enqueue = 50
enqueue_count = 0
zero_saved = False
T0 = None
zero_file = "center_zero_pose.npz"

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
    T_est = np.eye(4)
    T_est[:3,:3] = R_est
    T_est[:3,3] = t_est
    return T_est

def push_candidate(T_candidates, window):
    """鲁棒入队逻辑：滤波 + 外推 + 零位保存"""
    global frame_idx, enqueue_count, zero_saved, T0
    frame_idx += 1

    if window:
        T_avg = average_transform(list(window))
        t_avg = T_avg[:3,3]
        q_avg = R.from_matrix(T_avg[:3,:3]).as_quat()
    else:
        T_avg, t_avg, q_avg = None, None, None

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

    if valid:
        T_new = average_transform(valid)
        window.append(T_new)
        enqueue_count += 1
        # 自动保存零位
        if (not zero_saved) and (enqueue_count == zero_target_enqueue):
            T0 = T_new.copy()
            offset_local = np.array([0.0,0.0,-0.125]) # 米为单位
            T0[:3,3] += T0[:3,:3] @ offset_local
            np.savez(zero_file,
                     T0=T0,
                     enqueue_index=enqueue_count,
                     frame_idx_when_saved=frame_idx,
                     saved_time=time.time())
            zero_saved = True
            print("✅ 已保存零位并下移125mm")
        push_history(frame_idx, T_new)
    else:
        if frame_idx % 2 == 0 and window:
            window.popleft()

    if not window:
        T_est = extrapolate_center(frame_idx)
        if T_est is not None:
            print(f"[Frame {frame_idx}] window empty → 外推")
            return T_est
    return average_transform(list(window)) if window else None

# ---------- 主流程 ----------

def main():
    arr,q1r,q2r,q3r,step = load_fk_npz(FK_NPZ_PATH)
    q_min = [q1r[0],q2r[0],q3r[0]]
    q_max = [q1r[-1],q2r[-1],q3r[-1]]
    cap,det,K,dist,T_marker2center = setup_camera_and_detector()
    comm = None
    if ArduinoComm:
        try: comm = ArduinoComm(port=SERIAL_PORT,baudrate=SERIAL_BAUD,timeout=0.5)
        except: pass

    frame_idx=0
    T_cam_world=None; T_W_cam=None
    T_b_center_50=None
    last_vis=None; last_model=None; last_err=None

    while True:
        ok,frame=cap.read()
        if not ok: continue
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        res = det.detect(gray, estimate_tag_pose=True,
            camera_params=[K[0,0],K[1,1],K[0,2],K[1,2]], tag_size=TAG_SIZE)


        T_cam_centers = []
        for r in res:
            if r.tag_id not in (0,1,2): continue
            if r.hamming != 0: continue
            err = compute_reproj_error(r.pose_R, r.pose_t, r.corners, K, dist, TAG_SIZE)
            if err > 2.0: continue

            T_cam_marker = pose_to_T(r.pose_R, r.pose_t.reshape(3))
            T_cam_centers.append(T_cam_marker @ T_marker2center[r.tag_id])

        # 滤波
        T_cam_center_meas = push_candidate(T_cam_centers, window)


        frame_idx+=1
        if frame_idx==WORLD_INIT_FRAME and T_cam_center_meas is not None:
            T_cam_world = T_cam_center_meas @ trans_z(DOWN_28MM)
            T_W_cam = inv_T(T_cam_world)
            if comm:
                q50=np.clip(np.array(comm.current_angles)+90.0,q_min,q_max)
                pos,quat=fk_lookup_interp(q50,arr,q1r,q2r,q3r,step)
                T_b_center_50=pose_to_T(R.from_quat(quat).as_matrix(),pos/1000.0)
            print("World fixed.")

        if T_W_cam is not None:
            # Vision end
            if T_cam_center_meas is not None:
                T_W_center_vis = T_W_cam @ T_cam_center_meas
                T_W_end_vis = T_W_center_vis @ trans_z(UP_VISUAL)
                last_vis=T_W_end_vis[:3,3]
                T_cam_end_vis = inv_T(T_W_cam) @ T_W_end_vis
                draw_axes(frame,K,dist,T_cam_end_vis,0.06,2)
            # Model end
            if comm and T_b_center_50 is not None:
                q_now=np.clip(np.array(comm.current_angles)+90.0,q_min,q_max)
                pos,quat=fk_lookup_interp(q_now,arr,q1r,q2r,q3r,step)
                T_b_center_now=pose_to_T(R.from_quat(quat).as_matrix(),pos/1000.0)
                T_rel=inv_T(T_b_center_50) @ T_b_center_now
                T_W_center_model = trans_z(+0.028) @ T_rel
                T_W_end_model = T_W_center_model @ trans_z(UP_MODEL)
                last_model=T_W_end_model[:3,3]
                T_cam_end_model = inv_T(T_W_cam) @ T_W_end_model
                draw_axes(frame,K,dist,T_cam_end_model,0.06,3)
            if last_vis is not None and last_model is not None:
                last_err = last_vis-last_model

        y0,dy=36,24
        def put(txt,y,col=(255,255,255)): cv2.putText(frame,txt,(18,y),cv2.FONT_HERSHEY_SIMPLEX,0.65,col,2)
        if last_vis is not None: put("W end (vision): "+fmt_fixed(last_vis),y0)
        if last_model is not None: put("W end (model ): "+fmt_fixed(last_model),y0+dy,(0,255,255))
        if last_err is not None: put("Δpos (mm)    : "+fmt_fixed(last_err),y0+2*dy,(0,255,0))

        cv2.imshow("World overlay",frame)
        if cv2.waitKey(1)&0xFF==27: break

    cap.release(); cv2.destroyAllWindows()
    if comm: comm.close()

if __name__=="__main__":
    main()