import subprocess
import numpy as np
import cv2
from pupil_apriltags import Detector

# ====== 加载相机参数 ======
calib = np.load("calibration_data.npz")
K, dist = calib["K"], calib["dist"]

# ====== Apriltag 设置 ======
detector = Detector(
    families="tag36h11",
    nthreads=16,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=True,
    decode_sharpening=0.25,
)

device_name = "HD Pro Webcam C920"
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

marker_length = 0.04385  # m，黑框边长
frame_size = w * h * 3

while True:
    raw_frame = pipe.stdout.read(frame_size * 2)
    if not raw_frame:
        continue

    if len(raw_frame) >= frame_size:
        raw_frame = raw_frame[-frame_size:]
        try:
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((h, w, 3)).copy()
        except ValueError:
            continue

        # 灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ====== Apriltag 检测（直接解姿态） ======
        results = detector.detect(
            gray, estimate_tag_pose=True,
            camera_params=[K[0,0], K[1,1], K[0,2], K[1,2]],
            tag_size=marker_length
        )

        obs = {}  # 存三个tag的position

        for r in results:
            tag_id = r.tag_id
            if tag_id not in [0, 1, 2]:
                continue

            # pose_t: 平移向量 (3,)
            # pose_R: 旋转矩阵 (3,3)
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = r.pose_R
            T[:3, 3] = r.pose_t.reshape(3)

            print(f"\nID {tag_id} 的位姿矩阵:\n{T}")

            obs[tag_id] = r.pose_t.reshape(3)

            # 绘制tag
            corners = np.array(r.corners, dtype=np.int32)
            cv2.polylines(frame, [corners], True, (0,255,0), 2)
            cv2.drawFrameAxes(frame, K, dist,
                              cv2.Rodrigues(r.pose_R)[0],
                              r.pose_t.reshape(3,1),
                              0.03)
            cX, cY = corners.mean(axis=0).astype(int)
            cv2.putText(frame, f"ID {tag_id}", (cX, cY-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # ===== 如果三个ID都看到了，计算中心和法向 =====
        if len(obs) == 3:
            p0, p1, p2 = obs[0], obs[1], obs[2]
            center = (p0 + p1 + p2) / 3.0

            # 法向 (右手)
            v01 = p1 - p0
            v12 = p2 - p1
            z_axis = np.cross(v01, v12)
            z_axis /= np.linalg.norm(z_axis)

            # x 轴：center 指向 id0
            x_axis = p0 - center
            x_axis /= np.linalg.norm(x_axis)

            # y 轴：右手
            y_axis = np.cross(z_axis, x_axis)
            y_axis /= np.linalg.norm(y_axis)

            # 再正交化 x
            x_axis = np.cross(y_axis, z_axis)
            x_axis /= np.linalg.norm(x_axis)

            R_cb = np.column_stack([x_axis, y_axis, z_axis])
            T_cb = np.eye(4, dtype=np.float32)
            T_cb[:3,:3] = R_cb
            T_cb[:3, 3] = center

            print("Center pose (T_cam<-center):\n", T_cb)

            # 投影显示中心点
            center_pt, _ = cv2.projectPoints(center.reshape(1,3),
                                             np.zeros((3,1)), np.zeros((3,1)), K, dist)
            center_pt = tuple(center_pt[0].ravel().astype(int))
            cv2.circle(frame, center_pt, 5, (0,255,255), -1)
            cv2.putText(frame, "Center", (center_pt[0]+10, center_pt[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            # 在图像上画坐标轴
            rvec_vis, _ = cv2.Rodrigues(R_cb)
            tvec_vis = center.reshape(3,1).astype(np.float32)
            cv2.drawFrameAxes(frame, K, dist, rvec_vis, tvec_vis, 0.05)

        cv2.imshow("Apriltag Pose", frame)

    if cv2.waitKey(1) == 27:
        break

pipe.stdout.close()
pipe.terminate()
pipe.wait()
cv2.destroyAllWindows()
