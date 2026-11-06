import cv2, numpy as np, glob, os

# ========= 参数（请按你的路径修改） =========
INNER_ROWS, INNER_COLS = 5, 8                 # 5行 × 8列 内角点（注意行列）
SQUARE_SIZE = 14.57/1000.0                   # 棋盘格方块边长（米）                        
IMG_PATH = "./calibration_images_3/*.jpg"

# ========= 生成棋盘三维点（列在前！） =========
rows, cols = INNER_ROWS, INNER_COLS
objp = np.zeros((rows*cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * SQUARE_SIZE

objpoints, imgpoints, used_files = [], [], []

images = glob.glob(IMG_PATH)
print(f"找到 {len(images)} 张图片")

# 更稳的角点检测参数
cb_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

for i, f in enumerate(images, 1):
    img = cv2.imread(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # patternSize 传 (cols, rows)！！
    ret, corners = cv2.findChessboardCorners(gray, (cols, rows), cb_flags)
    print(f"[{i}/{len(images)}] {os.path.basename(f)} -> {ret}")
    if not ret:
        continue
    corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    objpoints.append(objp.copy())
    imgpoints.append(corners)
    used_files.append(f)

if len(objpoints) < 10:
    raise RuntimeError(f"成功角点不足（{len(objpoints)}），建议 ≥ 15。")

img_shape = cv2.imread(used_files[0]).shape[1::-1]  # (w,h)

# ========= 阶段A：稳健收敛（仅 k1,k2；切向=0；k3+固定） =========
calib_flags = (cv2.CALIB_ZERO_TANGENT_DIST |
               cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 |
               cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6)

rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_shape, None, None
)
print("\n阶段A：RMS =", rms)
print("K=\n", K)
print("dist(k1,k2,p1=0,p2=0,k3=0) =", dist.ravel())

# ========= 计算每张图误差，剔除离群 =========
def per_view_err(o, i, r, t, K, d):
    proj, _ = cv2.projectPoints(o, r, t, K, d)
    return float(np.sqrt(np.mean(np.sum((i - proj)**2, axis=2))))

errs = np.array([per_view_err(o, i, r, t, K, dist)
                 for o,i,r,t in zip(objpoints, imgpoints, rvecs, tvecs)])
med, mad = np.median(errs), np.median(np.abs(errs - np.median(errs))) + 1e-9
keep = errs < (med + 3.5*mad)

print("\n每张图误差（px）：")
for f,e,ok in zip(used_files, errs, keep):
    print(f"{os.path.basename(f):40s} {e:6.3f}px  {'KEEP' if ok else 'DROP'}")

objp2 = [o for o,kp in zip(objpoints, keep) if kp]
imgp2 = [i for i,kp in zip(imgpoints, keep) if kp]

# ========= 阶段B：在干净数据上可选放开 k3（仍固定切向=0） =========
calib_flags_B = (cv2.CALIB_ZERO_TANGENT_DIST |
                 cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6)

rmsB, KB, distB, rvecsB, tvecsB = cv2.calibrateCamera(
    objp2, imgp2, img_shape, None, None
)
print("\n阶段B：RMS =", rmsB)
print("K=\n", KB)
print("dist(k1,k2,0,0,k3) =", distB.ravel())

np.savez("calibration_data.npz", K=KB, dist=distB)
print("已保存到 calibration_data.npz")

# ========= 去畸变演示（先不用新矩阵，避免额外缩放） =========
test = cv2.imread(used_files[0])
und = cv2.undistort(test, KB, distB)  # 直接用 KB/distB
cv2.imwrite("undistorted.jpg", und)
print("\n已保存 undistorted.jpg")
