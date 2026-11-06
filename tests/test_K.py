import numpy as np

calib = np.load("calibration_data.npz")
K, dist = calib["K"], calib["dist"]

print("=== 从 calibration_data.npz 读取的相机参数 ===")
print("K =\n", K)
print("dist =\n", dist.ravel())

fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])
print(f"\nSanity Check: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

if not (200 < fx < 5000 and 200 < fy < 5000):
    print("⚠️ fx/fy 数值异常，请检查标定流程或分辨率缩放")
if not (0 < cx < 1920 and 0 < cy < 1080):
    print("⚠️ 主点 cx,cy 不在图像范围内，请检查")
