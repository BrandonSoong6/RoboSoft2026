import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

# 读取数据
df = pd.read_csv("canfield_lookup_pose_0p4deg_full.csv")
df_success = df[df["success"] == 1]

# 抽样（避免箭头太多）
df_sample = df_success.sample(n=2000, random_state=0)

Cx = df_sample["Cx_mm"].values
Cy = df_sample["Cy_mm"].values
Cz = df_sample["Cz_mm"].values

# 根据欧拉角恢复旋转矩阵，取每个姿态的 z 轴方向
eul = df_sample[["eul_z_deg", "eul_y_deg", "eul_x_deg"]].values
R_mats = R.from_euler("zyx", eul, degrees=True).as_matrix()
Z_dirs = R_mats[:, :, 2]   # 每个矩阵的第3列就是 z 轴方向

# 创建三维图
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

# 散点
ax.scatter(Cx, Cy, Cz, c=Cz, cmap="viridis", s=5, alpha=0.6)

# 画箭头（每个点的局部 z 轴）
scale = 3.0  # 箭头长度缩放
ax.quiver(Cx, Cy, Cz,
          Z_dirs[:,0], Z_dirs[:,1], Z_dirs[:,2],
          length=scale, normalize=True, color="b", alpha=0.6)

ax.set_xlabel("Cx [mm]")
ax.set_ylabel("Cy [mm]")
ax.set_zlabel("Cz [mm]")
ax.set_title("Canfield Joint Pose Distribution with Local Z-axis")

plt.show()

# 随机抽 10 个 Cy<-10 且 z 轴 y 分量 > 0 的点
mask = (df_sample["Cy_mm"] < -10)
mask &= (Z_dirs[:,1] > 0)
sampled_idx = df_sample[mask].sample(n=10, random_state=42).index

for idx in sampled_idx:
    row = df_sample.loc[idx]
    z_dir = Z_dirs[list(df_sample.index).index(idx)]
    print(
        f"q_act=({row['q1_deg']:.1f}, {row['q2_deg']:.1f}, {row['q3_deg']:.1f}) "
        f"C=({row['Cx_mm']:.2f}, {row['Cy_mm']:.2f}, {row['Cz_mm']:.2f}) "
        f"z_dir={z_dir}, res_norm={row['res_norm']}"
    )

# 筛选 res_norm > 1e-5 的行
bad_rows = df[df["res_norm"] > 1e-5]

print(f"总共有 {len(bad_rows)} 行 res_norm > 1e-5")
print(bad_rows.head(20))  # 打印前 20 行看看
