import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

def Rz(deg):
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])

# === 读取基准表（q1 最大） ===
df_base = pd.read_csv("canfield_lookup_0p4deg_pose.csv")

rows = []
# 定义三个对称置换：旋转角度 + 主/被动角的索引循环
perms = [
    (0,   [0, 1, 2]),  # 原始顺序
    (120, [1, 2, 0]),  # +120°：q1→q2, q2→q3, q3→q1
    (-120,[2, 0, 1])   # -120°：q1→q3, q2→q1, q3→q2
]

for _, row in df_base.iterrows():
    # 基准旋转矩阵
    eul = [row["eul_z_deg"], row["eul_y_deg"], row["eul_x_deg"]]
    R_top = R.from_euler("zyx", eul, degrees=True).as_matrix()
    C = np.array([row["Cx_mm"], row["Cy_mm"], row["Cz_mm"]])

    for rot_deg, idx in perms:
        R_rot = Rz(rot_deg)
        # 圆心坐标旋转
        C_new = R_rot @ C
        # 姿态旋转
        R_new = R_rot @ R_top
        eul_new = R.from_matrix(R_new).as_euler("zyx", degrees=True)

        # 主动角置换
        q_actives = row[["q1_deg","q2_deg","q3_deg"]].values
        q_new = q_actives[idx]

        # 被动角置换：成对移动
        q2s = row[["q2_1_deg","q2_2_deg","q2_3_deg"]].values
        q3s = row[["q3_1_deg","q3_2_deg","q3_3_deg"]].values
        q2_new = q2s[idx]
        q3_new = q3s[idx]

        rows.append({
            "q1_deg": q_new[0], "q2_deg": q_new[1], "q3_deg": q_new[2],
            "q2_1_deg": q2_new[0], "q3_1_deg": q3_new[0],
            "q2_2_deg": q2_new[1], "q3_2_deg": q3_new[1],
            "q2_3_deg": q2_new[2], "q3_3_deg": q3_new[2],
            "eul_z_deg": eul_new[0],
            "eul_y_deg": eul_new[1],
            "eul_x_deg": eul_new[2],
            "Cx_mm": C_new[0], "Cy_mm": C_new[1], "Cz_mm": C_new[2],
            "success": row["success"],
            "res_norm": row["res_norm"]
        })

df_full = pd.DataFrame(rows)
df_full.to_csv("canfield_lookup_pose_0p4deg_full.csv", index=False)
print(f"[done] expanded {len(df_base)} → {len(df_full)} rows (补全 1/3 → 全空间)")
