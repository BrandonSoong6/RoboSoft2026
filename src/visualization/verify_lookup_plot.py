import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time
from canfield_plot import chain_Ts_for_arm_manhattan, _draw_axis_arrow, _draw_line_with_label
from scipy.spatial.transform import Rotation as R

def plot_from_lookup(row, ax,
                     side=93.16, axis_to_center_deg=-16.0,
                     link_params=(39.5,34.1,5), h=10.0):
    """
    使用打表结果（主动角 + 被动角）在给定 ax 上绘制 Canfield 机构。
    —— 保留你原本的所有绘制步骤，不做省略。
    """
    # 主动角
    q_actives_deg = [row["q1_deg"], row["q2_deg"], row["q3_deg"]]
    q1s = np.deg2rad(q_actives_deg)

    # 从表格构造 q_pass（q4 = -q2）
    q_pass = []
    for k in range(1, 4):
        q2 = row[f"q2_{k}_deg"]
        q3 = row[f"q3_{k}_deg"]
        q4 = -q2
        q_pass.extend([np.deg2rad(q2), np.deg2rad(q3), np.deg2rad(q4)])
    q_pass = np.array(q_pass)

    # ========= 绘图逻辑（不省略） =========
    colors = {"link1":"red", "link2":"green", "link3":"blue", "link4":"purple"}
    A_pts, B_pts, z4_list, nodes = [], [], [], []

    for i in range(3):
        q1 = q1s[i]
        q2, q3, q4 = q_pass[3*i:3*i+3]
        T0,T1,T2d,T2a,T3,T4 = chain_Ts_for_arm_manhattan(
            i, q1, q2, q3, q4, side, axis_to_center_deg, link_params, h
        )
        A_pts.append(T0[:3,3]); B_pts.append(T4[:3,3]); z4_list.append(T4[:3,2])
        for tag, T in [("T0",T0),("T1",T1),("T2d",T2d),("T2a",T2a),("T3",T3),("T4",T4)]:
            nodes.append((i, tag, T))

        # 连杆骨架
        ax.plot([T0[0,3], T1[0,3]],[T0[1,3], T1[1,3]],[T0[2,3], T1[2,3]], color=colors["link1"], marker="o")
        ax.plot([T1[0,3], T2d[0,3], T2a[0,3]],
                [T1[1,3], T2d[1,3], T2a[1,3]],
                [T1[2,3], T2d[2,3], T2a[2,3]], color=colors["link2"], marker="o")
        ax.plot([T2a[0,3], T3[0,3]],[T2a[1,3], T3[1,3]],[T2a[2,3], T3[2,3]], color=colors["link3"], marker="o")
        ax.plot([T3[0,3], T4[0,3]],[T3[1,3], T4[1,3]],[T3[2,3], T4[2,3]], color=colors["link4"], marker="o")

    # 三角形轮廓（底板）
    A_loop = np.vstack([A_pts[0], A_pts[1], A_pts[2]])
    ax.add_collection3d(
        Poly3DCollection([A_loop], 
                        facecolors="black", alpha=0.15,  # 半透明填充
                        edgecolors="black", linewidths=1)
    )

    # 三角形轮廓（上板）
    B_loop = np.vstack([B_pts[0], B_pts[1], B_pts[2]])
    ax.add_collection3d(
        Poly3DCollection([B_loop], 
                        facecolors="red", alpha=0.3,
                        edgecolors="red", linewidths=1)
    )

    # 圆心与姿态（原样保留）
    C = (B_pts[0] + B_pts[1] + B_pts[2]) / 3.0
    n_top = np.cross(B_pts[1]-B_pts[0], B_pts[2]-B_pts[0])
    n_top /= np.linalg.norm(n_top) + 1e-12
    if n_top[2] < 0: 
        n_top = -n_top

    x_axis = (B_pts[0] - C)
    x_axis /= np.linalg.norm(x_axis) + 1e-12
    y_axis = np.cross(n_top, x_axis)
    R_top = np.column_stack([x_axis, y_axis, n_top])
    eul = R.from_matrix(R_top).as_euler("zyx", degrees=True)

    # 终端打印
    print("== 打表验证 ==")
    print("主动角 (deg):", q_actives_deg)
    print("被动角 (deg):", np.rad2deg(q_pass))
    print("C 坐标:", np.round(C,3), " 姿态(zyx°):", np.round(eul,2))

    # 上板姿态箭头
    _draw_axis_arrow(ax, C, n_top, "b", "n_top", length=15)

    # 固定视域范围
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_zlim(0, 200)   # z 轴一般正向，所以可以偏高一些

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Canfield Joint — Lookup 动态演示 (4 Hz)")

if __name__ == "__main__":
    df = pd.read_csv("canfield_lookup_pose_0p4deg_full.csv")
    df_ok = df[df["res_norm"] < 1e-3].reset_index(drop=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        # 用“时间”作为随机种子，确保每帧都能抽到不同样本
        seed = int(time.time() * 1e6) & 0xffffffff
        row = df_ok.sample(n=1, random_state=seed).iloc[0]

        ax.cla()  # 清空上一帧
        plot_from_lookup(row, ax)  # 在同一 Axes 上完整重绘
        # blit=False（3D绘图不适合 blit），返回空列表即可
        return []

    # 每 250 ms 更新一次 → 4 Hz
    ani = animation.FuncAnimation(fig, update, interval=250, blit=False)
    plt.show()
