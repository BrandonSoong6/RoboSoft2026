import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import matplotlib.font_manager as fm

# ==== 手动注册 Times New Roman ====
font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.weight'] = 'normal'

def darken(color, factor=0.8):
    rgb_dark = np.clip(np.array(color[:3]) * factor, 0, 1)
    return tuple(rgb_dark.tolist() + [0.4])

def plot_saved_trajectories(folder=".", prefix="traj_meas_", file_n=3):
    """
    从保存的 npz 文件中读取每一段样条和实测轨迹并重绘，
    并绘制每段轨迹的误差分布（距离误差 vs 采样点编号）。
    """
    fig = plt.figure(figsize=(10, 8))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor("white")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_title("Canfield Joint End-Effector Following Performance",
                 fontsize=22, fontweight='normal', fontname='Times New Roman')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='z', labelsize=14)

    # --- 统一色系（来自图中） ---
    colors_base = {
        1: (198/255, 28/255, 34/255, 1.0),   # 红 #c61c22
        2: (157/255, 206/255, 229/255, 1.0), # 蓝 #9dcee5
        3: (0/255, 176/255, 176/255, 1.0),   # 青 #00b0b0
    }
    colors_dark = {k: darken(v, 0.7) for k, v in colors_base.items()}
    label_map = {1: "N", 2: "U", 3: "S"}

    all_pts = []
    error_dict = {}

    for i in range(1, file_n + 1):
        fname = os.path.join(folder, f"{prefix}{i}.npz")
        if not os.path.exists(fname):
            print(f"⚠️ 找不到文件 {fname}，跳过。")
            continue

        data = np.load(fname)
        traj = data["traj_meas"]
        spline = data["spline_points"] if "spline_points" in data else None

        color_main = colors_base.get(i, (0.3, 0.3, 0.3, 1.0))
        color_dark = colors_dark.get(i, (0.2, 0.2, 0.2, 1.0))
        label = label_map.get(i, f"{i}")

        # 绘制样条（深色）
        if spline is not None:
            ax.plot(
                spline[:, 0], spline[:, 1], spline[:, 2],
                '-', color=color_dark[:3], alpha=color_dark[3],
                linewidth=3.0, label=f"Spline {label}"
            )
            all_pts.append(spline)

        # 误差计算
        tree = cKDTree(spline)
        dists, _ = tree.query(traj)
        error_dict[label] = dists

        # 绘制实测轨迹（原色）
        ax.plot(
            traj[:, 0], traj[:, 1], traj[:, 2],
            '--', color=color_main[:3], alpha=color_main[3],
            linewidth=3.0, label=f"Measured {label}"
        )
        ax.scatter(
            traj[:, 0], traj[:, 1], traj[:, 2],
            s=250, facecolors='none', edgecolors=color_main[:3],
            alpha=color_main[3], linewidths=2.0
        )
        all_pts.append(traj)

        print(f"✅ 已加载并绘制 {fname}")

    # 自动调整视角
    if all_pts:
        all_pts = np.vstack(all_pts)
        x_mid, y_mid, z_mid = np.mean(all_pts, axis=0)
        max_range = np.ptp(all_pts, axis=0).max() / 2
        ax.set_xlim(x_mid - max_range, x_mid + max_range)
        ax.set_ylim(y_mid - max_range, y_mid + max_range)
        ax.set_zlim(z_mid - max_range * 0.6, z_mid + max_range * 0.6)

    legend = ax.legend(fontsize=16, loc='best')
    for text in legend.get_texts():
        text.set_fontweight('normal')
        text.set_fontname('Times New Roman')

    # --- 误差分析图 ---
    if error_dict:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        fig2.patch.set_facecolor("white")
        ax2.set_facecolor("white")
        ax2.set_title("Tracking Error per Sample Point",
                      fontsize=20, fontweight="normal", fontname='Times New Roman')
        ax2.set_xlabel("Sample index", fontsize=14, fontname='Times New Roman')
        ax2.set_ylabel("Distance Error [mm]", fontsize=14, fontname='Times New Roman')
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.set_ylim(0, 10)

        for i, (label, dists) in enumerate(error_dict.items(), 1):
            color = colors_base.get(i, (0.4, 0.4, 0.4, 1.0))
            mean_err = np.mean(dists)
            max_err = np.max(dists)
            ax2.plot(
                np.arange(len(dists)), dists, '-',
                linewidth=2.5, color=color[:3],
                label=f"{label} (mean={mean_err:.2f}, max={max_err:.2f})"
            )
            print(f"[{label}] 平均误差 = {mean_err:.2f} mm, 最大误差 = {max_err:.2f} mm")

        leg2 = ax2.legend(fontsize=13)
        for text in leg2.get_texts():
            text.set_fontweight('normal')
            text.set_fontname('Times New Roman')

        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    plot_saved_trajectories(folder=".", prefix="traj_meas_demo1_final_", file_n=3)
