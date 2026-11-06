import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import matplotlib.font_manager as fm
from matplotlib import animation

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


def plot_saved_trajectories(folder=".", prefix="traj_meas_demo1_final_", file_n=3, hz=4, save_video=True):
    """播放 N→U→S 三段轨迹动画并导出视频"""
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

    # --- 统一色系 ---
    colors_base = {
        1: (198/255, 28/255, 34/255, 1.0),   # 红 #c61c22
        2: (157/255, 206/255, 229/255, 1.0), # 蓝 #9dcee5
        3: (0/255, 176/255, 176/255, 1.0),   # 青 #00b0b0
    }
    colors_dark = {k: darken(v, 0.7) for k, v in colors_base.items()}
    label_map = {1: "N", 2: "U", 3: "S"}

    all_pts, error_dict, trajs_info = [], {}, []

    # === 依次加载 N→U→S ===
    for i in [1, 2, 3]:
        fname = os.path.join(folder, f"{prefix}{i}.npz")
        if not os.path.exists(fname):
            print(f"⚠️ 找不到文件 {fname}，跳过。")
            continue

        data = np.load(fname)
        traj = data["traj_meas"]
        spline = data["spline_points"] if "spline_points" in data else None

        color_main = colors_base.get(i)
        color_dark = colors_dark.get(i)
        label = label_map.get(i, str(i))

        # 样条线：深色实线
        if spline is not None:
            ax.plot(spline[:, 0], spline[:, 1], spline[:, 2],
                    '-', color=color_dark[:3], alpha=color_dark[3],
                    linewidth=3.0, label=f"Spline {label}")
            all_pts.append(spline)

        # 误差计算
        tree = cKDTree(spline)
        dists, _ = tree.query(traj)
        error_dict[label] = dists

        # 创建空轨迹和空点对象（后续由动画更新）
        line, = ax.plot([], [], [], '--', color=color_main[:3],
                        alpha=color_main[3], linewidth=3.0, label=f"Measured {label}")
        point = ax.scatter([], [], [], s=250, facecolors='none',
                           edgecolors=color_main[:3], alpha=color_main[3], linewidths=2.0)

        trajs_info.append({'traj': traj, 'line': line, 'point': point, 'color': color_main, 'label': label})
        all_pts.append(traj)
        print(f"✅ 已加载 {fname}")

    # === 自动调整视角 ===
    if all_pts:
        all_pts = np.vstack(all_pts)
        x_mid, y_mid, z_mid = np.mean(all_pts, axis=0)
        max_range = np.ptp(all_pts, axis=0).max() / 2
        ax.set_xlim(x_mid - max_range, x_mid + max_range)
        ax.set_ylim(y_mid - max_range, y_mid + max_range)
        ax.set_zlim(z_mid - max_range * 0.6, z_mid + max_range * 0.6)
    
    # ✅ 固定视角
    ax.view_init(elev=30.6, azim=-73.47)

    legend = ax.legend(fontsize=16, loc='best')
    for text in legend.get_texts():
        text.set_fontweight('normal')
        text.set_fontname('Times New Roman')

    # === 动画部分 ===
    lens = [len(t['traj']) for t in trajs_info]
    total_frames = sum(lens)
    interval_ms = int(1000 / hz)

    blank_frames = int(0.25 * hz)  # 0.25秒空白阶段
    total_frames = blank_frames + sum(lens)

    def update(frame):
        artists = []

        # --- 前 0.25s：只显示样条线，不显示任何测量轨迹 ---
        if frame < blank_frames:
            for t in trajs_info:
                t['line'].set_data([], [])
                t['line'].set_3d_properties([])
                t['point']._offsets3d = ([], [], [])
            return artists

        # --- 从第 blank_frames 帧后才开始绘制轨迹 ---
        frame_adj = frame - blank_frames
        acc = 0
        for t in trajs_info:
            L = len(t['traj'])
            traj = t['traj']

            if frame_adj < acc:
                t['line'].set_data([], [])
                t['line'].set_3d_properties([])
                t['point']._offsets3d = ([], [], [])
            elif acc <= frame_adj < acc + L:
                idx = frame_adj - acc + 1
                xs, ys, zs = traj[:idx, 0], traj[:idx, 1], traj[:idx, 2]
                t['line'].set_data(xs, ys)
                t['line'].set_3d_properties(zs)
                t['point'].remove()
                t['point'] = ax.scatter(xs, ys, zs, s=250,
                                        facecolors='none',
                                        edgecolors=t['color'][:3],
                                        alpha=t['color'][3],
                                        linewidths=2.0)
            else:
                xs, ys, zs = traj[:, 0], traj[:, 1], traj[:, 2]
                t['line'].set_data(xs, ys)
                t['line'].set_3d_properties(zs)
                t['point'].remove()
                t['point'] = ax.scatter(xs, ys, zs, s=250,
                                        facecolors='none',
                                        edgecolors=t['color'][:3],
                                        alpha=t['color'][3],
                                        linewidths=2.0)
            acc += L
            artists.extend([t['line'], t['point']])
        return artists

    anim = animation.FuncAnimation(fig, update, frames=range(total_frames),
                                   interval=interval_ms, blit=False, repeat=False)

    # === ✅ 保存关键帧（None, N, U, S） ===
    lens = [len(t['traj']) for t in trajs_info]
    frame_N_end = lens[0] - 1
    frame_U_end = lens[0] + lens[1] - 1
    frame_S_end = lens[0] + lens[1] + lens[2] - 1

    key_frames = {
        "N_done": frame_N_end,
        "U_done": frame_U_end,
        "S_done": frame_S_end
    }

    # --- 先保存真正空白的 None_done ---
    for t in trajs_info:
        t['line'].set_data([], [])
        t['line'].set_3d_properties([])
        t['point']._offsets3d = ([], [], [])
    output_path = os.path.join(folder, "demo1_None_done.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"🖼️ 已保存关键帧: {output_path}")

    # --- 再保存 N/U/S 三阶段 ---
    for name, frame_idx in key_frames.items():
        update(frame_idx)
        output_path = os.path.join(folder, f"demo1_{name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"🖼️ 已保存关键帧: {output_path}")

    # === ✅ 同时保存动画（可选）===
    output_video = os.path.join(folder, "demo1_plot.mp4")
    print(f"🎞️ 正在导出动画到 {output_video} ...")
    writer = animation.FFMpegWriter(fps=hz, bitrate=2000)
    anim.save(output_video, writer=writer, dpi=150)
    print("✅ 视频已保存成功！")

    plt.show()

if __name__ == "__main__":
    plot_saved_trajectories()
