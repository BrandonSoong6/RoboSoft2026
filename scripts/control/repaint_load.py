import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.ndimage import gaussian_filter1d
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm

# ==== 手动注册 Times New Roman ====
font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.weight'] = 'normal'

FRONT_SIZE = 22

def limit_label_formatter(y, pos, limit=100):
    """只在 -limit~limit 范围内显示刻度文字"""
    if abs(y) > limit:
        return ""
    return f"{y:g}"

def darken(color, factor=0.8):
    """把颜色加深，factor 越小颜色越深"""
    rgb = mcolors.to_rgb(color)
    return tuple(factor * c for c in rgb)

def plot_servo_load(npz_path="servo_load_during_control.npz",
                    smooth_sigma=2.0,
                    offset=60.0):
    """读取并绘制舵机负载数据（双纵轴 + 两行三列图例布局）"""
    data = np.load(npz_path)
    t = data["t"]
    loads = -data["loads"]  # 取反推为正

    n_samples, n_servos = loads.shape
    print(f"[INFO] Loaded {npz_path} with {n_samples} samples × {n_servos} servos")

    # === 平滑滤波 ===
    if smooth_sigma > 0:
        loads_smooth = np.array([
            gaussian_filter1d(loads[:, i], sigma=smooth_sigma)
            for i in range(n_servos)
        ]).T
    else:
        loads_smooth = loads

    # === 处理 NaN/Inf ===
    mask = ~np.isfinite(loads_smooth)
    for i in range(n_servos):
        arr = loads_smooth[:, i]
        if np.any(mask[:, i]):
            arr[~np.isfinite(arr)] = np.interp(
                np.flatnonzero(~np.isfinite(arr)),
                np.flatnonzero(np.isfinite(arr)),
                arr[np.isfinite(arr)]
            )
            loads_smooth[:, i] = arr

    # === 配色 ===
    warm_colors = plt.cm.Reds(np.linspace(0.6, 0.9, 3))
    cool_colors = plt.cm.Blues(np.linspace(0.6, 0.9, 3))
    warm_ref, cool_ref = warm_colors[2], cool_colors[2]

    # === 绘图 ===
    fig, ax_left = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax_left.set_facecolor("white")
    ax_right = ax_left.twinx()

    # 上盘（左轴）
    for i in range(3):
        ax_left.plot(t, loads_smooth[:, i],
                     color=warm_colors[i], lw=2.2,
                     label=f"Servo {i+1} (UPPER)")

    # 下盘（右轴）
    for i in range(3, 6):
        ax_right.plot(t, loads_smooth[:, i],
                      color=cool_colors[i-3], lw=2.2,
                      label=f"Servo {i+1} (LOWER)")

    # === 坐标范围 ===
    valid_vals = loads_smooth[np.isfinite(loads_smooth)]
    lmin, lmax = (np.min(valid_vals) - 10, np.max(valid_vals) + 10) if valid_vals.size else (-50, 50)
    ax_left.set_ylim(lmin - offset, lmax)
    ax_right.set_ylim(lmin, lmax + offset)
    ax_left.set_xlim(-3, 63)
    ax_right.set_xlim(-3, 63)

    # === 零线与端线 ===
    ax_left.axhline(0, color=warm_ref, lw=1.8, ls="--", alpha=0.6)
    ax_right.axhline(0, color=cool_ref, lw=1.8, ls="--", alpha=0.6)

    # === 标签与样式 ===
    ax_left.set_xlabel("Time (s)", fontsize=18, fontname='Times New Roman', fontweight='normal')
    ax_left.set_ylabel("Upper Servos Load (% PWM duty)",
                       fontsize=18, fontname='Times New Roman',
                       color=darken(warm_ref, 0.8))
    ax_right.set_ylabel("Lower Servos Load (% PWM duty)",
                        fontsize=18, fontname='Times New Roman',
                        color=darken(cool_ref, 0.8))

    ax_left.tick_params(axis='x', labelsize=14, colors='black')
    ax_left.tick_params(axis='y', labelsize=14, colors=warm_ref)
    ax_right.tick_params(axis='y', labelsize=14, colors=cool_ref)

    formatter = ticker.FuncFormatter(lambda y, pos: limit_label_formatter(y, pos, limit=100))
    ax_left.yaxis.set_major_formatter(formatter)
    ax_right.yaxis.set_major_formatter(formatter)
    ax_left.yaxis.set_major_locator(ticker.MaxNLocator(nbins=15))
    ax_right.yaxis.set_major_locator(ticker.MaxNLocator(nbins=15))

    ax_left.grid(True, linestyle="--", alpha=0.35)
    ax_left.set_title("Servo Load Distribution During Control",
                      fontsize=22, fontweight="normal", fontname='Times New Roman')

    # === 图例优化（固定两行三列：1 2 3 / 4 5 6） ===
    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()

    lines_all  = lines_left + lines_right  # 预期顺序：1,2,3,4,5,6
    labels_all = labels_left + labels_right

    # 关键：为 column-major 排布重排成 [1,4,2,5,3,6]
    order = [0, 3, 1, 4, 2, 5]
    lines_ordered  = [lines_all[i]  for i in order]
    labels_ordered = [labels_all[i] for i in order]

    legend = ax_left.legend(lines_ordered, labels_ordered,
                            loc="upper right",
                            ncol=3,
                            fontsize=13,
                            frameon=True, facecolor='white', edgecolor='black',
                            columnspacing=0.8, handletextpad=0.4,
                            borderaxespad=0.4, labelspacing=0.4)
    legend.set_bbox_to_anchor((0.99, 0.985))
    for text in legend.get_texts():
        text.set_fontweight('normal')
        text.set_fontname('Times New Roman')

    plt.tight_layout()
    plt.show()

    # === 打印统计 ===
    avg_abs = np.nanmean(np.abs(loads), axis=0)
    avg_signed = np.nanmean(loads, axis=0)
    print("[INFO] 平均负载（符号反映方向）：")
    for i, (val_s, val_a) in enumerate(zip(avg_signed, avg_abs), start=1):
        print(f"  Servo {i}: mean={val_s:6.2f} %,  |mean|={val_a:6.2f} %")


if __name__ == "__main__":
    plot_servo_load("servo_load_during_control.npz", smooth_sigma=2.0, offset=80.0)
