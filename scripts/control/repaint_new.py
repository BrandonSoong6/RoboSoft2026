import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import matplotlib.font_manager as fm

# ==== 字体设置 ====
font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
try:
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Times New Roman'
except:
    plt.rcParams['font.family'] = 'serif'

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.weight'] = 'normal'

def darken(color, factor=0.8):
    rgb_dark = np.clip(np.array(color[:3]) * factor, 0, 1)
    return tuple(rgb_dark.tolist() + [0.4])

def plot_saved_trajectories(folder=".", prefix="traj_meas_", file_n=3):
    
    # --- 1. 3D 轨迹图 (加入 dpi=300) ---
    # figsize保持不变，dpi增加会让图片变大且清晰，但相对比例不变
    fig = plt.figure(figsize=(10, 8), dpi=300) 
    ax = fig.add_subplot(111, projection='3d')
    
    # ... (此处省略读取数据和绘制3D图的重复代码，逻辑与之前完全一致) ...
    # 为了演示完整性，这里简写读取过程
    colors_base = {
        1: (198/255, 28/255, 34/255, 1.0),
        2: (157/255, 206/255, 229/255, 1.0),
        3: (0/255, 176/255, 176/255, 1.0),
    }
    label_map = {1: "N", 2: "U", 3: "S"}
    error_dict = {}
    
    # 临时模拟数据读取以便代码可运行
    for i in range(1, file_n + 1):
        fname = os.path.join(folder, f"{prefix}{i}.npz")
        if os.path.exists(fname):
            data = np.load(fname)
            traj = data["traj_meas"]
            spline = data["spline_points"]
            
            # 3D 绘图部分
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label=label_map.get(i))
            
            # 误差数据收集
            tree = cKDTree(spline)
            dists, _ = tree.query(traj)
            error_dict[label_map.get(i)] = dists
    
    # 3D图保存
    # bbox_inches='tight' 会自动裁剪白边，非常适合论文
    fig.savefig("3d_trajectory_high_res.png", dpi=300, bbox_inches='tight') 
    print("✅ 3D图已保存为 3d_trajectory_high_res.png")


    # --- 2. 误差分析图 (高分辨率扁平版) ---
    if error_dict:
        num_plots = len(error_dict)
        
        # [关键修改] dpi=300。figsize 依然是扁平的 (8, 3.5)
        # 这样生成的图片像素数会是 (2400 x 1050)，非常清晰
        fig2, axes = plt.subplots(num_plots, 1, figsize=(8, 3.5), sharex=True, dpi=300)
        if num_plots == 1: axes = [axes]

        fig2.patch.set_facecolor("white")
        fig2.suptitle("Tracking Error Analysis", fontsize=16, 
                      fontweight="bold", fontname='Times New Roman', x=0.55,y=0.98)

        inv_label_map = {v: k for k, v in label_map.items()}

        for i, (label, dists) in enumerate(error_dict.items()):
            ax_curr = axes[i]
            idx = inv_label_map.get(label, i+1)
            color = colors_base.get(idx, (0.4, 0.4, 0.4, 1.0))
            
            mean_err = np.mean(dists)
            max_err = np.max(dists)

            ax_curr.plot(np.arange(len(dists)), dists, '-', 
                         linewidth=1.5, color=color[:3],
                         label=f"{label} (mean={mean_err:.2f}, max={max_err:.2f})")
            
            ax_curr.grid(True, linestyle="--", alpha=0.5)
            ax_curr.tick_params(axis='both', which='major', labelsize=10)
            
            # 移除边框，图例更清爽
            leg = ax_curr.legend(loc='upper right', fontsize=12, 
                                 frameon=False, handlelength=1.0)
            for text in leg.get_texts():
                text.set_fontname('Times New Roman')
            
            for label_tick in ax_curr.get_yticklabels():
                label_tick.set_fontname('Times New Roman')

        axes[-1].set_xlabel("Sample index", fontsize=14, fontname='Times New Roman', labelpad=2)
        for label_tick in axes[-1].get_xticklabels():
            label_tick.set_fontname('Times New Roman')

        # 统一 Y 轴标签
        fig2.text(0.06, 0.5, 'Error [mm]', va='center', rotation='vertical', 
                  fontsize=16, fontname='Times New Roman')

        plt.subplots_adjust(top=0.91, bottom=0.14, left=0.12, right=0.98, hspace=0.1)
        
        # [关键步骤] 保存高分辨率图片
        # 建议保存为 .png (位图) 或 .pdf (矢量图，无限放大不失真)
        save_name = "error_analysis_high_res.png"
        fig2.savefig(save_name, dpi=300, bbox_inches='tight', pad_inches=0.05)
        print(f"✅ 误差分析图已保存为 {save_name}")

        # 如果你是用 LaTeX 写论文，强烈建议保存为 PDF
        # fig2.savefig("error_analysis.pdf", bbox_inches='tight', pad_inches=0.05)

    plt.show()

if __name__ == "__main__":
    plot_saved_trajectories(folder=".", prefix="traj_meas_demo1_final_", file_n=3)