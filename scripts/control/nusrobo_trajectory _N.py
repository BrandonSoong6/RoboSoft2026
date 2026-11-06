import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_spline_N_centered(step_mm=0.1, z_base=220.0,
                               height=100.0,  # 高度固定
                               base_ratio=0.5,  # 保持原始比例：60/100
                               save_path="spline_N_centered.npz"):
    """
    生成中心在 (0,0,z_base) 的字母 N 三笔轨迹。
    约束：
      - 高度固定为 height（默认 220），不建议改；
      - 宽度 = height * base_ratio（默认 0.6，保持原始比例）；
      - 轨迹中心 (x=0, y=0)，范围 x∈[-W/2, +W/2], y∈[-H/2, +H/2]。
    """

    H = float(height)
    W = float(height * base_ratio)

    # 以中心在 (0,0) 为基准定义四个关键点
    # 左下、左上、右下、右上（z = z_base）
    start        = np.array([-W/2, -H/2, z_base])   # 左下
    top_left     = np.array([-W/2,  H/2, z_base])   # 左上
    bottom_right = np.array([ +W/2, -H/2, z_base])  # 右下
    top_right    = np.array([ +W/2,  H/2, z_base])  # 右上

    # 第一笔：左竖线（下→上）
    n1 = int(np.ceil(H / step_mm)) + 1
    seg1 = np.linspace(start, top_left, n1)

    # 第二笔：左上 → 右下（斜线）
    L2 = np.linalg.norm(bottom_right - top_left)
    n2 = int(np.ceil(L2 / step_mm)) + 1
    seg2 = np.linspace(top_left, bottom_right, n2)

    # 第三笔：右竖线（下→上）
    n3 = int(np.ceil(H / step_mm)) + 1
    seg3 = np.linspace(bottom_right, top_right, n3)

    # 拼接（避免重复点）
    points = np.vstack([seg1, seg2[1:], seg3[1:]])

    # 保存
    np.savez(save_path, points=points)
    print(f"[INFO] Saved centered N (H={H}, W={W}, ratio={base_ratio}) to {save_path}. "
          f"Total points: {len(points)}")

    return points

def plot_spline_3d(points, title="3D Centered 'N' Trajectory"):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(points[:,0], points[:,1], points[:,2], '-o', markersize=3)
    ax.set_title(title)
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")

    # 等比例视角
    max_range = np.ptp(points, axis=0).max() / 2.0
    mid_x = (points[:,0].max() + points[:,0].min()) / 2
    mid_y = (points[:,1].max() + points[:,1].min()) / 2
    mid_z = (points[:,2].max() + points[:,2].min()) / 2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # 画出中心原点辅助参照
    ax.scatter([0],[0],[points[0,2]], s=30, marker='x')

    plt.show()


if __name__ == "__main__":
    # 只调 z_base；高度固定 220，宽度自动=220*0.6=132，中心 (0,0,z_base)
    pts = generate_spline_N_centered()
    plot_spline_3d(pts)
