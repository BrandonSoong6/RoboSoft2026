import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_spline_U_centered(step_mm=0.1, z_base=200.0,
                               height=100.0,
                               base_ratio=0.5,
                               radius_ratio=0.25,
                               save_path="spline_U_centered.npz"):
    """
    生成中心在 (0,0,z_base) 的字母 U 轨迹：
      - 高度固定为 height；
      - 宽度 = height * base_ratio；
      - 半径 = height * radius_ratio；
      - 整体中心在 (0, 0, z_base)。
    """
    H = float(height)
    W = float(height * base_ratio)
    R = float(height * radius_ratio)

    z = z_base
    half_h = H / 2.0

    # 顶部和半圆上缘位置
    top_y = half_h
    arc_top_y = -half_h + R  # 半圆上缘（竖线下端）

    # 关键点
    top_left  = np.array([-W/2, top_y, z])
    top_right = np.array([ W/2, top_y, z])
    left_arc_start  = np.array([-W/2, arc_top_y, z])
    right_arc_start = np.array([ W/2, arc_top_y, z])

    # 第一笔：左竖线（上→下）
    n1 = int(np.ceil((H - R) / step_mm)) + 1
    seg1 = np.linspace(top_left, left_arc_start, n1)

    # 第二笔：底部半圆（左下→右下，向下凹）
    cx = 0.0
    cy = -half_h + R
    n2 = int(np.ceil(np.pi * R / step_mm)) + 1
    theta = np.linspace(np.pi, 0.0, n2)
    seg2 = np.zeros((n2, 3))
    seg2[:, 0] = cx + R * np.cos(theta)
    seg2[:, 1] = cy - R * np.sin(theta)
    seg2[:, 2] = z

    # 第三笔：右竖线（下→上）
    n3 = int(np.ceil((H - R) / step_mm)) + 1
    seg3 = np.linspace(right_arc_start, top_right, n3)

    # 拼接
    points = np.vstack([seg1, seg2[1:], seg3[1:]])

    np.savez(save_path, points=points)
    print(f"[INFO] Saved centered U (H={H}, W={W}, R={R}) to {save_path}. "
          f"Total points: {len(points)}")
    print("[CHK] seg1 end -> seg2 start:", np.linalg.norm(seg1[-1] - seg2[0]))
    print("[CHK] seg2 end -> seg3 start:", np.linalg.norm(seg2[-1] - seg3[0]))

    return points


def plot_spline_3d(points, title="3D Centered 'U' Trajectory"):
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

    # 原点参考
    ax.scatter([0],[0],[points[0,2]], s=30, marker='x')
    plt.show()


if __name__ == "__main__":
    pts = generate_spline_U_centered()
    plot_spline_3d(pts)
