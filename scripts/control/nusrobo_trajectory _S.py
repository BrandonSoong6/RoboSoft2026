import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_spline_S_centered(step_mm=0.1, z_base=180.0,
                               height=100.0,
                               base_ratio=0.5,
                               save_path="spline_S_centered.npz"):
    """
    生成中心在 (0,0,z_base) 的字母 S 轨迹。
    结构逻辑：
      - 字母总高 = height，宽 = height * base_ratio；
      - 上圆心在 (0, +height/4)，下圆心在 (0, -height/4)；
      - 上半弧：逆时针 3/4 圆；
      - 下半弧：顺时针 3/4 圆；
      - 整体几何中心 (0,0,z_base)。
    """
    z = z_base
    H = float(height)
    W = float(height * base_ratio)
    R = W / 2.0  # 半径由宽度决定，保持比例协调

    # ============ 上半弧 ============
    cx1, cy1 = 0.0, H / 4.0
    L1 = (3.0 * np.pi / 2.0) * R
    n1 = int(np.ceil(L1 / step_mm)) + 1
    theta1 = np.linspace(0.0, 3.0 * np.pi / 2.0, n1)  # 逆时针
    arc1 = np.zeros((n1, 3))
    arc1[:, 0] = cx1 + R * np.cos(theta1)
    arc1[:, 1] = cy1 + R * np.sin(theta1)
    arc1[:, 2] = z

    # ============ 下半弧 ============
    cx2, cy2 = 0.0, -H / 4.0
    L2 = (3.0 * np.pi / 2.0) * R
    n2 = int(np.ceil(L2 / step_mm)) + 1
    theta2 = np.linspace(np.pi / 2.0, -np.pi, n2)  # 顺时针
    arc2 = np.zeros((n2, 3))
    arc2[:, 0] = cx2 + R * np.cos(theta2)
    arc2[:, 1] = cy2 + R * np.sin(theta2)
    arc2[:, 2] = z

    # ============ 拼接 ============
    points = np.vstack([arc1, arc2[1:]])

    np.savez(save_path, points=points)
    print(f"[INFO] Saved centered S (H={H}, W={W}, R={R}) to {save_path}. Total points: {len(points)}")
    print("[CHK] arc1 end -> arc2 start:", np.linalg.norm(arc1[-1] - arc2[0]))

    return points


def plot_spline_3d(points, title="3D Centered 'S' Trajectory"):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(points[:, 0], points[:, 1], points[:, 2], '-o', markersize=3)
    ax.set_title(title)
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")

    max_range = np.ptp(points, axis=0).max() / 2.0
    mid_x = (points[:, 0].max() + points[:, 0].min()) / 2
    mid_y = (points[:, 1].max() + points[:, 1].min()) / 2
    mid_z = (points[:, 2].max() + points[:, 2].min()) / 2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # 中心原点
    ax.scatter([0], [0], [points[0, 2]], s=30, marker='x')
    plt.show()


if __name__ == "__main__":
    pts = generate_spline_S_centered()
    plot_spline_3d(pts)
