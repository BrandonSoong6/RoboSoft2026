import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_spline(step_mm=5.0, save_path="spline_trajectory.npz"):
    # 起点和终点
    start = np.array([0.0, 0.0, 200.0])
    end   = np.array([50.0, 0.0, 200.0])

    # ---------------- 直线段 ----------------
    line_len = np.linalg.norm(end - start)
    n_line = int(np.ceil(line_len / step_mm)) + 1
    line_points = np.linspace(start, end, n_line)

    # ---------------- 圆弧段（整圈） ----------------
    center = np.array([0.0, 0.0, 200.0])
    radius = 50.0
    arc_len = 2*np.pi*radius
    n_arc = int(np.ceil(arc_len / step_mm)) + 1
    theta = np.linspace(0, 2*np.pi, n_arc)
    circle_points = np.zeros((len(theta),3))
    circle_points[:,0] = center[0] + radius*np.cos(theta)
    circle_points[:,1] = center[1] + radius*np.sin(theta)
    circle_points[:,2] = 200.0

    # ---------------- 拼接 ----------------
    points = np.vstack([line_points, circle_points[1:]])

    # 保存
    np.savez(save_path, points=points)
    print(f"[INFO] Saved spline with {len(points)} points to {save_path}")

    return points

def plot_spline_3d(points):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(points[:,0], points[:,1], points[:,2], '-o', markersize=3)
    ax.set_title("3D Spline Trajectory")
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")

    # 设置相同尺度
    max_range = np.ptp(points, axis=0).max() / 2.0
    mid_x = (points[:,0].max() + points[:,0].min()) / 2
    mid_y = (points[:,1].max() + points[:,1].min()) / 2
    mid_z = (points[:,2].max() + points[:,2].min()) / 2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()

if __name__=="__main__":
    pts = generate_spline(step_mm=5.0)
    plot_spline_3d(pts)
