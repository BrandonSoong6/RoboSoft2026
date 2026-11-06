import numpy as np
import matplotlib.pyplot as plt

def _sample_polyline_3d(verts, step_mm=2.0):
    """按固定步长对3D折线均匀采样，包含首尾点且闭合不重复。"""
    pts = [verts[0].copy()]
    for i in range(len(verts)-1):
        a, b = verts[i], verts[i+1]
        seg = b - a
        L = np.linalg.norm(seg)
        if L == 0: 
            continue
        n = int(np.ceil(L / step_mm))
        for k in range(1, n+1):
            t = min(1.0, k / n)
            pts.append(a + t * seg)
    # 若闭合，去掉重复终点
    if np.linalg.norm(pts[0] - pts[-1]) < 1e-9:
        pts = pts[:-1]
    return np.array(pts)

def generate_H_by_recipe(L=40.0, x=5.0, y=20.0,
                         step_mm=2.0, sz = 200.0,
                         save_path="spline_trajectory.npz"):
    """按照指定笔顺生成单闭环'H'（气泡字外沿+横梁边缘一笔走完）"""
    # 参数校验
    assert L > 0 and x > 0 and y > 0, "L/x/y 必须为正数"
    assert L > x, "必须满足 L > x（横梁厚度小于总高度）"

    sx = -x-y/2
    sy = -L/2
    z = sz

    # 按“上L, 右x, 下(L-x)/2, 右y, 上(L-x)/2, 右x, 下L, 左x, 上(L-x)/2, 左y, 下(L-x)/2, 左x”
    h = (L - x) / 2.0

    p0  = np.array([sx,        sy,        z])          # 起点
    p1  = p0 + np.array([0.0,  L,         0.0])
    p2  = p1 + np.array([x,    0.0,       0.0])
    p3  = p2 + np.array([0.0, -h,         0.0])
    p4  = p3 + np.array([y,    0.0,       0.0])
    p5  = p4 + np.array([0.0,  h,         0.0])
    p6  = p5 + np.array([x,    0.0,       0.0])
    p7  = p6 + np.array([0.0, -L,         0.0])
    p8  = p7 + np.array([-x,   0.0,       0.0])
    p9  = p8 + np.array([0.0,  h,         0.0])
    p10 = p9 + np.array([-y,   0.0,       0.0])
    p11 = p10+ np.array([0.0, -h,         0.0])
    p12 = p11+ np.array([-x,   0.0,       0.0])        # 回到起点

    verts = np.stack([p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12], axis=0)

    # 采样为等步长点列（闭环）
    points = _sample_polyline_3d(verts, step_mm=step_mm)
    # 末尾再接回起点，显式闭合
    points = np.vstack([points, points[0]])

    np.savez(save_path, points=points)
    print(f"[INFO] Saved H one-loop trajectory with {len(points)} points to {save_path}")
    return points

def plot_spline_3d(points):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(points[:,0], points[:,1], points[:,2], '-o', markersize=2)
    ax.set_title("3D H One-Loop (by recipe)")
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")

    max_range = np.ptp(points, axis=0).max()/2
    mid = points.mean(axis=0)
    ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
    ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
    ax.set_zlim(mid[2]-max_range, mid[2]+max_range)
    plt.show()

if __name__ == "__main__":
    # 例子：中间净间距 20mm，笔画厚度 5mm，总高 40mm
    pts = generate_H_by_recipe(L=80.0, x=10.0, y=40.0, step_mm=0.5, sz = 200.0,
                               save_path="spline_trajectory.npz")
    plot_spline_3d(pts)
