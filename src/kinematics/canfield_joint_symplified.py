import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ================= 工具与底座几何 =================
def _make_T0s(side, h, axis_to_center_deg):
    T0s = np.empty((3, 4, 4), dtype=float)
    R_ = side / np.sqrt(3.0)
    thetas = np.deg2rad([0.0, 120.0, 240.0])
    axis = np.deg2rad(axis_to_center_deg)
    for i in range(3):
        theta0 = thetas[i]
        beta = theta0 - axis
        xi, yi = R_*np.cos(theta0), R_*np.sin(theta0)
        T0s[i,:,:] = np.array([
            [-np.sin(beta), 0.0,  np.cos(beta), xi],
            [ np.cos(beta), 0.0,  np.sin(beta), yi],
            [ 0.0,          1.0,  0.0,          h ],
            [ 0.0,          0.0,  0.0,          1.0 ]
        ])
    return T0s

def _A1_fast(q1):
    ct, st = np.cos(q1), np.sin(q1)
    return np.array([[ct, 0.0,  st, 0.0],
                     [st, 0.0, -ct, 0.0],
                     [0.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])

def reflect_point_across_plane(P, C, n):
    return P - 2.0 * np.dot(P - C, n) * n

def orthonormalize_keep_z(x, z):
    """保持给定 z，不改变方向；用 x 构造正交基。"""
    z = z / (np.linalg.norm(z) + 1e-12)
    x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    if np.linalg.norm(y) < 1e-12:
        # 退化时给 x 一个稳定备选
        x = np.array([1.0, 0.0, 0.0])
        y = np.cross(z, x)
    y /= (np.linalg.norm(y) + 1e-12)
    x = np.cross(y, z); x /= (np.linalg.norm(x) + 1e-12)
    return x, y, z

def set_axes_equal(ax):
    """强制 3D 等尺度显示，避免视觉变形。"""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_mid   = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_mid   = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_mid   = np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_mid - plot_radius, x_mid + plot_radius])
    ax.set_ylim3d([y_mid - plot_radius, y_mid + plot_radius])
    ax.set_zlim3d([z_mid - plot_radius, z_mid + plot_radius])

# ================= 镜像模型 =================
class CanfieldMirrorModel:
    """
    q -> 由 A 与 L1 得到 M 面 -> 以 M 面镜像 A 得到 B -> 由 B 求 Bo 与姿态。
    默认下连杆方向取 A1 的局部 x 轴（axis_idx=0）。如需改为 y 或 z，改 axis_idx 即可。
    """
    def __init__(self, side=93.16, axis_to_center_deg=-16.0, h=10.0,
                 L1=75.0, axis_idx=0):
        self.side = float(side)
        self.axis_to_center_deg = float(axis_to_center_deg)
        self.h = float(h)
        self.L1 = float(L1)
        self.axis_idx = int(axis_idx)
        self.T0s = _make_T0s(self.side, self.h, self.axis_to_center_deg)

    def forward(self, q_deg):
        q_rad = np.deg2rad(np.asarray(q_deg, dtype=float))
        A = np.zeros((3,3)); M = np.zeros((3,3))
        for i in range(3):
            T0 = self.T0s[i]
            T1 = T0 @ _A1_fast(q_rad[i])
            Ai = T0[:3, 3].copy()
            dir_vec = T1[:3, self.axis_idx]
            Mi = Ai + self.L1 * dir_vec
            A[i], M[i] = Ai, Mi

        # M 面 → 镜像 A 得到 B
        n = np.cross(M[1] - M[0], M[2] - M[1])
        n = np.array([0,0,1]) if np.linalg.norm(n)<1e-12 else n/np.linalg.norm(n)
        C = M.mean(axis=0)
        B = np.array([reflect_point_across_plane(A[i], C, n) for i in range(3)])

        # 临时中心
        Bo0 = B.mean(axis=0)

        # 由 B 面法向和 Bo0 算姿态
        z = np.cross(B[1]-B[0], B[2]-B[0])
        z = np.array([0,0,1]) if np.linalg.norm(z)<1e-12 else z/np.linalg.norm(z)
        x0 = B[0] - Bo0
        if np.linalg.norm(x0)<1e-12: x0 = np.array([1,0,0])
        x,y,z = orthonormalize_keep_z(x0, z)
        Rbo = np.column_stack([x,y,z])

        # ==== 真正的 Bo 上移 ====
        Bo = Bo0 + Rbo[:,2] * 5.0

        return A, M, B, Bo, Rbo


# ================= 可视化 =================
def make_cylinder(center, normal, radius, height, color, alpha=0.2, n=60):
    """
    在给定圆心和法向的平面上生成一个厚度为 height 的圆柱体（上下两个圆面 + 侧面）。
    返回 Poly3DCollection。
    """
    # 正交基：在 normal 平面里找两个正交方向
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    if abs(normal[0]) < 0.9:
        tangent = np.cross(normal, [1,0,0])
    else:
        tangent = np.cross(normal, [0,1,0])
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)

    # 上下圆心
    c1 = center
    c2 = center + normal * height

    # 圆边点
    theta = np.linspace(0, 2*np.pi, n)
    circle1 = np.array([c1 + radius*(np.cos(t)*tangent + np.sin(t)*bitangent) for t in theta])
    circle2 = np.array([c2 + radius*(np.cos(t)*tangent + np.sin(t)*bitangent) for t in theta])

    # 构造面
    verts = []
    # 侧壁
    for i in range(n):
        verts.append([circle1[i], circle1[(i+1)%n], circle2[(i+1)%n], circle2[i]])
    # 上下盖
    verts.append(circle1)
    verts.append(circle2[::-1])

    poly = Poly3DCollection(verts, facecolors=color, edgecolors='none', alpha=alpha)
    return poly

def draw_scene(ax, A, M, B, Bo, Rbo, side_expect=None):
    ax.cla()
    ax.scatter(*A.T, c="#1f77b4", s=30, label="A (base)")
    ax.scatter(*M.T, c="#2ca02c", s=30, label="M (lower-link tips)")
    ax.scatter(*B.T, c="#d62728", s=30, label="B (reflected)")
    ax.scatter(*Bo,   c="k", s=50, label="Bo (上板中心)")

    # 连杆 A-M, M-B
    for Ai, Mi, Bi in zip(A, M, B):
        ax.plot([Ai[0], Mi[0]], [Ai[1], Mi[1]], [Ai[2], Mi[2]],
                color="#9e9e9e", lw=2)
        ax.plot([Mi[0], Bi[0]], [Mi[1], Bi[1]], [Mi[2], Bi[2]],
                color="#ff9800", lw=2)
        # Ai → 底板竖直杆
        ax.plot([Ai[0], Ai[0]], [Ai[1], Ai[1]], [Ai[2], 0],
                color="black", lw=3)

    # ==== 下板圆柱 ====
    R = 70.0
    base_center = np.array([0,0,0])
    base_normal = np.array([0,0,1])
    cyl_bottom = make_cylinder(base_center, base_normal, R, -5.0, "gray", alpha=0.3)
    ax.add_collection3d(cyl_bottom)
    theta = np.linspace(0, 2*np.pi, 200)
    xb = R*np.cos(theta); yb = R*np.sin(theta)
    ax.plot(xb, yb, np.zeros_like(theta), color="black", lw=2)          # 上圆
    ax.plot(xb, yb, np.zeros_like(theta)-5, color="black", lw=2)        # 下圆

    # ==== 上板圆柱 ====
    cyl_top = make_cylinder(Bo, Rbo[:,2], R, 5.0, "red", alpha=0.3)
    ax.add_collection3d(cyl_top)
    circle_local = np.vstack([R*np.cos(theta), R*np.sin(theta), np.zeros_like(theta)])
    circle_world_bot = (Rbo @ circle_local).T + Bo              # 下表面
    circle_world_top = circle_world_bot + Rbo[:,2]*5.0          # 上表面
    ax.plot(circle_world_bot[:,0], circle_world_bot[:,1], circle_world_bot[:,2],
            color="darkred", lw=2)
    ax.plot(circle_world_top[:,0], circle_world_top[:,1], circle_world_top[:,2],
            color="darkred", lw=2)

    # Bi → 顶板竖直杆（到上板下表面）
    for Bi in B:
        top_point = Bi + Rbo[:,2]*5.0
        ax.plot([Bi[0], top_point[0]], [Bi[1], top_point[1]], [Bi[2], top_point[2]],
                color="black", lw=3)

    # Bo 坐标轴
    L = 20.0
    axes_colors = ["#e41a1c", "#4daf4a", "#377eb8"]
    for vec, c in zip(Rbo.T, axes_colors):
        ax.quiver(Bo[0], Bo[1], Bo[2], vec[0]*L, vec[1]*L, vec[2]*L,
                  color=c, arrow_length_ratio=0.08)

    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_box_aspect([1,1,1])

    # 自动范围
    all_pts = np.vstack([A, M, B, Bo.reshape(1,3), circle_world_top, circle_world_bot])
    pad = 100.0
    mins = all_pts.min(axis=0) - pad
    maxs = all_pts.max(axis=0) + pad
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])

# ================= 主入口 =================
def main(mode="show"):
    # 你的工程默认参数
    side = 93.16
    axis_to_center_deg = -16.0
    h = 10.0
    L1 = 75.0
    axis_idx = 0  # 0=x, 1=y, 2=z

    model = CanfieldMirrorModel(side, axis_to_center_deg, h, L1, axis_idx)

    if mode == "calc":
        print("进入计算模式 (输入三个角度 [0~90]，或 -1 -1 -1 退出)：")
        while True:
            try:
                line = input("请输入 q1 q2 q3 (deg): ").strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 3:
                    print("⚠️ 需要输入三个数，例如: 30 45 60")
                    continue
                q = [float(x) for x in parts]

                # 退出
                if all(abs(val + 1) < 1e-9 for val in q):
                    print("退出计算模式。")
                    break

                # 范围
                if any(val < 0 or val > 90 for val in q):
                    print("⚠️ 超出范围！请输入 0~90 之间的角度。")
                    continue

                A, M, B, Bo, Rbo = model.forward(q)
                # 数值验证：三边是否等长；z 是否为三角形法向
                d12 = np.linalg.norm(B[1]-B[0])
                d23 = np.linalg.norm(B[2]-B[1])
                d31 = np.linalg.norm(B[0]-B[2])
                z_tri = np.cross(B[1]-B[0], B[2]-B[0])
                z_tri /= (np.linalg.norm(z_tri) + 1e-12)
                z_rbo = Rbo[:,2]
                alignment = np.abs(np.dot(z_tri, z_rbo))  # 1 表示同向或反向重合
                print(f"输入角度: {q}")
                print("Bo =", np.round(Bo, 3))
                print("Rbo =\n", np.round(Rbo, 3))
                print(f"|B1B2|={d12:.3f}, |B2B3|={d23:.3f}, |B3B1|={d31:.3f}  (理论应≈ {side:.3f})")
                print(f"z(三角形法向) 与 Rbo[:,2] 的对齐度 = {alignment:.6f}")
            except Exception as e:
                print("输入错误:", e)
                continue
    else:
        # 交互显示
        q_init = [45.0, 45.0, 45.0]
        plt.figure(figsize=(9, 8))
        ax = plt.axes([0.08, 0.22, 0.88, 0.75], projection="3d")
        A, M, B, Bo, Rbo = model.forward(q_init)
        draw_scene(ax, A, M, B, Bo, Rbo, side_expect=side)

        ax_q1 = plt.axes([0.10, 0.12, 0.80, 0.03])
        ax_q2 = plt.axes([0.10, 0.08, 0.80, 0.03])
        ax_q3 = plt.axes([0.10, 0.04, 0.80, 0.03])
        s_q1 = Slider(ax_q1, "q1 (deg)", 0.0, 90.0, valinit=q_init[0])
        s_q2 = Slider(ax_q2, "q2 (deg)", 0.0, 90.0, valinit=q_init[1])
        s_q3 = Slider(ax_q3, "q3 (deg)", 0.0, 90.0, valinit=q_init[2])

        def on_change(_):
            q = [s_q1.val, s_q2.val, s_q3.val]
            A, M, B, Bo, Rbo = model.forward(q)
            draw_scene(ax, A, M, B, Bo, Rbo, side_expect=side)
            plt.draw()

        s_q1.on_changed(on_change)
        s_q2.on_changed(on_change)
        s_q3.on_changed(on_change)
        plt.show()

if __name__ == "__main__":
    # "show" 打开滑块界面；"calc" 进入命令行循环
    main(mode="show")
