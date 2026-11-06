import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree

# ==================== KDTree 查表 IK ====================
class IKLookup:
    def __init__(self, npz_file, require_success=True):
        d = np.load(npz_file)
        pos = d["pos"]        # (N,3), mm
        quat = d["quat"]      # (N,4)  [x,y,z,w] 目前未直接用
        angles = d["angles"]  # (N,3), deg
        success = d["success"]
        res_norm = d["res_norm"]

        mask = np.isfinite(pos).all(axis=1) & np.isfinite(angles).all(axis=1)
        if require_success and "success" in d:
            mask &= (success > 0)

        self.pos = pos[mask]
        self.angles = angles[mask]
        self.quat = quat[mask]
        self.res_norm = res_norm[mask] if res_norm.shape[0] == mask.shape[0] else None

        if self.pos.size == 0:
            raise ValueError("IKLookup: no valid samples after filtering.")
        self.tree = cKDTree(self.pos)

    def query(self, target_xyz_mm, k=1):
        dist, idx = self.tree.query(np.asarray(target_xyz_mm, dtype=float), k=k)
        return self.angles[idx], self.pos[idx], self.quat[idx], dist

# ========== DH 工具 ==========
def dh_transform(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d],
        [0,   0,      0,     1]
    ])

def make_T0(i, side, h, axis_to_center_deg):
    R = side / np.sqrt(3.0)
    theta0 = np.deg2rad([0, 120, 240][i])
    beta = theta0 - np.deg2rad(axis_to_center_deg)
    xi, yi = R*np.cos(theta0), R*np.sin(theta0)
    T0 = np.array([
        [-np.sin(beta), 0,  np.cos(beta), xi],
        [ np.cos(beta), 0,  np.sin(beta), yi],
        [ 0,            1,  0,            h ],
        [ 0,            0,  0,            1 ]
    ])
    return T0

def chain_Ts_for_arm_manhattan(i, q1, q2, q3, q4,
                               side, axis_to_center_deg, link_params, T0, h=10.0):
    L1, L2, L3 = link_params
    A1  = dh_transform(q1, 0,      0,  +np.pi/2)
    A2d = dh_transform(q2, L2+L1,  0,  -np.pi/2)
    A2a = dh_transform(0,  0,      L3,  0)
    A3  = dh_transform(q3, 0,      L3, -np.pi/2)
    A4  = dh_transform(q4, L2+L1,  0,  +np.pi/2)

    T1   = T0 @ A1
    T2d  = T1 @ A2d
    T2a  = T2d @ A2a
    T3   = T2a @ A3
    T4   = T3 @ A4
    return T0, T1, T2d, T2a, T3, T4

# ========== 严格对齐 FK 模型 ==========
class CanfieldFKModel:
    def __init__(self, side=93.16, axis_to_center_deg=-16.0, h=10.0,
                 link_params=(39.5,34.1,5)):
        self.side = side
        self.axis_to_center_deg = axis_to_center_deg
        self.h = h
        self.link_params = link_params
        self.T0s = [make_T0(i, side, h, axis_to_center_deg) for i in range(3)]

    def forward(self, q_deg, q_pass_deg=None):
        """
        q_deg: (3,) 主动角 (deg)
        q_pass_deg: (6,) 或 (9,) 被动角 (deg)，可选。
                    如果不传，就直接返回 A/B 点位置。
        """
        q1s = np.deg2rad(q_deg)
        A_pts, B_pts = [], []
        Ts_all = []
        for i in range(3):
            q1 = q1s[i]
            if q_pass_deg is not None:
                q2, q3, q4 = np.deg2rad(q_pass_deg[3*i:3*i+3])
            else:
                q2 = q3 = q4 = 0.0
            T0, T1, T2d, T2a, T3, T4 = chain_Ts_for_arm_manhattan(
                i, q1, q2, q3, q4,
                self.side, self.axis_to_center_deg,
                self.link_params, self.T0s[i], self.h
            )
            A_pts.append(T0[:3,3]); B_pts.append(T4[:3,3])
            Ts_all.append((T0,T1,T2d,T2a,T3,T4))

        A = np.vstack(A_pts)
        B = np.vstack(B_pts)
        # 顶板中心
        Bo = B.mean(axis=0)
        # 顶板法向
        n_top = np.cross(B[1]-B[0], B[2]-B[0])
        n_top /= np.linalg.norm(n_top)+1e-12
        if n_top[2] < 0: n_top = -n_top
        # 局部坐标基
        x0 = B[0]-Bo
        if np.linalg.norm(x0)<1e-9: x0 = np.array([1,0,0])
        x0 /= np.linalg.norm(x0)
        y0 = np.cross(n_top, x0); y0 /= np.linalg.norm(y0)
        z0 = np.cross(x0, y0)
        Rbo = np.column_stack([x0,y0,z0])
        return A, B, Bo, Rbo, Ts_all

# ==================== 画上/下圆盘、A/M/B、轴等（与你原函数一致） ====================
def make_cylinder(center, normal, radius, height, color, alpha=0.2, n=60):
    center = np.asarray(center, dtype=float)
    normal = np.asarray(normal, dtype=float)
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    if abs(normal[0]) < 0.9:
        tangent = np.cross(normal, [1,0,0])
    else:
        tangent = np.cross(normal, [0,1,0])
    tangent /= np.linalg.norm(tangent) + 1e-12
    bitangent = np.cross(normal, tangent)
    c1 = center
    c2 = center + normal * height
    theta = np.linspace(0, 2*np.pi, n)
    circle1 = np.array([c1 + radius*(np.cos(t)*tangent + np.sin(t)*bitangent) for t in theta])
    circle2 = np.array([c2 + radius*(np.cos(t)*tangent + np.sin(t)*bitangent) for t in theta])
    verts = []
    for i in range(n):
        verts.append([circle1[i], circle1[(i+1)%n], circle2[(i+1)%n], circle2[i]])
    verts.append(circle1)
    verts.append(circle2[::-1])
    poly = Poly3DCollection(verts, facecolors=color, edgecolors='none', alpha=alpha)
    return poly

def draw_scene(ax, A, B, Bo, Rbo, side_expect=None):
    ax.cla()
    ax.scatter(*A.T, c="#1f77b4", s=30, label="A (base)")
    ax.scatter(*B.T, c="#d62728", s=30, label="B (upper)")
    ax.scatter(*Bo,   c="k", s=50, label="Bo (center)")
    
    for Ai, Bi in zip(A, B):
        ax.plot([Ai[0], Bi[0]], [Ai[1], Bi[1]], [Ai[2], Bi[2]],
                color="#ff9800", lw=2)
        ax.plot([Ai[0], Ai[0]], [Ai[1], Ai[1]], [Ai[2], 0],
                color="black", lw=3)

    R = 70.0
    base_center = np.array([0,0,0]); base_normal = np.array([0,0,1])
    cyl_bottom = make_cylinder(base_center, base_normal, R, -5.0, "gray", alpha=0.3)
    ax.add_collection3d(cyl_bottom)
    theta = np.linspace(0, 2*np.pi, 200)
    xb = R*np.cos(theta); yb = R*np.sin(theta)
    ax.plot(xb, yb, np.zeros_like(theta), color="black", lw=2)
    ax.plot(xb, yb, np.zeros_like(theta)-5, color="black", lw=2)

    cyl_top = make_cylinder(Bo + Rbo[:,2] * 5.0, Rbo[:,2], R, 5.0, "red", alpha=0.3)
    ax.add_collection3d(cyl_top)
    circle_local = np.vstack([R*np.cos(theta), R*np.sin(theta), np.zeros_like(theta)])
    circle_world_bot = (Rbo @ circle_local).T + Bo  + Rbo[:,2] * 5.0
    circle_world_top = circle_world_bot + Rbo[:,2]*5.0
    ax.plot(circle_world_bot[:,0], circle_world_bot[:,1], circle_world_bot[:,2], color="darkred", lw=2)
    ax.plot(circle_world_top[:,0],  circle_world_top[:,1],  circle_world_top[:,2],  color="darkred", lw=2)

    for Bi in B:
        top_point = Bi + Rbo[:,2]*5.0
        ax.plot([Bi[0], top_point[0]], [Bi[1], top_point[1]], [Bi[2], top_point[2]], color="black", lw=3)

    L = 20.0
    axes_colors = ["#e41a1c", "#4daf4a", "#377eb8"]
    for vec, c in zip(Rbo.T, axes_colors):
        ax.quiver(Bo[0], Bo[1], Bo[2], vec[0]*L, vec[1]*L, vec[2]*L, color=c, arrow_length_ratio=0.08)

    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_box_aspect([1,1,1])
    ax.set_xlim(-100, 100); ax.set_ylim(-100, 100); ax.set_zlim(0, 200)

# ==================== 主程序：查表 IK + 你的交互 UI ====================
def main(kind="ik", mode="show"):
    # --- 文件路径 ---
    ik_npz_path = "canfield_ik_0p4deg_lookup.npz"

    # --- 机构参数（保持你的风格） ---
    side=93.16; axis_to_center_deg=-16.0; h=10.0; L1=75.0; axis_idx=0
    fk_model = CanfieldFKModel(side, axis_to_center_deg, h, link_params=(39.5,34.1,5))
    ik_lookup = IKLookup(ik_npz_path, require_success=True)

    if kind=="ik" and mode=="show":
        Bo_init=[0,0,45]
        max_err_mm = 15.0  # 允许的最近邻距离阈值：超出判定为无效

        fig = plt.figure(figsize=(10,8))
        ax3d = plt.axes([0.05,0.3,0.6,0.65], projection="3d")
        ax2d = plt.axes([0.7,0.3,0.25,0.25])   # xy 控制板
        ax_z  = plt.axes([0.1,0.1,0.8,0.03])   # z 滑块

        # 背景圆
        Rmax = 70
        circle = plt.Circle((0,0), Rmax, color="lightgray", alpha=0.3)
        ax2d.add_patch(circle)
        ax2d.axhline(0, color="k", lw=1, ls="--")
        ax2d.axvline(0, color="k", lw=1, ls="--")
        dot, = ax2d.plot([Bo_init[0]], [Bo_init[1]], "ro")
        ax2d.set_xlim(-Rmax, Rmax); ax2d.set_ylim(-Rmax, Rmax)
        ax2d.set_aspect("equal"); ax2d.set_title("XY Control")

        s_z = Slider(ax_z, "Bo_z", 10, 160, valinit=Bo_init[2])

        Bo_current = np.array(Bo_init, dtype=float)
        dragging = False   # 拖动优先标志

        # ==== 警告文字 ====
        warning_text = ax3d.text2D(
            0.35, 0.90, "Invalid Move", fontsize=16, color="red",
            transform=ax3d.transAxes
        )
        warning_text.set_visible(False)

        def set_warning_visible(flag: bool):
            warning_text.set_visible(bool(flag))
            fig.canvas.draw_idle()

        # ==== 控制状态 ====
        key_states = {"up":False,"down":False,"left":False,"right":False,
                      "z":False,"x":False}
        speed_xy = 1.0   # XY 步长
        speed_z  = 2.0   # Z 步长

        # ==== 6 个操作按钮（可视化 UI）====
        offset_x = 0.70; offset_y = 0.75; dx = 0.06; dy = 0.08
        button_defs = {
            "left":  {"label": "X←", "pos": (offset_x,       offset_y)},
            "right": {"label": "X→", "pos": (offset_x+2*dx, offset_y)},
            "up":    {"label": "Y↑", "pos": (offset_x+dx,   offset_y+dy)},
            "down":  {"label": "Y↓", "pos": (offset_x+dx,   offset_y-dy)},
            "z":     {"label": "Z↑", "pos": (offset_x+3*dx, offset_y+dy)},
            "x":     {"label": "Z↓", "pos": (offset_x+3*dx, offset_y-dy)},
        }
        button_patches = {}
        button_texts   = {}
        for key,info in button_defs.items():
            rect = mpatches.Rectangle(
                (info["pos"][0]-0.03, info["pos"][1]-0.025), 0.06, 0.05,
                transform=fig.transFigure,
                facecolor="lightgray", edgecolor="black"
            )
            fig.patches.append(rect)
            txt = fig.text(
                info["pos"][0], info["pos"][1],
                info["label"],
                ha="center", va="center",
                fontsize=10, color="black"
            )
            button_patches[key] = rect
            button_texts[key]   = txt

        def set_button_color(key, active):
            button_patches[key].set_facecolor("red" if active else "lightgray")
            button_texts[key].set_color("white" if active else "black")
            fig.canvas.draw_idle()

        # ===== 绘制（用 KDTree → q_deg → forward → draw_scene）=====
        def draw_from_Bo():
            ax3d.cla()
            q_deg, pos_nn, quat_nn, dist = ik_lookup.query(Bo_current, k=1)
            pos_nn = np.asarray(pos_nn).reshape(-1)   # (3,)

            A, B, Bo_fk, Rbo, Ts_all = fk_model.forward(q_deg)
            Bo_draw = pos_nn
            draw_scene(ax3d, A, B, Bo_draw, Rbo)

            ax3d.set_title(
                f"target=({Bo_current[0]:.1f}, {Bo_current[1]:.1f}, {Bo_current[2]:.1f}) | "
                f"nn=({pos_nn[0]:.1f}, {pos_nn[1]:.1f}, {pos_nn[2]:.1f}) | "
                f"fk=({Bo_fk[0]:.1f}, {Bo_fk[1]:.1f}, {Bo_fk[2]:.1f}) | "
                f"err={float(dist):.2f} mm"
            )
            set_warning_visible(False)
            fig.canvas.draw_idle()

        # ===== 鼠标拖动事件 =====
        def on_press(event):
            nonlocal dragging, Bo_current
            if event.inaxes is not ax2d: return
            x,y = event.xdata, event.ydata
            if np.hypot(x,y) > Rmax: return
            dragging = True
            Bo_current[0], Bo_current[1] = x, y
            dot.set_data([x],[y])
            draw_from_Bo()
            fig.canvas.draw_idle()

        def on_motion(event):
            nonlocal Bo_current
            if not dragging: return
            if event.inaxes is not ax2d: return
            x,y = event.xdata, event.ydata
            if np.hypot(x,y) > Rmax: return
            Bo_current[0], Bo_current[1] = x, y
            dot.set_data([x],[y])
            draw_from_Bo()
            fig.canvas.draw_idle()

        def on_release(event):
            nonlocal dragging
            dragging = False

        def update_z(val):
            Bo_current[2] = float(val)
            draw_from_Bo()
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("button_press_event", on_press)
        fig.canvas.mpl_connect("motion_notify_event", on_motion)
        fig.canvas.mpl_connect("button_release_event", on_release)
        s_z.on_changed(update_z)

        # ===== 键盘事件 =====
        def on_key(event):
            k = (event.key or "").lower()
            if k in key_states:
                key_states[k] = True
                set_button_color(k, True)

        def on_key_release(event):
            k = (event.key or "").lower()
            if k in key_states:
                key_states[k] = False
                set_button_color(k, False)

        fig.canvas.mpl_connect("key_press_event", on_key)
        fig.canvas.mpl_connect("key_release_event", on_key_release)

        # ===== 定时器：周期性更新 =====
        def step_update(*args):
            nonlocal Bo_current, dragging
            if dragging:
                return   # 拖动时暂停键盘控制

            dx = dy = dz = 0.0
            if key_states["up"] and not key_states["down"]:    dy += speed_xy
            if key_states["down"] and not key_states["up"]:   dy -= speed_xy
            if key_states["left"] and not key_states["right"]:dx -= speed_xy
            if key_states["right"] and not key_states["left"]:dx += speed_xy
            if key_states["z"] and not key_states["x"]:       dz += speed_z
            if key_states["x"] and not key_states["z"]:       dz -= speed_z

            if dx!=0 or dy!=0 or dz!=0:
                candidate = Bo_current + np.array([dx,dy,dz])
                q_deg, pos_nn, quat_nn, dist = ik_lookup.query(candidate, k=1)
                err = float(dist if np.isscalar(dist) else dist[0])
                if err <= max_err_mm and np.all(np.isfinite(q_deg)):
                    Bo_current = candidate
                    dot.set_data([Bo_current[0]],[Bo_current[1]])
                    s_z.set_val(Bo_current[2])  # 更新 slider
                    draw_from_Bo()
                else:
                    set_warning_visible(True)
                    ax3d.set_title(f"Invalid Move (err={err:.1f} mm)", color="red")
                    fig.canvas.draw_idle()

        timer = fig.canvas.new_timer(interval=50)
        timer.add_callback(step_update)
        timer.start()

        draw_from_Bo()
        plt.show()

if __name__=="__main__":
    main(kind="ik", mode="show")
