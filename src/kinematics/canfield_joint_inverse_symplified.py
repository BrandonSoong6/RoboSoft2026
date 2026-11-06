import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ================= 通用工具 =================
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
    z = z / (np.linalg.norm(z) + 1e-12)
    x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    if np.linalg.norm(y) < 1e-12:
        x = np.array([1.0, 0.0, 0.0])
        y = np.cross(z, x)
    y /= (np.linalg.norm(y) + 1e-12)
    x = np.cross(y, z); x /= (np.linalg.norm(x) + 1e-12)
    return x, y, z

# ================= Forward =================
class CanfieldMirrorModel:
    def __init__(self, side=93.16, axis_to_center_deg=-16.0, h=10.0,
                 L1=75.0, axis_idx=0):
        self.side = side; self.axis_to_center_deg = axis_to_center_deg
        self.h = h; self.L1 = L1; self.axis_idx = axis_idx
        self.T0s = _make_T0s(side, h, axis_to_center_deg)

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

        n = np.cross(M[1]-M[0], M[2]-M[1])
        n = np.array([0,0,1]) if np.linalg.norm(n)<1e-12 else n/np.linalg.norm(n)
        C = M.mean(axis=0)
        B = np.array([reflect_point_across_plane(A[i], C, n) for i in range(3)])
        Bo0 = B.mean(axis=0)
        z = np.cross(B[1]-B[0], B[2]-B[0])
        z = np.array([0,0,1]) if np.linalg.norm(z)<1e-12 else z/np.linalg.norm(z)
        x0 = B[0] - Bo0
        if np.linalg.norm(x0)<1e-12: x0 = np.array([1,0,0])
        x,y,z = orthonormalize_keep_z(x0, z)
        Rbo = np.column_stack([x,y,z])
        Bo = Bo0
        return A, M, B, Bo, Rbo

# ================= Inverse =================
class CanfieldMirrorIK:
    def __init__(self, side=93.16, axis_to_center_deg=-16.0, h=10.0,
                 L1=75.0, axis_idx=0):
        self.side = side; self.axis_to_center_deg = axis_to_center_deg
        self.h = h; self.L1 = L1; self.axis_idx = axis_idx
        self.T0s = _make_T0s(side, h, axis_to_center_deg)
        self.Ai = np.stack([self.T0s[i,:3,3] for i in range(3)], axis=0)
        self.R0 = np.stack([self.T0s[i,:3,:3] for i in range(3)], axis=0)
        self.A0 = np.array([0,0,h], dtype=float)

    @staticmethod
    def _wrap_to_0_2pi(x):
        x = np.fmod(x, 2*np.pi)
        if x < 0: x += 2*np.pi
        return x

    def solve_q_for_branch(self,i,Bo,prefer_range=(0,np.deg2rad(90))):
        eps=1e-12
        Bo=np.asarray(Bo).reshape(3,)
        C=0.5*(self.A0+Bo)
        n=Bo-self.A0; n/=np.linalg.norm(n)
        Ai=self.Ai[i]; R0=self.R0[i]
        w=R0.T@n
        a,b=w[0],w[1]; r=np.hypot(a,b)
        s=np.dot(C-Ai,n)/(self.L1+eps)
        if r<eps: return 0.25*np.pi,True
        phi=np.arctan2(b,a); 
        val = s / r
        if abs(val) > 1.0:
            return 0.0, True   # 无解，触发 warning
        ang = np.arccos(val)
        q1 = self._wrap_to_0_2pi(phi + ang)
        q2 = self._wrap_to_0_2pi(phi - ang)
        lo,hi=prefer_range
        in_range=[q for q in (q1,q2) if lo<q<hi]
        if in_range: return in_range[0],False
        return q1,True

    def solve(self,Bo):
        qs=[];flags=[]
        for i in range(3):
            q,w=self.solve_q_for_branch(i,Bo)
            qs.append(q);flags.append(w)
        return np.rad2deg(np.array(qs)),{"warnings":flags}

def make_cylinder(center, normal, radius, height, color, alpha=0.2, n=60):
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    if abs(normal[0]) < 0.9:
        tangent = np.cross(normal, [1,0,0])
    else:
        tangent = np.cross(normal, [0,1,0])
    tangent /= np.linalg.norm(tangent)
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

def draw_scene(ax, A, M, B, Bo, Rbo, side_expect=None):
    ax.cla()
    ax.scatter(*A.T, c="#1f77b4", s=30, label="A (base)")
    ax.scatter(*M.T, c="#2ca02c", s=30, label="M (lower-link tips)")
    ax.scatter(*B.T, c="#d62728", s=30, label="B (reflected)")
    ax.scatter(*Bo,   c="k", s=50, label="Bo (upper plate center)")
    for Ai, Mi, Bi in zip(A, M, B):
        ax.plot([Ai[0], Mi[0]], [Ai[1], Mi[1]], [Ai[2], Mi[2]], color="#9e9e9e", lw=2)
        ax.plot([Mi[0], Bi[0]], [Mi[1], Bi[1]], [Mi[2], Bi[2]], color="#ff9800", lw=2)
        ax.plot([Ai[0], Ai[0]], [Ai[1], Ai[1]], [Ai[2], 0], color="black", lw=3)
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

import matplotlib.patches as mpatches

def main(kind="ik", mode="show"):
    side=93.16; axis_to_center_deg=-16.0; h=10.0; L1=75.0; axis_idx=0
    fk_model = CanfieldMirrorModel(side,axis_to_center_deg,h,L1,axis_idx)
    ik_model = CanfieldMirrorIK(side,axis_to_center_deg,h,L1,axis_idx)

    if kind=="ik" and mode=="show":
        Bo_init=[0,0,45]

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
        # === 全局调节参数 ===
        offset_x = 0.70   # 整体水平基准
        offset_y = 0.75   # 整体竖直基准
        dx = 0.06         # 按钮横向间距
        dy = 0.08         # 按钮纵向间距

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
            if active:
                button_patches[key].set_facecolor("red")
            else:
                button_patches[key].set_facecolor("lightgray")
            fig.canvas.draw_idle()

        def draw_from_Bo():
            ax3d.cla()
            q_deg,info = ik_model.solve(Bo_current)
            if any(info["warnings"]):
                set_warning_visible(True)
                ax3d.set_title("Invalid Move", color="red")
                fig.canvas.draw_idle()
                return
            A,M,B,Bo_calc,Rbo = fk_model.forward(q_deg)
            draw_scene(ax3d,A,M,B,Bo_calc,Rbo,side_expect=side)
            ax3d.set_title(
                f"q=({q_deg[0]:6.1f}, {q_deg[1]:6.1f}, {q_deg[2]:6.1f})   |   "
                f"Bo=({Bo_calc[0]:7.1f}, {Bo_calc[1]:7.1f}, {Bo_calc[2]:7.1f})"
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
            Bo_current[2] = s_z.val
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
                button_patches[k].set_facecolor("red")
                button_texts[k].set_color("white")
                fig.canvas.draw_idle()

        def on_key_release(event):
            k = (event.key or "").lower()
            if k in key_states:
                key_states[k] = False
                button_patches[k].set_facecolor("lightgray")
                button_texts[k].set_color("black")
                fig.canvas.draw_idle()

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
                q_deg,info = ik_model.solve(candidate)
                if not any(info["warnings"]):
                    Bo_current = candidate
                    dot.set_data([Bo_current[0]],[Bo_current[1]])
                    s_z.set_val(Bo_current[2])  # 更新 slider
                    draw_from_Bo()
                else:
                    set_warning_visible(True)
                    ax3d.set_title("Invalid Move", color="red")
                    fig.canvas.draw_idle()

        timer = fig.canvas.new_timer(interval=50)
        timer.add_callback(step_update)
        timer.start()

        draw_from_Bo()
        plt.show()

if __name__=="__main__":
    main(kind="ik", mode="show")
