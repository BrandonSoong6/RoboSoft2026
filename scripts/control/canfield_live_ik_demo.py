import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import serial
import threading
import time
import struct

comm = None

# ================= 通用工具 =================
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

def signed_angle_in_plane(u, v, n):
    u_p = u - np.dot(u, n) * n
    v_p = v - np.dot(v, n) * n
    nu = np.linalg.norm(u_p); nv = np.linalg.norm(v_p)
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    u_p /= nu; v_p /= nv
    s = np.dot(n, np.cross(u_p, v_p))
    c = np.dot(u_p, v_p)
    return np.arctan2(s, c)

def wrap_pi(x):
    return (x + np.pi) % (2*np.pi) - np.pi

def ang_err_mod_pi(angle, target):
    e = wrap_pi(angle - target)
    if e > np.pi/2: e -= np.pi
    elif e < -np.pi/2: e += np.pi
    return e

# ---------- 画图小工具 ----------
def _draw_axis_arrow(ax, o, v, color, label, length=8.0, fsize=7):
    n = np.linalg.norm(v) + 1e-12
    v = v / n
    ax.quiver(o[0], o[1], o[2], v[0], v[1], v[2],
              length=length, normalize=True, color=color)
    tip = o + v * length
    ax.text(tip[0], tip[1], tip[2], label, fontsize=fsize, color=color)

def _draw_line_with_label(ax, p, q, color, label, alpha=0.8, fsize=7):
    ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]],
            color=color, alpha=alpha, linewidth=1.5)
    mid = 0.5*(p + q)
    ax.text(mid[0], mid[1], mid[2], label, fontsize=fsize, color=color)

# =============================
# 链：曼哈顿拆分
# =============================
def chain_Ts_for_arm_manhattan(i, q1, q2, q3, q4,
                               side, axis_to_center_deg, link_params, h=10.0):
    L1, L2, L3 = link_params
    T0 = make_T0(i, side, h, axis_to_center_deg)
    A1 = dh_transform(q1, 0, 0, +np.pi/2)            # a1 = 0
    A2d = dh_transform(q2, L2 + L1, 0, -np.pi/2)     # d2 = L2 + L1
    A2a = dh_transform(0, 0, L3, 0)
    A3  = dh_transform(q3, 0,     L3, -np.pi/2)
    A4  = dh_transform(q4, L2+L1, 0,  +np.pi/2)

    T1   = T0 @ A1
    T2d  = T1 @ A2d
    T2a  = T2d @ A2a
    T3   = T2a @ A3
    T4   = T3 @ A4
    return T0, T1, T2d, T2a, T3, T4


# ================= Forward =================
class CanfieldDH_FK:
    """
    Canfield Joint 正运动学查表模型
    --------------------------------
    - 支持从 npz 查表快速插值 FK 结果
    - 返回 A, M, B, Bo, Rbo 及所有被动角信息
    """
    def __init__(self, fk_npz_path):
        """
        参数
        ----
        fk_npz_path : str
            由 csv_to_npz_fast() 生成的 .npz 查表文件
        """
        d = np.load(fk_npz_path, allow_pickle=True)
        self.arr = d["arr"]
        self.q1_range = d["q1_range"]
        self.q2_range = d["q2_range"]
        self.q3_range = d["q3_range"]
        self.step = float(d["step"])
        self.features = [s for s in d["features"]]
        self.shape = self.arr.shape
        print(f"[FK] Loaded {fk_npz_path}, shape={self.shape}")

    def _idx_frac(self, qv, q_range):
        """计算查表的索引与插值权重"""
        i = (qv - q_range[0]) / self.step
        j = int(np.floor(i))
        t = i - j
        j = np.clip(j, 0, len(q_range) - 2)
        return j, t

    def _trilinear_interp(self, vals, t1, t2, t3):
        """三线性插值"""
        return (
            vals[0,0,0]*(1-t1)*(1-t2)*(1-t3) +
            vals[1,0,0]*t1*(1-t2)*(1-t3) +
            vals[0,1,0]*(1-t1)*t2*(1-t3) +
            vals[0,0,1]*(1-t1)*(1-t2)*t3 +
            vals[1,1,0]*t1*t2*(1-t3) +
            vals[1,0,1]*t1*(1-t2)*t3 +
            vals[0,1,1]*(1-t1)*t2*t3 +
            vals[1,1,1]*t1*t2*t3
        )

    def forward(self, q_deg):
        """
        输入: q_deg = [q1,q2,q3] (度)
        输出: (A, M, B, Bo, Rbo, angles_dict)
        其中 angles_dict 含 q2_*, q3_*，方便绘制三支链。
        """
        q1, q2, q3 = q_deg
        i1, t1 = self._idx_frac(q1, self.q1_range)
        i2, t2 = self._idx_frac(q2, self.q2_range)
        i3, t3 = self._idx_frac(q3, self.q3_range)

        # 局部 2x2x2 立方体
        vals = self.arr[i1:i1+2, i2:i2+2, i3:i3+2, :]

        # 各字段索引（按生成顺序）
        # "q2_1_deg","q3_1_deg","q2_2_deg","q3_2_deg","q2_3_deg","q3_3_deg",
        # "eul_z_deg","eul_y_deg","eul_x_deg","Cx_mm","Cy_mm","Cz_mm"
        interp = lambda idx: self._trilinear_interp(vals[..., idx], t1, t2, t3)
        q2_1 = interp(0); q3_1 = interp(1)
        q2_2 = interp(2); q3_2 = interp(3)
        q2_3 = interp(4); q3_3 = interp(5)
        eul_z = interp(6); eul_y = interp(7); eul_x = interp(8)
        Cx = interp(9); Cy = interp(10); Cz = interp(11)

        # 平均姿态与位置
        Bo = np.array([Cx, Cy, Cz], dtype=float)
        Rbo = R.from_euler("zyx", [eul_z, eul_y, eul_x], degrees=True).as_matrix()

        # === 返回结构 ===
        # 因为表格没记录 A/M/B 三点坐标，只能留空或后续计算
        # 若你有固定几何，可后补生成（暂用 None）
        A = np.zeros((3,3))
        M = np.zeros((3,3))
        B = np.zeros((3,3))

        angles = {
            "q2_1": q2_1, "q3_1": q3_1,
            "q2_2": q2_2, "q3_2": q3_2,
            "q2_3": q2_3, "q3_3": q3_3,
            "eul_z": eul_z, "eul_y": eul_y, "eul_x": eul_x
        }

        return A, M, B, Bo, Rbo, angles
    
# ================= Inverse =================
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

class CanfieldDH_IK:
    """
    查表式 Canfield Joint 反解求解器
    ---------------------------------
    - 保持旧接口兼容（solve / solve_q_for_branch）
    - 内部基于查表 (npz) + KDTree 最近邻搜索
    - 支持同伦门控 (homotopy gating) + IDW 平滑插值
    """
    def __init__(self, ik_npz_path, prefer_deg=(90, 160), K=32, tau0=3.0, tau_step=0.4, tau_max=10.0):
        """
        参数
        ----
        ik_npz_path : str
            查表文件路径 (.npz)，需包含 pos, quat, angles 三个键
        prefer_deg : tuple(float, float)
            角度上下限（度），用于 clip 防止舵机越界
        K : int
            KNN 近邻数量
        tau0, tau_step, tau_max : float
            同伦门控参数
        """
        self.ik_npz_path = ik_npz_path
        self.q_min, self.q_max = prefer_deg
        self.K = K
        self.tau0 = tau0
        self.tau_step = tau_step
        self.tau_max = tau_max

        # === 载入 IK 数据 ===
        d = np.load(ik_npz_path, allow_pickle=True)
        self.pos_table = d["pos"].astype(np.float64)
        self.quat_table = d["quat"].astype(np.float64)
        self.angles_table = d["angles"].astype(np.float64)
        self.tree = cKDTree(self.pos_table)

    @staticmethod
    def _wrap_to_0_2pi(x):
        x = np.fmod(x, 2 * np.pi)
        if x < 0:
            x += 2 * np.pi
        return x

    def solve(self, Bo, q_curr=None):
        """
        对给定末端点 Bo（mm），计算舵机角度解。
        返回角度数组 (deg)，以及附带状态信息。
        """
        if q_curr is None:
            q_curr = np.array([120.0, 120.0, 120.0])

        q_sol, pos_nn, quat_nn, info = self.ik_with_homotopy_idw(
            pos_target=Bo, q_curr=q_curr
        )

        pos_err = np.linalg.norm(pos_nn - Bo)
        info["pos_err"] = pos_err

        flags = [info["mode"] == "nn_fallback"] * 3
        return np.array(q_sol), {"warnings": flags, **info}

    def ik_with_homotopy_idw(self, pos_target, q_curr,
                             sigma_p=1.0, eps=1e-6, m=1):
        """
        核心求解算法：
        - 最近邻查表 (KDTree)
        - 同伦门控
        - IDW 平滑插值
        """
        dist, idxs = self.tree.query(pos_target, k=self.K)
        idxs = np.atleast_1d(idxs)
        tau = self.tau0
        dq_inf_all = np.max(np.abs(self.angles_table[idxs] - q_curr), axis=1)

        while True:
            mask = dq_inf_all <= tau
            if np.any(mask):
                sel = idxs[mask]
                break
            tau += self.tau_step
            if tau > self.tau_max:
                sel = idxs
                break

        pos_sel = self.pos_table[sel]
        q_sel = self.angles_table[sel]

        d_task = np.linalg.norm((pos_sel - pos_target) / sigma_p, axis=1)
        if len(d_task) > m:
            keep = np.argsort(d_task)[:m]
            pos_sel = pos_sel[keep]
            q_sel = q_sel[keep]
            d_task = d_task[keep]

        w = 1.0 / (d_task + eps) ** 2
        w_sum = np.sum(w)
        if w_sum < 1e-12:
            best_idx = sel[np.argmin(d_task)]
            return (
                self.angles_table[best_idx],
                self.pos_table[best_idx],
                self.quat_table[best_idx],
                {"tau_used": tau, "n_sel": len(sel), "mode": "nn_fallback"},
            )

        q_blend = (w[:, None] * q_sel).sum(axis=0) / w_sum
        q_blend = np.clip(q_blend, self.q_min, self.q_max)

        best_idx = sel[np.argmin(d_task)]
        return (
            q_blend,
            self.pos_table[best_idx],
            self.quat_table[best_idx],
            {"tau_used": tau, "n_sel": len(sel), "mode": "idw_blend"},
        )

def chain_Ts_for_arm_manhattan(i, q1, q2, q3, q4,
                               side, axis_to_center_deg, link_params, h=10.0):
    L1, L2, L3 = link_params
    T0 = make_T0(i, side, h, axis_to_center_deg)
    A1 = dh_transform(q1, 0, 0, +np.pi/2)            # a1 = 0
    A2d = dh_transform(q2, L2 + L1, 0, -np.pi/2)     # d2 = L2 + L1
    A2a = dh_transform(0, 0, L3, 0)
    A3  = dh_transform(q3, 0,     L3, -np.pi/2)
    A4  = dh_transform(q4, L2+L1, 0,  +np.pi/2)

    T1   = T0 @ A1
    T2d  = T1 @ A2d
    T2a  = T2d @ A2a
    T3   = T2a @ A3
    T4   = T3 @ A4
    return T0, T1, T2d, T2a, T3, T4

def _make_disk_vertices(center, normal, radius, height, n=100):
    """生成带厚度的圆盘顶点（带外侧描边可见感）"""
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    # 构造局部坐标系
    if abs(normal[0]) < 0.9:
        tangent = np.cross(normal, [1, 0, 0])
    else:
        tangent = np.cross(normal, [0, 1, 0])
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    bitangent /= np.linalg.norm(bitangent)

    # 上下两层圆
    c1 = center - normal * height/2
    c2 = center + normal * height/2
    theta = np.linspace(0, 2*np.pi, n)
    circle1 = np.array([c1 + radius*(np.cos(t)*tangent + np.sin(t)*bitangent) for t in theta])
    circle2 = np.array([c2 + radius*(np.cos(t)*tangent + np.sin(t)*bitangent) for t in theta])

    # 外壳侧面 + 顶/底面
    verts = []
    for i in range(n):
        verts.append([circle1[i], circle1[(i+1)%n], circle2[(i+1)%n], circle2[i]])
    verts.append(circle1)
    verts.append(circle2[::-1])
    return verts, circle1, circle2


def plot_from_lookup(row, ax,
                     side=93.16, axis_to_center_deg=-16.0,
                     link_params=(39.5,34.1,5), h=10.0):
    """高保真渲染版：带厚度、描边、放大圆盘"""
    q_actives_deg = [row["q1_deg"], row["q2_deg"], row["q3_deg"]]
    q1s = np.deg2rad(q_actives_deg)

    # 被动角
    q_pass = []
    for k in range(1, 4):
        q2 = row[f"q2_{k}_deg"]
        q3 = row[f"q3_{k}_deg"]
        q4 = -q2
        q_pass.extend([np.deg2rad(q2), np.deg2rad(q3), np.deg2rad(q4)])
    q_pass = np.array(q_pass)

    # 彩色连杆（柔和工业风）
    colors = {
        "link1":"#546E7A",  # 深灰蓝
        "link2":"#1E88E5",  # 蓝
        "link3":"#FBC02D",  # 金黄
        "link4":"#8E24AA"   # 紫
    }

    A_pts, B_pts = [], []
    R_base = side / np.sqrt(3.0)
    R_disk = R_base * 1.5   # 圆盘放大系数

    # === 绘制三支链 ===
    for i in range(3):
        q1 = q1s[i]
        q2, q3, q4 = q_pass[3*i:3*i+3]
        T0,T1,T2d,T2a,T3,T4 = chain_Ts_for_arm_manhattan(
            i, q1, q2, q3, q4, side, axis_to_center_deg, link_params, h
        )
        A_pts.append(T0[:3,3])
        B_pts.append(T4[:3,3])

        ROD_SCALE = 1.7

        # 连杆（包含垂直段）
        for (p, q, col, lw) in [
            (T0[:3,3], T1[:3,3], colors["link1"], 3.0 * ROD_SCALE),   # 主动
            (T1[:3,3], T2d[:3,3], colors["link2"], 2.8 * ROD_SCALE),  # 垂直
            (T2d[:3,3], T2a[:3,3], colors["link2"], 2.8 * ROD_SCALE), # 延伸
            (T2a[:3,3], T3[:3,3], colors["link3"], 2.8 * ROD_SCALE),  # 上臂
            (T3[:3,3], T4[:3,3], colors["link4"], 3.0 * ROD_SCALE),   # 拉杆
        ]:
            ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color=col, lw=lw, alpha=0.95)

    # === 上下圆盘 ===
    A_loop = np.vstack(A_pts)
    B_loop = np.vstack(B_pts)
    base_center = np.mean(A_loop, axis=0)
    base_normal = np.cross(A_loop[1]-A_loop[0], A_loop[2]-A_loop[1])
    base_normal /= np.linalg.norm(base_normal)

    top_center = np.mean(B_loop, axis=0)
    top_normal = np.cross(B_loop[1]-B_loop[0], B_loop[2]-B_loop[1])
    top_normal /= np.linalg.norm(top_normal)
    if top_normal[2] < 0:
        top_normal = -top_normal

    verts_bottom, circ1_bot, circ2_bot = _make_disk_vertices(base_center, base_normal, R_disk, height=3.0)
    verts_top, circ1_top, circ2_top = _make_disk_vertices(top_center, top_normal, R_disk, height=3.0)

    # 填充与描边
    ax.add_collection3d(Poly3DCollection(
        verts_bottom, facecolors=(0.6,0.6,0.6,0.3),
        edgecolors=(0.3,0.3,0.3,0.8), linewidths=0.5))
    ax.add_collection3d(Poly3DCollection(
        verts_top, facecolors=(1,0.2,0.2,0.25),
        edgecolors=(0.5,0,0,0.7), linewidths=0.8))

    # 边缘圈线（更明显轮廓）
    ax.plot(circ1_bot[:,0], circ1_bot[:,1], circ1_bot[:,2], color="black", lw=1.0, alpha=0.5)
    ax.plot(circ2_bot[:,0], circ2_bot[:,1], circ2_bot[:,2], color="black", lw=1.0, alpha=0.5)
    ax.plot(circ1_top[:,0], circ1_top[:,1], circ1_top[:,2], color="darkred", lw=1.0, alpha=0.6)
    ax.plot(circ2_top[:,0], circ2_top[:,1], circ2_top[:,2], color="darkred", lw=1.0, alpha=0.6)

    # === 顶板姿态箭头 ===
    C = np.mean(B_loop, axis=0)
    ax.quiver(C[0], C[1], C[2], top_normal[0]*20, top_normal[1]*20, top_normal[2]*20,
              color='blue', arrow_length_ratio=0.12)

    # === 视觉参数 ===
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_zlim(0, 200)
    ax.set_box_aspect([1,1,1])
    ax.set_facecolor((0.94, 0.95, 0.97))
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    ax.set_title("Canfield Joint — 圆盘描边渲染版 (R×1.5, 厚3mm)")

def main(kind="ik", mode="show"):
    side=93.16; axis_to_center_deg=-16.0; h=10.0; link_params=(39.5,34.1,5); axis_idx=0
    fk_model = CanfieldDH_FK("canfield_lookup_pose_0p4deg_full.npz")
    ik_model = CanfieldDH_IK("canfield_ik_0p4deg_lookup.npz")
    prev_q = None

    if kind=="ik" and mode=="show":
        Bo_init=[0,0,100]

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

        s_z = Slider(ax_z, "Bo_z", 10, 150, valinit=Bo_init[2])

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

        def draw_from_Bo():
            nonlocal prev_q
            """
            在 UI 中重绘当前末端位置 Bo_current 的完整 Canfield 几何结构
            —— 使用 IK 查表解主动角 + FK 查表解被动角 + 完整绘制逻辑
            """
            ax3d.cla()

            # === IK: 从目标 Bo_current 反解主动角 ===
            q_deg, info = ik_model.solve(Bo_current, q_curr=prev_q)
            prev_q = q_deg
            
            if any(info["warnings"]) or info["pos_err"] > 2.0:  # 阈值可调，单位 mm
                set_warning_visible(True)
                ax3d.set_title(f"Invalid Move (IK err={info['pos_err']:.2f} mm)", color="red")
                fig.canvas.draw_idle()
                return
            
            # === FK: 用主动角三线插值查表，获得被动角与姿态 ===
            A, M, B, Bo_calc, Rbo, angles = fk_model.forward(q_deg)

            # === 准备“伪造一行 CSV”数据（用于复用 plot_from_lookup 逻辑） ===
            row = {
                "q1_deg": q_deg[0],
                "q2_deg": q_deg[1],
                "q3_deg": q_deg[2],
                "q2_1_deg": angles["q2_1"], "q3_1_deg": angles["q3_1"],
                "q2_2_deg": angles["q2_2"], "q3_2_deg": angles["q3_2"],
                "q2_3_deg": angles["q2_3"], "q3_3_deg": angles["q3_3"],
                # 可选姿态验证
                "Cx_mm": Bo_calc[0], "Cy_mm": Bo_calc[1], "Cz_mm": Bo_calc[2],
                "eul_z_deg": angles["eul_z"], "eul_y_deg": angles["eul_y"], "eul_x_deg": angles["eul_x"],
            }

            # === 复用标准绘图逻辑 ===
            plot_from_lookup(row, ax3d,
                            side=side,
                            axis_to_center_deg=axis_to_center_deg,
                            link_params=(39.5, 34.1, 5),
                            h=h)

            # === UI 显示更新 ===
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
                # q_deg,info = ik_model.solve(candidate)
                # if not any(info["warnings"]):
                Bo_current = candidate
                dot.set_data([Bo_current[0]],[Bo_current[1]])
                s_z.set_val(Bo_current[2])  # 更新 slider
                draw_from_Bo()
                # else:
                #     set_warning_visible(True)
                #     ax3d.set_title("Invalid Move", color="red")
                #     fig.canvas.draw_idle()

        timer = fig.canvas.new_timer(interval=50)
        timer.add_callback(step_update)
        timer.start()

        draw_from_Bo()
        plt.show()

if __name__=="__main__":
    main(kind="ik", mode="show")
