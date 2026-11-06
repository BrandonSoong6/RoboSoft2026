# canfield_plot.py
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# =============================
# 工具：角度与变换
# =============================
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

# =============================
# 残差：6未知数版本 (q4 = -q2)
# =============================
def canfield_residuals(q_pass6, q_actives, side, axis_to_center_deg, link_params):
    B = []; z4 = []; T0_list = []
    for i in range(3):
        q1 = q_actives[i]
        q2, q3 = q_pass6[2*i:2*i+2]
        q4 = -q2  # 强制对称关系
        T0,T1,T2d,T2a,T3,T4 = chain_Ts_for_arm_manhattan(
            i, q1,q2,q3,q4, side, axis_to_center_deg, link_params
        )
        B.append(T4[:3,3]); z4.append(T4[:3,2]); T0_list.append(T0)

    B1,B2,B3 = B
    z41,z42,z43 = z4

    # 上板法向
    n = np.cross(B2-B1, B3-B1)
    n /= (np.linalg.norm(n)+1e-12)
    if n[2] < 0: n = -n

    C = (B1+B2+B3)/3.0
    r1 = (B1-C) / (np.linalg.norm(B1-C)+1e-12)
    r2 = (B2-C) / (np.linalg.norm(B2-C)+1e-12)
    r3 = (B3-C) / (np.linalg.norm(B3-C)+1e-12)
    target = np.deg2rad(axis_to_center_deg)

    s = side
    r = [np.linalg.norm(B2-B1)-s, np.linalg.norm(B3-B2)-s, np.linalg.norm(B1-B3)-s]

    # 平面约束
    r += [np.dot(n, z41), np.dot(n, z42), np.dot(n, z43)]

    # 角度约束
    a1 = signed_angle_in_plane(z41, r1, n)
    a2 = signed_angle_in_plane(z42, r2, n)
    a3 = signed_angle_in_plane(z43, r3, n)
    r += [ang_err_mod_pi(a1, target), ang_err_mod_pi(a2, target), ang_err_mod_pi(a3, target)]
    return np.array(r)


# =============================
# 求解：6未知数版本
# =============================
def solve_passives(q_actives_deg, side=93.16, axis_to_center_deg=164.0,
                           link_params=(39.5,34.1,5)):
    q_act = np.deg2rad(np.asarray(q_actives_deg, dtype=float))
    global q1_1, q1_2, q1_3
    q0 = np.deg2rad(np.asarray([
                    0.0, -q1_1*2,
                    0.0, -q1_2*2,
                    0.0, -q1_3*2
                    ]))

    fun = lambda q: canfield_residuals(q, q_act, side, axis_to_center_deg, link_params)

    lower = np.full(6, -np.pi)
    upper = np.full(6,  np.pi)
    
    # for idx in [0, 2, 4]:
    #     lower[idx] = np.deg2rad(-180.0)
    #     upper[idx] = np.deg2rad(0.0)

    for idx in [1, 3, 5]:
        lower[idx] = np.deg2rad(-180.0)
        upper[idx] = np.deg2rad(0.0)

    sol = least_squares(
        fun, q0,
        bounds=(lower, upper),
        method='trf',
        loss='soft_l1',
        ftol=1e-13, xtol=1e-13, gtol=1e-13,
        max_nfev=10000
    )

    # 还原成完整 9 维解
    q_pass6 = sol.x
    q_pass9 = []
    for i in range(3):
        q2, q3 = q_pass6[2*i:2*i+2]
        q4 = -q2
        q_pass9.extend([q2, q3, q4])
    return np.array(q_pass9), sol


# =============================
# 绘制（完整 + 坐标系 + 法向 + 径向连线）
# =============================
def plot_geometry(q_actives_deg, side=93.16, axis_to_center_deg=-16.0,
                  link_params=(39.5,34.1,5), h=10.0, draw_full=True, draw_base=True,
                  print_angles=True):
    """
    - 径向：按你的要求，画 Ai→O 与 Bi→C 的线段并标注。
    - link1 坐标系：沿底板法向抬升一点，并用半透明灰线指回真实位置。
    - 所有箭头（坐标轴/法向）均加文字标签。
    """
    # ======= 原来这里调用了求解 =======
    q_pass, sol = solve_passives(q_actives_deg, side, axis_to_center_deg, link_params)
    q1s = np.deg2rad(np.asarray(q_actives_deg, dtype=float))

    # ======= 改成直接用 0 度 =======
    # q_pass = np.zeros(9)   # 全部被动角直接置 0
    # sol = None             # 不再有求解结果
    # q1s = np.deg2rad(np.asarray(q_actives_deg, dtype=float))

    colors = {"link1":"red", "link2":"green", "link3":"blue", "link4":"purple"}
    A_pts, B_pts, z4_list, T0_list = [], [], [], []
    nodes = []  # (i, tag, T)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(3):
        q1 = q1s[i]
        q2, q3, q4 = q_pass[3*i:3*i+3]
        T0,T1,T2d,T2a,T3,T4 = chain_Ts_for_arm_manhattan(
            i, q1,q2,q3,q4, side, axis_to_center_deg, link_params, h
        )
        A_pts.append(T0[:3,3]); T0_list.append(T0)
        B_pts.append(T4[:3,3]); z4_list.append(T4[:3,2])
        for tag, T in [("T0",T0),("T1",T1),("T2d",T2d),("T2a",T2a),("T3",T3),("T4",T4)]:
            nodes.append((i, tag, T))

        # 连杆骨架
        ax.plot([T0[0,3], T1[0,3]],[T0[1,3], T1[1,3]],[T0[2,3], T1[2,3]], color=colors["link1"], marker="o")
        ax.plot([T1[0,3], T2d[0,3], T2a[0,3]],
                [T1[1,3], T2d[1,3], T2a[1,3]],
                [T1[2,3], T2d[2,3], T2a[2,3]], color=colors["link2"], marker="o")
        if draw_full:
            ax.plot([T2a[0,3], T3[0,3]],[T2a[1,3], T3[1,3]],[T2a[2,3], T3[2,3]], color=colors["link3"], marker="o")
            ax.plot([T3[0,3], T4[0,3]],[T3[1,3], T4[1,3]],[T3[2,3], T4[2,3]], color=colors["link4"], marker="o")

    # 三角形轮廓
    if draw_base:
        A1,A2,A3 = A_pts
        A_loop = np.vstack([A1,A2,A3,A1])
        ax.plot(A_loop[:,0], A_loop[:,1], A_loop[:,2], linewidth=1, color="black")
    if draw_full and len(B_pts)==3:
        B1,B2,B3 = B_pts
        B_loop = np.vstack([B1,B2,B3,B1])
        ax.plot(B_loop[:,0], B_loop[:,1], B_loop[:,2], linewidth=1, color="gray")

    # 圆心与法向
    O = np.mean(np.vstack(A_pts), axis=0)
    n_base = np.array([0,0,1.0])
    _draw_axis_arrow(ax, O, n_base, "k", "n_base", length=15, fsize=8)

    # 上板圆心 C 和姿态
    if len(B_pts)==3:
        B1,B2,B3 = B_pts
        C = (B1+B2+B3)/3.0
        n_top = np.cross(B2-B1, B3-B1); n_top /= (np.linalg.norm(n_top)+1e-12)
        if n_top[2] < 0: n_top = -n_top

        # 定义 C 坐标系：z = n_top, x = C→B1, y = z×x
        x_axis = B1 - C
        x_axis -= np.dot(x_axis, n_top)*n_top   # 投影到平面
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(n_top, x_axis)
        R_top = np.column_stack([x_axis, y_axis, n_top])

        # 打印姿态角
        eul = R.from_matrix(R_top).as_euler("zyx", degrees=True)
        print("\n== 上板圆心 C ==")
        print("坐标:", np.round(C,3))
        print("姿态角 (yaw, pitch, roll) zyx [deg]:", np.round(eul,3))

        # 画 n_top 向量
        _draw_axis_arrow(ax, C, n_top, "orange", "n_top", length=15, fsize=8)
        # 画上板圆心坐标系
        _draw_axis_arrow(ax, C, x_axis, "r", "Cx", length=12, fsize=8)
        _draw_axis_arrow(ax, C, y_axis, "g", "Cy", length=12, fsize=8)
        _draw_axis_arrow(ax, C, n_top, "b", "Cz", length=12, fsize=8)

    # 六根径向（画线段 + 文本）
    for i, Ai in enumerate(A_pts):
        _draw_line_with_label(ax, Ai, O, "cyan", f"A{i+1}→O", alpha=0.7, fsize=7)
    if len(B_pts)==3:
        for i, Bi in enumerate(B_pts):
            _draw_line_with_label(ax, Bi, C, "magenta", f"B{i+1}→C", alpha=0.7, fsize=7)

    # 各 link 坐标系：link1 抬升并回指
    axis_len = 8.0
    for i, tag, T in nodes:
        o = T[:3,3]; x=T[:3,0]; y=T[:3,1]; z=T[:3,2]
        o_draw = o
        if tag == "T1":  # 抬升显示
            offset = np.array([20.0,0,-15.0])   # 偏移距离 (mm)
            o_draw = o + offset
            ax.plot([o[0], o_draw[0]], [o[1], o_draw[1]], [o[2], o_draw[2]],
                    color="0.5", alpha=0.35, linestyle="--")
        _draw_axis_arrow(ax, o_draw, x, "r", f"{i+1}:{tag}-x", length=axis_len)
        _draw_axis_arrow(ax, o_draw, y, "g", f"{i+1}:{tag}-y", length=axis_len)
        _draw_axis_arrow(ax, o_draw, z, "b", f"{i+1}:{tag}-z", length=axis_len)

    # 视域
    bounds = [*A_pts, *(B_pts if draw_full else [])]
    if len(bounds) == 0: bounds = [np.zeros(3)]
    all_pts = np.vstack(bounds)
    mins,maxs = all_pts.min(axis=0), all_pts.max(axis=0)
    rng = (maxs-mins).max(); center=(maxs+mins)/2
    ax.set_xlim(center[0]-rng/2, center[0]+rng/2)
    ax.set_ylim(center[1]-rng/2, center[1]+rng/2)
    ax.set_zlim(center[2]-rng/2, center[2]+rng/2)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    ax.set_title("Canfield Joint — 坐标系、法向、径向连线")
    plt.show()

    # 残差详细核验：显示 target 与当前值
    if print_angles and len(B_pts) == 3:
        q_act = np.deg2rad(np.asarray(q_actives_deg, dtype=float))
        res = canfield_residuals(q_pass, q_act, side, axis_to_center_deg, link_params)

        B1,B2,B3 = B_pts
        C = (B1+B2+B3)/3.0
        n_top = np.cross(B2-B1, B3-B1); n_top /= (np.linalg.norm(n_top)+1e-12)
        if n_top[2] < 0: n_top = -n_top

        # 三边长度目标 = side
        lengths = [np.linalg.norm(B2-B1), np.linalg.norm(B3-B2), np.linalg.norm(B1-B3)]
        # z4 在平面内 → dot(n_top, z4)=0
        dots = [np.dot(n_top, z4_list[i]) for i in range(3)]
        # 角度：z4 与径向的平面角
        target_angle = np.deg2rad(axis_to_center_deg)
        cur_angles = []
        for i in range(3):
            ri = (B_pts[i]-C); ri /= np.linalg.norm(ri)+1e-12
            ang = signed_angle_in_plane(z4_list[i], ri, n_top)
            cur_angles.append(ang)

        print("\n== 约束逐项对照 ==")
        for i in range(3):
            print(f"边长约束{i+1}: 目标={side:.3f}, 当前={lengths[i]:.3f}, 残差={res[i]:+.6f}")
        for i in range(3):
            print(f"平面约束{i+1}: 目标=0.000, 当前={dots[i]:+.6f}, 残差={res[3+i]:+.6f}")
        for i in range(3):
            print(f"角度约束{i+1}: 目标={np.rad2deg(target_angle):+.3f} deg, "
                  f"当前={np.rad2deg(cur_angles[i]):+.3f} deg, 残差={np.rad2deg(res[6+i]):+.6f} deg")

        if sol is not None:
            print("LM 残差平方和:", sol.cost, "  收敛:", sol.success)

    return np.rad2deg(q_pass)

def test_random_cases(n_trials=1000, side=93.16, axis_to_center_deg=-16.0,
                      link_params=(39.5,34.1,5), tol=1e-9):
    bad_cases = 0
    for trial in range(n_trials):
        # 随机生成主动角 (0~90°)
        global q1_1, q1_2, q1_3
        q1_all = np.random.uniform(0.0, 90.0, size=3)
        q1_1, q1_2, q1_3 = q1_all  # 随机生成
        q_actives_deg = [90 + q for q in q1_all]

        # 解算
        q_pass, sol = solve_passives(q_actives_deg, side, axis_to_center_deg, link_params)
        q_act = np.deg2rad(np.asarray(q_actives_deg, dtype=float))
        res = canfield_residuals(q_pass[::3*0 or None], q_act, side, axis_to_center_deg, link_params)
        # 注意：canfield_residuals 现在是 6维输入，这里直接传 q_pass9 会报错
        # 所以我们需要传 6维等价形式
        q_pass6 = []
        for i in range(3):
            q2, q3, q4 = q_pass[3*i:3*i+3]
            q_pass6.extend([q2, q3])
        q_pass6 = np.array(q_pass6)

        res = canfield_residuals(q_pass6, q_act, side, axis_to_center_deg, link_params)
        err = np.linalg.norm(res)

        if err > tol:
            bad_cases += 1
            q2_signs = [np.sign(q_pass[3*i]) for i in range(3)]
            print(f"Case {trial}: res={err:.2e}, q2 signs={q2_signs}, q2 values={q_pass[::3]}")
    print(f"\n统计结果: {bad_cases}/{n_trials} 个 case 残差 > {tol}")

# =============================
# 示例
# =============================
if __name__ == "__main__":
    test_random_cases(100)

