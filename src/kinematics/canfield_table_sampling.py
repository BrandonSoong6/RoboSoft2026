import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import gzip
import os
from time import time

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
    """底板第 i 个基座的齐次变换（i ∈ {0,1,2}）。轴到中心夹角使用 axis_to_center_deg。"""
    R0 = side / np.sqrt(3.0)
    theta0 = np.deg2rad([0, 120, 240][i])
    beta = theta0 - np.deg2rad(axis_to_center_deg)
    xi, yi = R0*np.cos(theta0), R0*np.sin(theta0)
    return np.array([
        [-np.sin(beta), 0,  np.cos(beta), xi],
        [ np.cos(beta), 0,  np.sin(beta), yi],
        [ 0,            1,  0,            h ],
        [ 0,            0,  0,            1 ]
    ])

def signed_angle_in_plane(u, v, n):
    """向量 u 到 v 在法向 n 所定义平面内的带符号夹角（[-pi, pi]）。"""
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
    """将角度误差按 180° 对称等价类收敛到 [-90°, 90°] 区间（以弧度计）。"""
    e = wrap_pi(angle - target)
    if e > np.pi/2: e -= np.pi
    elif e < -np.pi/2: e += np.pi
    return e

# =============================
# 链：曼哈顿拆分
# =============================
def chain_Ts_for_arm_manhattan(i, q1, q2, q3, q4,
                               side, axis_to_center_deg, link_params, h=10.0):
    """返回第 i 条臂端 T4（末端上板关节的齐次变换）。"""
    L1, L2, L3 = link_params
    T0 = make_T0(i, side, h, axis_to_center_deg)
    A1  = dh_transform(q1, 0, 0, +np.pi/2)
    A2d = dh_transform(q2, L2 + L1, 0, -np.pi/2)
    A2a = dh_transform(0, 0, L3, 0)
    A3  = dh_transform(q3, 0, L3, -np.pi/2)
    A4  = dh_transform(q4, L2 + L1, 0, +np.pi/2)

    T1   = T0 @ A1
    T2d  = T1 @ A2d
    T2a  = T2d @ A2a
    T3   = T2a @ A3
    T4   = T3 @ A4
    return T4

# =============================
# 残差：9 条约束
#   1-3：上板三边长度等于 side
#   4-6：每条臂的 z4 与上板法向 n 垂直（z4 在上板平面内）
#   7-9：每条臂的 z4 与外向径向向量的平面夹角 = axis_to_center_deg
# =============================
def canfield_residuals(q_passive, q_actives, side, axis_to_center_deg, link_params):
    B = []; z4 = []
    for i in range(3):
        q1 = q_actives[i]
        q2, q3, q4 = q_passive[3*i:3*i+3]
        T4 = chain_Ts_for_arm_manhattan(i, q1, q2, q3, q4, side, axis_to_center_deg, link_params)
        B.append(T4[:3,3]); z4.append(T4[:3,2])

    B1, B2, B3 = B
    z41, z42, z43 = z4

    # 上板法向（与 +Z 同向以固定法向符号）
    n = np.cross(B2 - B1, B3 - B1)
    n /= (np.linalg.norm(n) + 1e-12)
    if n[2] < 0:
        n = -n

    C = (B1 + B2 + B3) / 3.0
    r1 = (B1 - C) / (np.linalg.norm(B1 - C) + 1e-12)
    r2 = (B2 - C) / (np.linalg.norm(B2 - C) + 1e-12)
    r3 = (B3 - C) / (np.linalg.norm(B3 - C) + 1e-12)
    target = np.deg2rad(axis_to_center_deg)

    s = side
    res = [
        np.linalg.norm(B2 - B1) - s,
        np.linalg.norm(B3 - B2) - s,
        np.linalg.norm(B1 - B3) - s,
        np.dot(n, z41),
        np.dot(n, z42),
        np.dot(n, z43),
        ang_err_mod_pi(signed_angle_in_plane(z41, r1, n), target),
        ang_err_mod_pi(signed_angle_in_plane(z42, r2, n), target),
        ang_err_mod_pi(signed_angle_in_plane(z43, r3, n), target),
    ]
    return np.array(res)

# =============================
# 工具：角度差 wrap 到 [-pi, pi]
# =============================
def angdiff(a, b):
    """返回 a-b 的角度差，逐元素 wrap 到 [-pi, pi]。"""
    d = a - b
    return (d + np.pi) % (2*np.pi) - np.pi

def make_following_residual(fun_core, q_prev_rad=None, lam=0.0):
    """
    包装核心残差，加一个 L2 正则项 sqrt(lam)*(q - q_prev_wrapped)，
    以“吸附”到上一步的解（同伦延续时稳定跟踪同一分支）。
    """
    if q_prev_rad is None or lam <= 0.0:
        return lambda q: fun_core(q)

    sqrt_lam = np.sqrt(lam)
    def fun(q):
        r_core = fun_core(q)
        dq = angdiff(q, q_prev_rad)
        r_reg = sqrt_lam * dq
        return np.concatenate([r_core, r_reg])
    return fun

# =============================
# 单点被动角求解（最快）
# =============================
def solve_passives(q_actives_deg, side=93.16, axis_to_center_deg=-16.0,
                   link_params=(39.5,34.1,5)):
    """
    给定主动角（deg），解被动角（rad）。这一版追求速度，不保证不跳支。
    返回 (q_pass[rad], solver_result)
    """
    q_act = np.deg2rad(np.asarray(q_actives_deg, dtype=float))

    # 初值（可按你的机构偏好修改）
    q0 = np.deg2rad(np.asarray([
        0.0, -90.0, 0.0,
        0.0, -90.0, 0.0,
        0.0, -90.0, 0.0
    ]))

    fun = lambda q: canfield_residuals(q, q_act, side, axis_to_center_deg, link_params)

    # 边界：示例只限制第 2、5、8 元素（对应各臂的 q3）在 [-180°, 0°]
    lower = np.full(9, -np.pi)
    upper = np.full(9,  np.pi)
    for idx in [1, 4, 7]:
        lower[idx] = np.deg2rad(-180.0)
        upper[idx] = np.deg2rad(0.0)

    sol = least_squares(
        fun, q0,
        bounds=(lower, upper),
        method='trf',
        loss='soft_l1',
        ftol=1e-10, xtol=1e-10, gtol=1e-10,
        max_nfev=8000
    )
    return sol.x, sol

# =============================
# 从(q_act, q_pass)计算上板圆心位姿
# =============================
def center_pose_from_q(q_act_rad, q_pass_rad, side=93.16, axis_to_center_deg=-16.0,
                       link_params=(39.5,34.1,5), h=10.0):
    """给定主动/被动角（rad），返回 (C[3], eul_zyx[3 deg])。"""
    B = []
    for i in range(3):
        q1 = q_act_rad[i]
        q2, q3, q4 = q_pass_rad[3*i:3*i+3]
        T4 = chain_Ts_for_arm_manhattan(i, q1, q2, q3, q4,
                                        side, axis_to_center_deg, link_params, h)
        B.append(T4[:3,3])
    B1, B2, B3 = B
    C = (B1 + B2 + B3) / 3.0

    # 上板姿态（z = 法向，x = C->B1 在平面投影，y = z×x）
    n = np.cross(B2 - B1, B3 - B1)
    n /= (np.linalg.norm(n) + 1e-12)
    if n[2] < 0:
        n = -n
    x_axis = (B1 - C)
    x_axis -= np.dot(x_axis, n) * n
    x_axis /= (np.linalg.norm(x_axis) + 1e-12)
    y_axis = np.cross(n, x_axis)
    R_top = np.column_stack([x_axis, y_axis, n])
    eul_zyx = R.from_matrix(R_top).as_euler("zyx", degrees=True)
    C = 5 * n + C  # 圆心稍微抬高一点
    return C, eul_zyx

# =============================
# 同伦延续：沿起点分支走到目标（稳定）
# =============================
def solve_center_pose_homotopy(q_act_start_deg, q_act_goal_deg,
                               side=93.16, axis_to_center_deg=-16.0,
                               link_params=(39.5,34.1,5), h=10.0,
                               n_steps=25, lam_follow=1e-4,
                               bounds_q3_deg=(-180.0, 0.0),
                               verbose=False):
    """
    从起点主动角走到目标主动角，用同伦延续保证始终跟随同一分支。
    返回目标位姿 (C, eul_zyx)，以及最终被动角（deg）。
    - n_steps: 线性插值步数（越大越稳，越小越快）
    - lam_follow: “跟随”正则强度（>0 则吸附到上一步解）
    - bounds_q3_deg: 对每条臂的第2个被动角(q3)的范围限制（按你的机构可调整/去掉）
    """
    q_act_start = np.array(q_act_start_deg, dtype=float)
    q_act_goal  = np.array(q_act_goal_deg,  dtype=float)

    # 起点先解一次（可能相对慢，但只做一次）
    q_prev, sol0 = solve_passives(q_act_start, side, axis_to_center_deg, link_params)
    if verbose:
        print(f"[init] cost={sol0.cost:.3e}, success={sol0.success}")

    # 线性同伦路径
    qs = np.linspace(0.0, 1.0, n_steps)
    for k, t in enumerate(qs[1:], start=1):
        q_act_try_deg = (1.0 - t) * q_act_start + t * q_act_goal
        q_act_try_rad = np.deg2rad(q_act_try_deg)

        core_fun = lambda q: canfield_residuals(q, q_act_try_rad, side, axis_to_center_deg, link_params)
        # fun = make_following_residual(core_fun, q_prev_rad=q_prev, lam=lam_follow)
        fun = core_fun

        # 边界（示例只约束 q3 分量）
        lower = np.full(9, -np.pi)
        upper = np.full(9,  np.pi)
        lo_q3, hi_q3 = np.deg2rad(bounds_q3_deg[0]), np.deg2rad(bounds_q3_deg[1])
        for idx in [1, 4, 7]:
            lower[idx] = lo_q3
            upper[idx] = hi_q3

        sol = least_squares(
            fun, q_prev,
            bounds=(lower, upper),
            method='trf', loss='linear',
            ftol=1e-12, xtol=1e-12, gtol=1e-12,
            max_nfev=8000
        )
        q_prev = sol.x.copy()
        if verbose:
            res_core = core_fun(q_prev)
            print(f"[{k:03d}/{n_steps-1}] t={t:.3f}, ||res||={np.linalg.norm(res_core):.3e}, success={sol.success}")

    # 终点位姿
    C, eul = center_pose_from_q(np.deg2rad(q_act_goal), q_prev, side, axis_to_center_deg, link_params, h)
    return C, eul, np.rad2deg(q_prev)

# ---------- 非均匀步长轴生成 (0-70 deg) ----------
def nonuniform_axis():
    a1 = np.arange(90.1, 115.0 + 1e-9, 3.0)   # 0-25, step 0.5
    a2 = np.arange(116.0, 138.0 + 1e-9, 3.0)  # 26-48, step 1
    a3 = np.arange(140.0, 160.0 + 1e-9, 3.0)  # 50-70, step 2
    axis = np.concatenate([a1, a2]).astype(float)
    return axis

def count_simplex_points(N):
    # 约束 q1>=q2>=q3 的组合数：C(N+2, 3) = N*(N+1)*(N+2)/6
    return N*(N+1)*(N+2)//6

# ---------- 简单x层(固定q1)下的三角网格蛇形 ----------
def simplex_snake_loop(ax):
    """
    逐层(i1)扫，每层内对(i2,i3)按三角网格蛇形：
      - i2 行方向在该层内交替（根据 i1 奇偶翻转，保证跨层衔接更平滑）
      - 每行的 i3 方向也交替（根据 i1+行号奇偶）
    邻接性：层内单步仅变更一维；跨层时(i1→i1+1)不可避免多一维变化，但跳幅最小。
    """
    N = len(ax)
    for i1 in range(N):
        if i1 % 2 == 0:
            i2_rows = range(0, i1+1)           # 0..i1
        else:
            i2_rows = range(i1, -1, -1)        # i1..0

        for r_idx, i2 in enumerate(i2_rows):
            # 该层内第 r_idx 行的 i3 方向交替
            if ((i1 + r_idx) % 2 == 0):
                i3_cols = range(0, i2+1)       # 0..i2
            else:
                i3_cols = range(i2, -1, -1)    # i2..0
            for i3 in i3_cols:
                yield (ax[i1], ax[i2], ax[i3])

# ---------- 单点求解：用上一步的 q_pass 作为初值 + 跟随正则 ----------
def solve_passives_follow(q_act_deg, q_prev_pass_rad,
                          side=93.16, axis_to_center_deg=-16.0, link_params=(39.5,34.1,5),
                          lam_follow=2e-3, bounds_q3_deg=(-180.0, 0.0),
                          ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=8000):
    q_act = np.deg2rad(np.asarray(q_act_deg, dtype=float))
    core_fun = lambda q: canfield_residuals(q, q_act, side, axis_to_center_deg, link_params)
    fun = make_following_residual(core_fun, q_prev_pass_rad, lam=lam_follow)

    lower = np.full(9, -np.pi); upper = np.full(9, np.pi)
    lo_q3, hi_q3 = np.deg2rad(bounds_q3_deg[0]), np.deg2rad(bounds_q3_deg[1])
    for idx in [1, 4, 7]:
        lower[idx] = lo_q3; upper[idx] = hi_q3

    sol = least_squares(fun, q_prev_pass_rad,
                        bounds=(lower, upper),
                        method='trf', loss='linear',
                        ftol=ftol, xtol=xtol, gtol=gtol,
                        max_nfev=max_nfev)
    
    res_core = core_fun(sol.x.copy())
    return sol, res_core

# ---------- 打表主流程 ----------
def sample_workspace_table(
    side=93.16,
    axis_to_center_deg=-16.0,
    link_params=(39.5,34.1,5),
    h=10.0,
    start_center=(135.0, 135.0, 135.0),
    out_path="samples_canfield.csv",
    buffer_size=2000,
    lam_follow=2e-4,
    bounds_q3_deg=(-180.0, 0.0),
    resume=False
):
    ax = nonuniform_axis()
    N = len(ax)
    total_pts = count_simplex_points(N)
    print(f"[info] axis_len={N} → simplex_total={total_pts}")

    # 中心 → 原点 (90,90,90) 的同伦
    _C0, _eul0, qpass0_deg = solve_center_pose_homotopy(
        start_center, (90.1, 90.1, 90.1),
        side=side, axis_to_center_deg=axis_to_center_deg,
        link_params=link_params, h=h,
        n_steps=60, lam_follow=lam_follow,
        bounds_q3_deg=bounds_q3_deg, verbose=True
    )
    q_prev = np.deg2rad(qpass0_deg)  # 作为原点邻域的初值

    t0 = time()

    # 输出文件头
    header = "q1_deg,q2_deg,q3_deg,eul_z_deg,eul_y_deg,eul_x_deg,Cx_mm,Cy_mm,Cz_mm,success,res_norm\n"
    mode = "ab" if (resume and os.path.exists(out_path)) else "wb"
    fopener = gzip.open if out_path.endswith(".gz") else open
    with fopener(out_path, mode) as f:
        if mode == "wb":
            f.write(header.encode("utf-8"))

        buf = []
        t0 = time()
        done = 0

        angle_thresh = np.deg2rad(20.0)  # 允许的最大相邻姿态旋转角
        R_prev = None  # 上一步的旋转矩阵   

        # 3) 从 (90,90,90) 开始蛇形遍历
        for (q1, q2, q3) in simplex_snake_loop(ax):
            q_act_deg = (q1, q2, q3)
            # 上一步解作初值
            sol, res_core = solve_passives_follow(
                q_act_deg, q_prev,
                side=side, axis_to_center_deg=axis_to_center_deg, link_params=link_params,
                lam_follow=lam_follow, bounds_q3_deg=bounds_q3_deg,
                ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=8000
            )
            q_prev = sol.x.copy()

            # 末端位姿
            C, eul = center_pose_from_q(np.deg2rad(q_act_deg), q_prev,
                                        side, axis_to_center_deg, link_params, h)
            R_curr = R.from_euler("zyx", eul, degrees=True).as_matrix()

            # 分支一致性检查
            jump_flag = True
            if R_prev is not None:
                dR = R.from_matrix(R_prev.T @ R_curr)
                angle = dR.magnitude()
                if angle > angle_thresh:
                    jump_flag = True
                    print(f"\n[warn] jump detected at q={q_act_deg}, Δangle={np.rad2deg(angle):.1f} deg")

            R_prev = R_curr

            # 写出行，若 jump_flag=True 则 success=0
            row = (*q_act_deg, *eul.tolist(), *C.tolist(),
                   float(sol.success and not jump_flag), np.linalg.norm(res_core))
            buf.append(",".join(f"{v:.6f}" for v in row) + "\n")

            # 缓冲写盘
            if len(buf) >= buffer_size:
                f.writelines([s.encode("utf-8") for s in buf])
                buf.clear()

            done += 1
            pct = 100.0 * done / total_pts

            # 每100步更新一次进度条
            if done % 10 == 0:
                bar_len = 40
                filled = int(bar_len * done / total_pts)
                bar = "#" * filled + "-" * (bar_len - filled)
                dt = time() - t0
                rate_per_min = (done / dt) * 60.0
                eta = (total_pts - done) / (done / dt)           # 剩余秒
                eta_min, eta_sec = divmod(int(eta), 60)
                print(f"\r[progress] |{bar}| {done}/{total_pts} ({pct:.2f}%) "
                      f"| {rate_per_min:.1f} pts/min | ETA {eta_min}m{eta_sec}s",
                      end="", flush=True)

            # 每1000步输出一次详细信息
            if done % 100 == 0:
                dt = time() - t0
                rate = done / max(dt, 1e-9)
                print(f"\n[detail] {done}/{total_pts} ({pct:.2f}%) "
                      f"| {rate:.1f} pts/s | res={np.linalg.norm(res_core):.3e}")

        # 写完尾巴
        if buf:
            f.writelines([s.encode("utf-8") for s in buf])
            buf.clear()

    print(f"[done] wrote: {out_path}")

if __name__ == "__main__":
    # 你也可以把打表放到原 __main__ 里，这里给一个独立入口
    sample_workspace_table(
        side=93.16,
        axis_to_center_deg=-16.0,
        link_params=(39.5, 34.1, 5),
        h=10.0,
        start_center=(135.0, 135.0, 135.0),
        out_path="samples_canfield.csv",    # 改成 .csv 输出普通 CSV
        buffer_size=2000,
        lam_follow=2e-4,
        bounds_q3_deg=(-180.0, 0.0),
        resume=False
    )
