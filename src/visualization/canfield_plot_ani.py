# canfield_plot.py
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import warnings
import time
from matplotlib import animation
warnings.filterwarnings("ignore", category=UserWarning)

# =============================
# 工具：角度与变换
# =============================

# ========= 加速算子：缓存T0 + 专用DH快算 =========
try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

def _make_T0s(side, h, axis_to_center_deg):
    T0s = np.empty((3, 4, 4), dtype=float)
    R_ = side / np.sqrt(3.0)
    thetas = np.deg2rad([0.0, 120.0, 240.0])
    axis = np.deg2rad(axis_to_center_deg)
    for i in range(3):
        theta0 = thetas[i]
        beta = theta0 - axis
        xi, yi = R_*np.cos(theta0), R_*np.sin(theta0)
        T0 = np.array([
            [-np.sin(beta), 0.0,  np.cos(beta), xi],
            [ np.cos(beta), 0.0,  np.sin(beta), yi],
            [ 0.0,          1.0,  0.0,          h ],
            [ 0.0,          0.0,  0.0,          1.0 ]
        ])
        T0s[i,:,:] = T0
    return T0s

# —— 专用DH快算（alpha固定为±pi/2），少乘/少alloc —— #
def _A1_fast(q1):
    ct, st = np.cos(q1), np.sin(q1)
    # alpha = +pi/2, a=0, d=0
    return np.array([[ct, 0.0,  st, 0.0],
                     [st, 0.0, -ct, 0.0],
                     [0.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])

def _A2d_fast(q2, d):
    ct, st = np.cos(q2), np.sin(q2)
    # alpha = -pi/2, a=0, d=d
    return np.array([[ct, 0.0, -st, 0.0],
                     [st, 0.0,  ct, 0.0],
                     [0.0,-1.0, 0.0,   d],
                     [0.0, 0.0, 0.0, 1.0]])

def _A2a_fast(L3):
    # q=0, a=L3
    return np.array([[1.0, 0.0, 0.0, L3],
                     [0.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])

def _A3_fast(q3, L3):
    ct, st = np.cos(q3), np.sin(q3)
    # alpha = -pi/2, a=L3, d=0
    return np.array([[ct, 0.0, -st, L3*ct],
                     [st, 0.0,  ct, L3*st],
                     [0.0,-1.0, 0.0,    0.0],
                     [0.0, 0.0, 0.0,    1.0]])

def _A4_fast(q4, d):
    ct, st = np.cos(q4), np.sin(q4)
    # alpha = +pi/2, a=0, d=d
    return np.array([[ct, 0.0,  st, 0.0],
                     [st, 0.0, -ct, 0.0],
                     [0.0, 1.0, 0.0,   d],
                     [0.0, 0.0, 0.0, 1.0]])

def _chain_end_fast(T0, q1, q2, q3, q4, L1, L2, L3, out_B, out_z):
    d = L1 + L2
    A1  = _A1_fast(q1)
    A2d = _A2d_fast(q2, d)
    A2a = _A2a_fast(L3)
    A3  = _A3_fast(q3, L3)
    A4  = _A4_fast(q4, d)

    T1  = T0 @ A1
    T2d = T1 @ A2d
    T2a = T2d @ A2a
    T3  = T2a @ A3
    T4  = T3 @ A4

    # 只取你需要的：末端位置 B、末端 z 轴
    out_B[:] = T4[:3, 3]
    out_z[:] = T4[:3, 2]

# （可选）Numba JIT：把小核JIT掉，单次残差能快一些
if _HAS_NUMBA:
    _A1_fast     = njit(cache=True, fastmath=True)(_A1_fast)
    _A2d_fast    = njit(cache=True, fastmath=True)(_A2d_fast)
    _A2a_fast    = njit(cache=True, fastmath=True)(_A2a_fast)
    _A3_fast     = njit(cache=True, fastmath=True)(_A3_fast)
    _A4_fast     = njit(cache=True, fastmath=True)(_A4_fast)

class CanfieldOps:
    """离线缓存 + 快速残差构造器"""
    def __init__(self, side=93.16, axis_to_center_deg=-16.0, link_params=(39.5,34.1,5), h=10.0, use_numba=True):
        self.side = float(side)
        self.axis_to_center_deg = float(axis_to_center_deg)
        self.h = float(h)
        self.L1, self.L2, self.L3 = map(float, link_params)
        self.T0s = _make_T0s(self.side, self.h, self.axis_to_center_deg)
        self.target = np.deg2rad(self.axis_to_center_deg)
        self.use_numba = bool(use_numba and _HAS_NUMBA)

        # 预分配工作区，避免反复分配
        self._B = np.empty((3,3), dtype=float)    # 三个端点 Bi
        self._z = np.empty((3,3), dtype=float)    # 三个末端 z 轴
        self._r = np.empty(9, dtype=float)        # 残差

    def make_residual(self, q_actives_rad):
        """返回 f(q_passive_rad_6) -> r(9,)
        输入 q_passive_rad_6 = [q2,q3]*3
        自动补上 q4 = -q2
        """
        q_act = np.asarray(q_actives_rad, dtype=float)
        assert q_act.shape == (3,)

        def residual(q_pass6):
            # —— 展开成 9 维 —— #
            q_pass = np.zeros(9, dtype=float)
            for i in range(3):
                q2, q3 = q_pass6[2*i:2*i+2]
                q4 = -q2
                q_pass[3*i:3*i+3] = [q2, q3, q4]

            # —— 三条支链末端 —— #
            for i in range(3):
                q2,q3,q4 = q_pass[3*i:3*i+3]
                T0 = self.T0s[i]
                _chain_end_fast(T0, q_act[i], q2, q3, q4,
                                self.L1, self.L2, self.L3,
                                self._B[i], self._z[i])

            B1, B2, B3 = self._B
            z41, z42, z43 = self._z

            # —— 2) 组装几何量 —— #
            v21 = B2 - B1
            v32 = B3 - B2
            v13 = B1 - B3

            # 上板法向（方向统一）
            n = np.cross(v21, v32)
            nn = np.linalg.norm(n)
            if nn < 1e-15:
                n[:] = 0.0
            else:
                n /= nn
            if n[2] < 0.0: n = -n

            C = (B1 + B2 + B3) / 3.0

            # 外向径向向量
            r1 = B1 - C; r1 /= (np.linalg.norm(r1) + 1e-12)
            r2 = B2 - C; r2 /= (np.linalg.norm(r2) + 1e-12)
            r3 = B3 - C; r3 /= (np.linalg.norm(r3) + 1e-12)

            # —— 3) 残差：长度约束（更快的范数计算） —— #
            self._r[0] = np.linalg.norm(v21) - self.side
            self._r[1] = np.linalg.norm(v32) - self.side
            self._r[2] = np.linalg.norm(v13) - self.side

            # —— 4) 平面内约束 —— #
            self._r[3] = np.dot(n, z41)
            self._r[4] = np.dot(n, z42)
            self._r[5] = np.dot(n, z43)

            # —— 5) 平面内带符号角度约束 —— #
            # 用更直接的“atan2( n·(u×v) , u·v )”形式，不再反复创建投影
            def signed_ang(u, v, n):
                # 假设u,v已基本在平面内；如需更严谨，可投影一次
                s = np.dot(n, np.cross(u, v))
                c = np.dot(u, v)
                return np.arctan2(s, c)

            a1 = signed_ang(z41 - np.dot(z41, n)*n, r1, n)  # 投影一次，稳一点
            a2 = signed_ang(z42 - np.dot(z42, n)*n, r2, n)
            a3 = signed_ang(z43 - np.dot(z43, n)*n, r3, n)

            # wrap到[-pi,pi]并考虑±180翻面
            def ang_err_mod_pi_local(angle, target):
                e = (angle - target + np.pi) % (2*np.pi) - np.pi
                if e >  np.pi/2: e -= np.pi
                if e < -np.pi/2: e += np.pi
                return e

            self._r[6] = ang_err_mod_pi_local(a1, self.target)
            self._r[7] = ang_err_mod_pi_local(a2, self.target)
            self._r[8] = ang_err_mod_pi_local(a3, self.target)

            # print(f"\n[debug] \nlen_res=({self._r[0]:+.3e}, {self._r[1]:+.3e}, {self._r[2]:+.3e}) \n"
            # f"plane_res=({self._r[3]:+.3e}, {self._r[4]:+.3e}, {self._r[5]:+.3e}) \n"
            # f"ang_res=({self._r[6]:+.3e}, {self._r[7]:+.3e}, {self._r[8]:+.3e})\n")

            return self._r.copy()
            
        return residual
    
def cluster_ok_fast(q_act_rad, q_pass_rad, ops, margin_deg, eps=1e-5):
    """
    简化版：同簇判定只看 q2 的符号。
    - 若任意一个 q2 落在 [-eps, +eps]，直接认为同簇。
    - 否则要求三个 q2 不全同号。
    - 返回 (ok, q2_vals)。
    """
    q2_vals = [q_pass_rad[3*i] for i in range(3)]

    # 如果有一个接近 0，则直接判定为同簇
    if any(abs(val) < eps for val in q2_vals):
        return True, q2_vals

    # 否则正常判定符号
    signs = [np.sign(val) for val in q2_vals]
    all_pos = all(s > 0 for s in signs)
    all_neg = all(s < 0 for s in signs)
    ok = not (all_pos or all_neg)

    return ok, q2_vals


# =============================
# 求解
# =============================
def solve_passives(q_actives_deg, side=93.16, axis_to_center_deg=-16.0,
                   link_params=(39.5,34.1,5)):
    # 初始值：6 维 [q2,q3]*3
    q0 = np.deg2rad(np.asarray([
        0.0, -90.0,
        0.0, -90.0,
        0.0, -90.0
    ]))

    q_act_rad = np.deg2rad(np.asarray(q_actives_deg, dtype=float))
    ops = CanfieldOps(side=side, axis_to_center_deg=axis_to_center_deg,
                      link_params=link_params, h=10.0)
    fun = ops.make_residual(q_act_rad)

    sol = least_squares(
        fun, q0,
        method='trf',
        loss='linear',
        ftol=1e-10, xtol=1e-10, gtol=1e-10,
        max_nfev=8000
    )

    # —— 解恢复成 9 维以兼容绘图 —— #
    q_pass9 = np.zeros(9)
    for i in range(3):
        q2,q3 = sol.x[2*i:2*i+2]
        q4 = -q2
        q_pass9[3*i:3*i+3] = [q2,q3,q4]

    return q_pass9, sol

# =============================
# 工具：角度差 wrap 到 [-pi, pi]
# =============================
def angdiff(a, b):
    """返回 a-b 的角度差，wrap 到 [-pi, pi]。a,b 可是向量"""
    d = a - b
    return (d + np.pi) % (2*np.pi) - np.pi

# =============================
# 包装残差：加小正则以跟随上一帧解
# =============================
def make_following_residual(fun_core, q_prev_rad=None, lam=0.0):
    """
    fun_core(q) -> r_core (长度9)
    返回 fun(q) -> concat([r_core, sqrt(lam)*(q - q_prev_wrapped)])
    """
    if q_prev_rad is None or lam <= 0.0:
        return lambda q: fun_core(q)

    sqrt_lam = np.sqrt(lam)
    def fun(q):
        r_core = fun_core(q)
        dq = angdiff(q, q_prev_rad)  # wrap 比较
        r_reg = sqrt_lam * dq
        return np.concatenate([r_core, r_reg])
    return fun

# =============================
# 连续延续：从 q_act_start 到 q_act_goal
# =============================
def continue_solve_path(q_act_start_deg, q_act_goal_deg,
                        side=93.16, axis_to_center_deg=-16.0,
                        link_params=(39.5,34.1,5),
                        max_halves=18,
                        tol_success=1e-5,
                        angle_margin_deg=0.00001,  # 仍保留，用于接口一致
                        verbose=True):
    """
    自适应步长延续（纯几何残差；内部 6 维优化：每链仅 (q2,q3)，且 q4=-q2）
    同簇判定（简化版）：三个 q2 的符号不全相同即可视为同簇。

    返回：q_act_list (Nx3, deg), q_pass_list (Nx9, deg；含自动补 q4=-q2)
    """
    q_act_start = np.array(q_act_start_deg, dtype=float)
    q_act_goal  = np.array(q_act_goal_deg,  dtype=float)
    print(f"q_act_goal = {q_act_goal_deg}")
    assert q_act_start.shape == (3,) and q_act_goal.shape == (3,)

    # 起点解（solve_passives 已经做 6 维优化，并返回 9 维兼容解）
    q_pass0, sol0 = solve_passives(q_act_start, side, axis_to_center_deg, link_params)
    if verbose:
        print(f"[init] start_cost={sol0.cost:.6e}, success={sol0.success}")

    q_pass_list = [np.rad2deg(q_pass0)]   # 存 9 维，用于绘图
    q_act_list  = [q_act_start.copy()]

    t = 0.0
    step = 0.01
    q_prev6 = sol0.x.copy()               # 6 维变量的当前锚点

    ops = CanfieldOps(side=side, axis_to_center_deg=axis_to_center_deg,
                      link_params=link_params, h=10.0)

    while t < 1.0 - 1e-12:
        # 试探下一步
        t_try = min(1.0, t + step)
        q_act_try = (1.0 - t_try) * q_act_start + t_try * q_act_goal
        q_act_rad = np.deg2rad(q_act_try)

        # 为当前 t_try 构造纯几何残差（6 维输入）
        core_fun = ops.make_residual(q_act_rad)

        ok = False
        halves = 0
        q0 = q_prev6.copy()               # 本轮优化的初值（6 维）

        while halves <= max_halves:
            # —— 6 维优化 —— #
            sol = least_squares(core_fun, q0,
                                method='trf', loss='linear',
                                ftol=1e-14, xtol=1e-14, gtol=1e-14,
                                max_nfev=4000)
            r_core = core_fun(sol.x)
            res_core = np.linalg.norm(r_core)

            # —— 将 6 维解展开为 9 维（用于同簇判定与绘图）—— #
            q_pass9 = np.zeros(9)
            for i in range(3):
                q2, q3 = sol.x[2*i:2*i+2]
                q4 = -q2
                q_pass9[3*i:3*i+3] = [q2, q3, q4]

            # —— 同簇判定（简化规则：三个 q2 符号不全相同）—— #
            same_cluster, q2_vals = cluster_ok_fast(q_act_rad, q_pass9, ops, margin_deg=angle_margin_deg)

            if verbose:
                q2_deg = np.rad2deg([q_pass9[0], q_pass9[3], q_pass9[6]])
                q3_deg = np.rad2deg([q_pass9[1], q_pass9[4], q_pass9[7]])
                print(f"[try ] t={t:.3f}→{t_try:.3f} step={step:.3f} halves={halves} "
                      f"| res_core={res_core:.3e} | q2(deg)={q2_deg} | q3(deg)={q3_deg} | same_cluster={same_cluster}")

            # 通过条件：几何残差小 + 同簇
            if sol.success and same_cluster:
                # 检查 q3 是否非法
                q3_vals = [q_pass9[1], q_pass9[4], q_pass9[7]]
                if any(val > 1e-8 for val in q3_vals):   # 容差 1e-8 rad
                    if verbose:
                        print(f"[warn] q3>0 detected at t={t_try:.3f}, "
                              f"q3(rad)={q3_vals}, q3(deg)={np.rad2deg(q3_vals)} "
                              f"→ 继续，但几何上不符合真实构型")

                ok = True
                accepted_q_pass9 = q_pass9.copy()
                accepted_res_core = res_core
                q_prev6 = sol.x.copy()    # 更新 6 维锚点
                break

            # —— 缩步重试：不推进 t，本轮只减小 step —— #
            halves += 1
            step *= 0.5
            t_try = min(1.0, t + step)
            q_act_try = (1.0 - t_try) * q_act_start + t_try * q_act_goal
            q_act_rad = np.deg2rad(q_act_try)
            core_fun = ops.make_residual(q_act_rad)

        if not ok:
            raise RuntimeError(f"Continuation failed near t~{t:.3f}→{t_try:.3f}")

        # —— 接受该步：推进 t，并记录输出（9 维，用于绘图）—— #
        t = t_try
        q_pass_list.append(np.rad2deg(accepted_q_pass9))
        q_act_list.append(q_act_try.copy())

        if verbose:
            print(f"[step] t={t:.3f}, step={step:.3f}, res_core={accepted_res_core:.3e}")

        # —— 步长自适应：仅在成功且未缩步时放大 —— #
        if halves == 0:
            step = min(step * 1.0, 1.0 - t)

    return np.array(q_act_list), np.array(q_pass_list)

def chain_full_for_plot(i, q1,q2,q3,q4, side, axis_to_center_deg, link_params, h):
    L1,L2,L3 = link_params
    T0s = _make_T0s(side,h,axis_to_center_deg)
    T0 = T0s[i]
    d = L1 + L2
    A1  = _A1_fast(q1)
    A2d = _A2d_fast(q2, d)
    A2a = _A2a_fast(L3)
    A3  = _A3_fast(q3, L3)
    A4  = _A4_fast(q4, d)

    T1  = T0 @ A1
    T2d = T1 @ A2d
    T2a = T2d @ A2a
    T3  = T2a @ A3
    T4  = T3 @ A4
    return T0,T1,T2d,T2a,T3,T4

def animate_path_continuation(q_act_start_deg, q_act_goal_deg,
                              side=93.16, axis_to_center_deg=-16.0,
                              link_params=(39.5,34.1,5),
                              lam_follow_base=1e-6,
                              h=10.0, save_path=None,
                              n_show_frames=10):
    """
    使用 continue_solve_path 生成解，再逐帧绘制动画。
    - lam_follow: 正则权重（让解跟随上一步，避免跳解）
    - n_show_frames: 最终显示的帧数（从完整路径里等间隔采样）
    """
    # === 先解完整路径 ===
    t0 = time.time()
    q_act_hist, q_pass_hist = continue_solve_path(
        q_act_start_deg, q_act_goal_deg,
        side=side, axis_to_center_deg=axis_to_center_deg,
        link_params=link_params,
        verbose=True
    )
    t1 = time.time()
    print(f"[timing] continue_solve_path took {t1 - t0:.3f} seconds")

    total_frames = len(q_act_hist)
    if n_show_frames > total_frames:
        n_show_frames = total_frames

    # 计算等间隔索引
    idxs = np.linspace(0, total_frames-1, n_show_frames, dtype=int)
    q_act_hist = q_act_hist[idxs]
    q_pass_hist = q_pass_hist[idxs]

    # === 动画绘制 ===
    fig = plt.figure(figsize=(12, 10))   # 原本默认大约是(6,4)
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.cla()
        q_act = np.deg2rad(q_act_hist[frame])
        q_pass = np.deg2rad(q_pass_hist[frame])

        colors = {"link1":"red", "link2":"green", "link3":"blue", "link4":"purple"}
        A_pts, B_pts = [], []

        for i in range(3):
            q1 = q_act[i]
            q2, q3, q4 = q_pass[3*i:3*i+3]
            T0,T1,T2d,T2a,T3,T4 = chain_full_for_plot(
                i,q1,q2,q3,q4, side, axis_to_center_deg, link_params, h)

            A_pts.append(T0[:3,3])
            B_pts.append(T4[:3,3])
            ax.plot([T0[0,3],T1[0,3]],[T0[1,3],T1[1,3]],[T0[2,3],T1[2,3]], color=colors["link1"])
            ax.plot([T1[0,3],T2d[0,3],T2a[0,3]],
                    [T1[1,3],T2d[1,3],T2a[1,3]],
                    [T1[2,3],T2d[2,3],T2a[2,3]], color=colors["link2"])
            ax.plot([T2a[0,3],T3[0,3]],[T2a[1,3],T3[1,3]],[T2a[2,3],T3[2,3]], color=colors["link3"])
            ax.plot([T3[0,3],T4[0,3]],[T3[1,3],T4[1,3]],[T3[2,3],T4[2,3]], color=colors["link4"])

        # 三角形轮廓
        A_loop = np.vstack([A_pts, A_pts[0]])
        B_loop = np.vstack([B_pts, B_pts[0]])
        ax.plot(A_loop[:,0],A_loop[:,1],A_loop[:,2], 'k--')
        ax.plot(B_loop[:,0],B_loop[:,1],B_loop[:,2], 'gray')

        ax.set_xlim(-100,100); ax.set_ylim(-100,100); ax.set_zlim(0,150)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_title(f"Frame {frame+1}/{len(q_act_hist)}")

    ani = animation.FuncAnimation(fig, update, frames=len(q_act_hist),
                                  interval=300, repeat=False)

    if save_path:
        ani.save(save_path, writer="ffmpeg", fps=5)
    else:
        plt.show()

if __name__ == "__main__":
    q1_1, q1_2, q1_3 = 0, 0, 0
    q1_all = np.random.uniform(0.0, 90.0, size=3)
    q1_1, q1_2, q1_3 = q1_all # 随机生成
    q_start = [135, 135, 135]
    q_goal  = [120.0, 100.0 ,160.0]

    animate_path_continuation(
    q_start, q_goal,
    side=93.16,
    axis_to_center_deg=-16.0,
    link_params=(39.5,34.1,5),
    n_show_frames=10,   # 只显示10帧
    save_path=None
    )

    q_start = [135, 135, 135]
    q_goal  = [160.0, 120.0 ,100.0]

    animate_path_continuation(
    q_start, q_goal,
    side=93.16,
    axis_to_center_deg=-16.0,
    link_params=(39.5,34.1,5),
    n_show_frames=10,   # 只显示10帧
    save_path=None
    )

    q_start = [135, 135, 135]
    q_goal  = [100.0, 160.0 ,120.0]

    animate_path_continuation(
    q_start, q_goal,
    side=93.16,
    axis_to_center_deg=-16.0,
    link_params=(39.5,34.1,5),
    n_show_frames=10,   # 只显示10帧
    save_path=None
    )


