# canfield_core.py
import numpy as np
from scipy.optimize import least_squares
import warnings
import time
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

# ========= 加速算子 =========
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

def _A2d_fast(q2, d):
    ct, st = np.cos(q2), np.sin(q2)
    return np.array([[ct, 0.0, -st, 0.0],
                     [st, 0.0,  ct, 0.0],
                     [0.0,-1.0, 0.0,   d],
                     [0.0, 0.0, 0.0, 1.0]])

def _A2a_fast(L3):
    return np.array([[1.0, 0.0, 0.0, L3],
                     [0.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])

def _A3_fast(q3, L3):
    ct, st = np.cos(q3), np.sin(q3)
    return np.array([[ct, 0.0, -st, L3*ct],
                     [st, 0.0,  ct, L3*st],
                     [0.0,-1.0, 0.0,    0.0],
                     [0.0, 0.0, 0.0,    1.0]])

def _A4_fast(q4, d):
    ct, st = np.cos(q4), np.sin(q4)
    return np.array([[ct, 0.0,  st, 0.0],
                     [st, 0.0, -ct, 0.0],
                     [0.0, 1.0, 0.0,   d],
                     [0.0, 0.0, 0.0, 1.0]])

def _chain_end_fast(T0, q1, q2, q3, q4, L1, L2, L3, out_B, out_z):
    d = L1 + L2
    T1  = T0 @ _A1_fast(q1)
    T2d = T1 @ _A2d_fast(q2, d)
    T2a = T2d @ _A2a_fast(L3)
    T3  = T2a @ _A3_fast(q3, L3)
    T4  = T3 @ _A4_fast(q4, d)
    out_B[:] = T4[:3, 3]
    out_z[:] = T4[:3, 2]

if _HAS_NUMBA:
    _A1_fast  = njit(cache=True, fastmath=True)(_A1_fast)
    _A2d_fast = njit(cache=True, fastmath=True)(_A2d_fast)
    _A2a_fast = njit(cache=True, fastmath=True)(_A2a_fast)
    _A3_fast  = njit(cache=True, fastmath=True)(_A3_fast)
    _A4_fast  = njit(cache=True, fastmath=True)(_A4_fast)

# ========= 运算核心 =========
class CanfieldOps:
    def __init__(self, side=93.16, axis_to_center_deg=-16.0, link_params=(39.5,34.1,5), h=10.0):
        self.side = float(side)
        self.axis_to_center_deg = float(axis_to_center_deg)
        self.h = float(h)
        self.L1, self.L2, self.L3 = map(float, link_params)
        self.T0s = _make_T0s(self.side, self.h, self.axis_to_center_deg)
        self.target = np.deg2rad(self.axis_to_center_deg)
        self._B = np.empty((3,3), dtype=float)
        self._z = np.empty((3,3), dtype=float)
        self._r = np.empty(9, dtype=float)

    def make_residual(self, q_actives_rad):
        q_act = np.asarray(q_actives_rad, dtype=float)
        def residual(q_pass6):
            q_pass = np.zeros(9)
            for i in range(3):
                q2,q3 = q_pass6[2*i:2*i+2]
                q4 = -q2
                q_pass[3*i:3*i+3] = [q2,q3,q4]
            for i in range(3):
                q2,q3,q4 = q_pass[3*i:3*i+3]
                _chain_end_fast(self.T0s[i], q_act[i], q2, q3, q4,
                                self.L1, self.L2, self.L3,
                                self._B[i], self._z[i])
            B1,B2,B3 = self._B; z41,z42,z43 = self._z
            v21, v32, v13 = B2-B1, B3-B2, B1-B3
            n = np.cross(v21,v32); n/=np.linalg.norm(n)+1e-15
            if n[2]<0: n=-n
            C=(B1+B2+B3)/3
            r1=(B1-C)/np.linalg.norm(B1-C); r2=(B2-C)/np.linalg.norm(B2-C); r3=(B3-C)/np.linalg.norm(B3-C)
            self._r[0]=np.linalg.norm(v21)-self.side
            self._r[1]=np.linalg.norm(v32)-self.side
            self._r[2]=np.linalg.norm(v13)-self.side
            self._r[3]=np.dot(n,z41); self._r[4]=np.dot(n,z42); self._r[5]=np.dot(n,z43)
            def signed_ang(u,v,n): return np.arctan2(np.dot(n,np.cross(u,v)), np.dot(u,v))
            a1=signed_ang(z41-n*np.dot(z41,n),r1,n)
            a2=signed_ang(z42-n*np.dot(z42,n),r2,n)
            a3=signed_ang(z43-n*np.dot(z43,n),r3,n)
            def ang_err(angle,target):
                e=(angle-target+np.pi)%(2*np.pi)-np.pi
                if e>np.pi/2: e-=np.pi
                if e<-np.pi/2: e+=np.pi
                return e
            self._r[6]=ang_err(a1,self.target)
            self._r[7]=ang_err(a2,self.target)
            self._r[8]=ang_err(a3,self.target)
            return self._r.copy()
        return residual

def cluster_ok_fast(q_pass_rad, eps=1e-5):
    q2_vals=[q_pass_rad[3*i] for i in range(3)]
    if any(abs(v)<eps for v in q2_vals): return True
    signs=[np.sign(v) for v in q2_vals]
    return not(all(s>0 for s in signs) or all(s<0 for s in signs))

def solve_passives(q_actives_deg, side=93.16, axis_to_center_deg=-16.0, link_params=(39.5,34.1,5)):
    q0=np.deg2rad([0,-90,0,-90,0,-90])
    q_act_rad=np.deg2rad(q_actives_deg)
    ops=CanfieldOps(side,axis_to_center_deg,link_params)
    fun=ops.make_residual(q_act_rad)
    sol=least_squares(fun,q0,method='trf',loss='linear',ftol=1e-10,xtol=1e-10,gtol=1e-10,max_nfev=8000)
    q_pass9=np.zeros(9)
    for i in range(3):
        q2,q3=sol.x[2*i:2*i+2]; q4=-q2
        q_pass9[3*i:3*i+3]=[q2,q3,q4]
    return q_pass9,sol

def continue_solve_path(q_act_start_deg,q_act_goal_deg,
                        side=93.16,axis_to_center_deg=-16.0,
                        link_params=(39.5,34.1,5),
                        max_halves=18,tol_success=1e-6,verbose=True):
    q_act_start=np.array(q_act_start_deg,float)
    q_act_goal=np.array(q_act_goal_deg,float)
    q_pass0,sol0=solve_passives(q_act_start,side,axis_to_center_deg,link_params)
    if verbose: print(f"[init] start_cost={sol0.cost:.3e}, success={sol0.success}")
    q_pass_list=[np.rad2deg(q_pass0)]
    q_act_list=[q_act_start.copy()]
    res_list=[sol0.cost]   # 保存每步残差
    t,step=0.0,1.0; q_prev6=sol0.x.copy()
    ops=CanfieldOps(side,axis_to_center_deg,link_params)
    while t<1-1e-12:
        t_try=min(1.0,t+step)
        q_act_try=(1-t_try)*q_act_start+t_try*q_act_goal
        q_act_rad=np.deg2rad(q_act_try)
        core_fun=ops.make_residual(q_act_rad)
        ok=False; halves=0; q0=q_prev6.copy()
        while halves<=max_halves:
            sol=least_squares(core_fun,q0,method='trf',loss='linear',
                              ftol=1e-14,xtol=1e-14,gtol=1e-14,max_nfev=4000)
            q_pass9=np.zeros(9)
            for i in range(3):
                q2,q3=sol.x[2*i:2*i+2]; q4=-q2
                q_pass9[3*i:3*i+3]=[q2,q3,q4]
            r_core = core_fun(sol.x)
            res_core = np.linalg.norm(r_core)
            if sol.success and cluster_ok_fast(q_pass9) and res_core <= tol_success:
                ok = True
                q_prev6 = sol.x.copy()
                accepted_q_pass9  = q_pass9.copy()
                accepted_res_core = res_core
                break
            halves+=1; step*=0.5
            t_try=min(1.0,t+step)
            q_act_try=(1-t_try)*q_act_start+t_try*q_act_goal
            q_act_rad=np.deg2rad(q_act_try)
            core_fun=ops.make_residual(q_act_rad)
        if not ok: raise RuntimeError(f"fail near t~{t:.3f}")
        t=t_try
        q_pass_list.append(np.rad2deg(accepted_q_pass9))
        q_act_list.append(q_act_try.copy())
        res_list.append(accepted_res_core)
        if verbose: print(f"[step] t={t:.3f}, step={step:.3f}, res={accepted_res_core:.3e}")
        if halves==0: step=min(step*2,1-t)
    q_act_arr=np.array(q_act_list)
    q_pass_arr=np.array(q_pass_list)[:,[0,1,3,4,6,7]]
    return q_act_arr,q_pass_arr,np.array(res_list)

# ========= 体积蛇形遍历（仅 q1 为最大那 1/3）=========

# ========= 体积蛇形遍历（仅 q1 为最大那 1/3）=========

def _iter_snake_keypoints(q_min=90, q_max=160, step=1.0, strict=False):
    """生成蛇形关键点序列（层内蛇形 + 层间贴合）。支持浮点步长。"""
    def admissible(q1, q2, q3):
        return (q1 > q2 and q1 > q3) if strict else (q1 >= q2 and q1 >= q3)

    Q = np.arange(q_min, q_max + 1e-9, step, dtype=float)
    last_q = None
    for q1 in Q:
        q2_range = np.arange(q_min, q1 + 1e-9, step, dtype=float)
        for i_q2, q2 in enumerate(q2_range):
            q3_line = np.arange(q_min, q1 + 1e-9, step, dtype=float)
            if i_q2 % 2 == 1:
                q3_line = q3_line[::-1]

            for q3 in q3_line:
                if not admissible(q1, q2, q3):
                    continue
                # 层间贴合
                if last_q is not None and q1 > last_q[0] and i_q2 == 0 and abs(q3 - q3_line[0]) < 1e-9:
                    q2 = min(last_q[1], q1 if not strict else q1 - step)
                    q3 = min(last_q[2], q1 if not strict else q1 - step)
                    if not admissible(q1, q2, q3):
                        continue
                yield (float(q1), float(q2), float(q3))
                last_q = (q1, q2, q3)

def _expand_pair(a, b, step=1, strict=False):
    def clamp(q1, q2, q3):
        if strict:
            q2 = min(q2, q1 - step); q3 = min(q3, q1 - step)
        else:
            q2 = min(q2, q1);        q3 = min(q3, q1)
        return q1, q2, q3

    q1, q2, q3 = a
    q1b, q2b, q3b = b

    # 先调 q1
    while abs(q1 - q1b) >= step - 1e-12:
        q1 += np.sign(q1b - q1) * step
        q1, q2, q3 = clamp(q1, q2, q3)
        yield (q1, q2, q3)
    # 再调 q2
    while abs(q2 - q2b) >= step - 1e-12:
        q2 += np.sign(q2b - q2) * step
        q1, q2, q3 = clamp(q1, q2, q3)
        yield (q1, q2, q3)
    # 最后调 q3
    while abs(q3 - q3b) >= step - 1e-12:
        q3 += np.sign(q3b - q3) * step
        q1, q2, q3 = clamp(q1, q2, q3)
        yield (q1, q2, q3)

def iter_snake_branch1_q1max(q_min=90, q_max=160, step=1, strict=False):
    """与当前逻辑等价，但流式产出逐度相邻的点（不占大内存）。"""
    key_iter = _iter_snake_keypoints(q_min, q_max, step, strict)
    try:
        prev = next(key_iter)
    except StopIteration:
        return
    # 先把第一个关键点吐出来
    yield prev
    # 逐对关键点在线补间
    for nxt in key_iter:
        for p in _expand_pair(prev, nxt, step=step, strict=strict):
            yield p
        prev = nxt

def solve_passives_warm(q_actives_deg, q0_prev6, ops):
    """
    在目标点直接做一次最小二乘，初值用上一个点的 6 维被动角 (q2_1,q3_1,q2_2,q3_2,q2_3,q3_3)。
    返回：q_pass9 (rad), sol, res_norm
    """
    q_act_rad = np.deg2rad(q_actives_deg)
    fun = ops.make_residual(q_act_rad)
    # 适度放宽容差 & 限制迭代步，保证单次收敛速度
    sol = least_squares(fun, q0_prev6, method='trf', loss='linear',
                        ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=1000)
    q_pass9 = np.zeros(9)
    for i in range(3):
        q2, q3 = sol.x[2*i:2*i+2]
        q4 = -q2
        q_pass9[3*i:3*i+3] = [q2, q3, q4]
    res_norm = float(np.linalg.norm(fun(sol.x)))
    return q_pass9, sol, res_norm

def run_table_branch1_fast(outfile,
                           q_start_init=(135.0, 135.0, 135.0),
                           q_anchor=(90.0, 90.0, 90.0),
                           side=93.16, axis_to_center_deg=-16.0,
                           link_params=(39.5, 34.1, 5),
                           strict=False,
                           verbose=True,
                           flush_every=100000,
                           tol_success=1e-5,
                           max_halves=6,
                           step=1.0):   # <== 新增参数
    import csv, gzip, gc

    if verbose:
        print(f"[warmup] from {q_start_init} → {q_anchor} ...")

    # 只做一次：构造 ops
    ops = CanfieldOps(side, axis_to_center_deg, link_params)

    # 先在锚点建立初值
    q_pass9_anchor, sol_anchor = solve_passives(q_anchor, side, axis_to_center_deg, link_params)
    q0_prev6 = sol_anchor.x.copy()

    # === 预先展开迭代器，算总数 ===
    keypoints = list(iter_snake_branch1_q1max(90, 160, step, strict=strict))  # <== 用 step
    total_pts = len(keypoints)

    buf = []
    header = [
        "q1_deg","q2_deg","q3_deg",
        "q2_1_deg","q3_1_deg","q2_2_deg","q3_2_deg","q2_3_deg","q3_3_deg",
        "res_norm"
    ]
    total = 0
    t0 = time.time()

    prev_act = np.array(q_anchor, float)  # 上一个点的主动角

    for (q1, q2, q3) in keypoints:
        q_target = np.array([q1, q2, q3], float)

        step_frac = 1.0
        t = 0.0
        ok = False
        halves = 0
        q0 = q0_prev6.copy()

        while not ok and halves <= max_halves:
            t_try = min(1.0, t + step_frac)
            q_act_try = (1 - t_try) * prev_act + t_try * q_target
            try:
                q_pass9, sol, res = solve_passives_warm(q_act_try, q0, ops)
            except Exception:
                res = np.inf

            if np.isfinite(res) and res <= tol_success and cluster_ok_fast(q_pass9):
                ok = True
                q0 = sol.x.copy()
                q0_prev6 = q0
                q_pass_deg = np.rad2deg(q_pass9)[[0,1,3,4,6,7]]
                buf.append([
                    f"{q1:.1f}", f"{q2:.1f}", f"{q3:.1f}",
                    f"{q_pass_deg[0]:.6f}", f"{q_pass_deg[1]:.6f}",
                    f"{q_pass_deg[2]:.6f}", f"{q_pass_deg[3]:.6f}",
                    f"{q_pass_deg[4]:.6f}", f"{q_pass_deg[5]:.6f}",
                    f"{res:.6e}",
                ])
                total += 1
                prev_act = q_act_try
            else:
                step_frac *= 0.5
                halves += 1

        if not ok and verbose:
            print(f"[skip] ({q1:.1f},{q2:.1f},{q3:.1f}) res={res:.2e}")

        if verbose and (total % 1000 == 0):
            dt = time.time() - t0
            rate = total / dt if dt > 0 else 0.0
            percent = total / total_pts * 100
            print(
                f"\r[wip] {total:,}/{total_pts:,} ({percent:5.1f}%) | "
                f"current=({q1:.1f},{q2:.1f},{q3:.1f}) "
                f"res={res:.2e} | rate≈{rate:.1f} pts/s",
                end=""
            )

        if (total % flush_every) == 1 and buf:
            mode = "wt" if total == 1 else "at"
            with gzip.open(outfile, mode, newline="") as f:
                w = csv.writer(f)
                if total == 1:
                    w.writerow(header)
                w.writerows(buf)
            buf.clear()
            gc.collect()

    if buf:
        mode = "wt" if total == 0 else "at"
        with gzip.open(outfile, mode, newline="") as f:
            w = csv.writer(f)
            if total == 0:
                w.writerow(header)
            w.writerows(buf)

    if verbose:
        dt = time.time() - t0
        print(f"\n[done] total rows={total:,}/{total_pts:,}  elapsed={dt:.1f}s  file={outfile}")

# if __name__=="__main__":
#     n_trials = 1000
#     t0 = time.time()
#     for _ in range(n_trials):
#         q_start=[135,135,135]
#         q_goal  = np.random.uniform(90.0, 180.0, size=3)
#         # 只跑，不打印
#         q_act_hist, q_pass_hist, res_hist = continue_solve_path(
#             q_start, q_goal, verbose=False
#         )
#         print(f"Number of steps: {_}", end='\r')
#     t1 = time.time()

#     elapsed = t1 - t0
#     per_minute = n_trials / elapsed * 60.0
#     print(f"\nRan {n_trials} trials in {elapsed:.2f} seconds")
#     print(f"≈ {per_minute:.1f} continuations per minute")
if __name__ == "__main__":
    # 跑表（1° 分辨率）
    # run_table_branch1_fast(
    #     outfile="canfield_lookup.csv.gz",
    #     q_start_init=(135.0, 135.0, 135.0),
    #     q_anchor=(90.0, 90.0, 90.0),
    #     side=93.16, axis_to_center_deg=-16.0,
    #     link_params=(39.5, 34.1, 5),
    #     verbose=True,
    #     step=1.0
    # )

    # 跑表（0.5° 分辨率）
    run_table_branch1_fast(
        outfile="canfield_lookup_0p4deg.csv.gz",
        q_start_init=(135.0, 135.0, 135.0),
        q_anchor=(90.0, 90.0, 90.0),
        side=93.16, axis_to_center_deg=-16.0,
        link_params=(39.5, 34.1, 5),
        verbose=True,
        step=0.4
    )

