import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

# ----------------- 数据加载 -----------------
def load_fk_npz(fk_npz):
    d = np.load(fk_npz, allow_pickle=True)
    arr = d["arr"]
    q1_range, q2_range, q3_range = d["q1_range"], d["q2_range"], d["q3_range"]
    step = float(d["step"])
    return arr, q1_range, q2_range, q3_range, step

def load_ik_npz(ik_npz):
    d = np.load(ik_npz, allow_pickle=True)
    pos = d["pos"].astype(np.float64)
    quat = d["quat"].astype(np.float64)
    angles = d["angles"].astype(np.float64)
    tree = cKDTree(pos)  # 仅位置建树
    return tree, angles, pos, quat

# ----------------- FK 三线性插值（位置 + 姿态）-----------------
def fk_lookup_interp(q, arr, q1_range, q2_range, q3_range, step):
    q1, q2, q3 = q
    def idx_frac(qv, q_range):
        i = (qv - q_range[0]) / step
        j = int(np.floor(i))
        t = i - j
        j = np.clip(j, 0, len(q_range)-2)
        return j, t
    i1,t1 = idx_frac(q1, q1_range)
    i2,t2 = idx_frac(q2, q2_range)
    i3,t3 = idx_frac(q3, q3_range)

    vals_pos = arr[i1:i1+2, i2:i2+2, i3:i3+2, 9:12]   # (Cx,Cy,Cz)
    vals_eul = arr[i1:i1+2, i2:i2+2, i3:i3+2, 6:9]    # (z,y,x)

    def trilerp(vals):
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
    pos = trilerp(vals_pos)
    eul = trilerp(vals_eul)
    quat = R.from_euler("zyx", eul, degrees=True).as_quat()
    return pos, quat

def quat_angle_error(q1, q2):
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    q2 = q2 / (np.linalg.norm(q2) + 1e-12)
    dot = np.clip(abs(np.dot(q1, q2)), -1.0, 1.0)
    return 2 * np.arccos(dot) * 180/np.pi

# ----------------- IK：KNN + 同伦门控 + IDW 平滑 -----------------
def ik_with_homotopy_idw(tree, angles, pos_table, quat_table,
                         pos_target, q_curr,
                         K=36,          # 近邻数量
                         tau0=1.2,      # 初始同伦阈值
                         tau_step=0.4,  # 放宽步长
                         tau_max=8.0,  # 最大放宽
                         sigma_p=1.0,    # 位置归一化（mm）
                         eps=1e-6,
                         m=2):   # ★ 新增参数：均值时只取最近 m 个
    # 1) K 近邻（按位置）
    dist, idxs = tree.query(pos_target, k=K)
    idxs = np.atleast_1d(idxs)

    # 2) 同伦门控
    tau = tau0
    sel = None
    dq_inf_all = np.max(np.abs(angles[idxs] - q_curr), axis=1)
    while True:
        mask = dq_inf_all <= tau
        if np.any(mask):
            sel = idxs[mask]
            break
        tau += tau_step
        if tau > tau_max:
            sel = idxs
            break

    # 3A) 在选中集合上做 IDW
    pos_sel = pos_table[sel]
    q_sel   = angles[sel]

    d_task = np.linalg.norm((pos_sel - pos_target) / sigma_p, axis=1)

    # === 新增：只取最近的 m 个 ===
    if len(d_task) > m:
        keep_idx = np.argsort(d_task)[:m]
        pos_sel = pos_sel[keep_idx]
        q_sel   = q_sel[keep_idx]
        d_task  = d_task[keep_idx]

    w = 1.0 / (d_task + eps)**2
    w_sum = np.sum(w)
    if w_sum < 1e-12:
        best_idx = sel[np.argmin(d_task)]
        return angles[best_idx], pos_table[best_idx], quat_table[best_idx], {
            "tau_used": tau, "n_sel": len(sel), "mode": "nn_fallback"
        }

    q_blend = (w[:, None] * q_sel).sum(axis=0) / w_sum
    q_blend = np.clip(q_blend, 90.0, 160.0)

    return q_blend, None, None, {"tau_used": tau, "n_sel": len(sel), "mode": "idw_msubset"}

# ----------------- 评测：误差 Top-K -----------------
def cross_check_top_errors(fk_npz, ik_npz, n_tests=1000, top_k=10,
                           K=10, tau0=2.0, tau_step=1.0, tau_max=10.0,
                           sigma_p=1.0):
    arr, q1r, q2r, q3r, step = load_fk_npz(fk_npz)
    tree, angles, pos_table, quat_table = load_ik_npz(ik_npz)

    records = []
    for _ in range(n_tests):
        # 随机目标（同时当做“当前角度”以模拟闭环）
        q_true = np.array([
            np.random.uniform(q1r[0], q1r[-1]),
            np.random.uniform(q2r[0], q2r[-1]),
            np.random.uniform(q3r[0], q3r[-1]),
        ])
        pos_true, quat_true = fk_lookup_interp(q_true, arr, q1r, q2r, q3r, step)

        # IK（KNN + 同伦 + IDW）
        q_ik, pos_ik_nn, quat_ik_nn, meta = ik_with_homotopy_idw(
            tree, angles, pos_table, quat_table,
            pos_true, q_true,
            K=K, tau0=tau0, tau_step=tau_step, tau_max=tau_max,
            sigma_p=sigma_p
        )

        # 用 FK 连续出最终的 (pos, quat)
        pos_fk_from_ik, quat_fk_from_ik = fk_lookup_interp(q_ik, arr, q1r, q2r, q3r, step)

        # 误差统计（与真实点比较）
        err_angles = np.abs(q_true - q_ik)
        err_pos_fk = np.linalg.norm(pos_true - pos_fk_from_ik)
        err_rot_fk = quat_angle_error(quat_true, quat_fk_from_ik)

        # 仅用于观察（“最近邻候选”的直接误差）
        if pos_ik_nn is None:
            # 我们选择了 IDW，最近邻误差仅用于展示：取 KDTree 最近的一个
            _, idx0 = tree.query(pos_true, k=1)
            pos_ik_nn = pos_table[idx0]
            quat_ik_nn = quat_table[idx0]
        else:
            quat_ik_nn = quat_ik_nn

        err_pos_direct = np.linalg.norm(pos_true - pos_ik_nn)
        err_rot_direct = quat_angle_error(quat_true, quat_ik_nn)

        records.append((
            err_angles.max(), err_angles, q_true, q_ik,
            pos_true, pos_ik_nn, pos_fk_from_ik,
            quat_true, quat_ik_nn, quat_fk_from_ik,
            err_pos_direct, err_pos_fk,
            err_rot_direct, err_rot_fk,
            meta
        ))

    # 按角度最大误差排序
    records.sort(key=lambda x: -x[0])

    print(f"\n误差最大的 {top_k} 条:")
    for i in range(top_k):
        (max_err, err_angles, q_true, q_ik,
         pos_true, pos_ik, pos_fk_from_ik,
         quat_true, quat_ik, quat_fk_from_ik,
         err_pos_direct, err_pos_fk,
         err_rot_direct, err_rot_fk,
         meta) = records[i]

        print(f"[{i+1}]  mode={meta['mode']}, tau_used={meta['tau_used']:.1f}°, n_sel={meta['n_sel']}")
        print(f"  max_angle_err = {max_err:.2f}°")
        print(f"  angle_err     = [{err_angles[0]:.2f}, {err_angles[1]:.2f}, {err_angles[2]:.2f}]°")
        print(f"  pos_err       = IKtable(NN): {err_pos_direct:.3f} mm,  FK@IK(IDW): {err_pos_fk:.3f} mm")
        print(f"  rot_err       = IKtable(NN): {err_rot_direct:.2f}°,    FK@IK(IDW): {err_rot_fk:.2f}°")
        print(f"  q_true        = {np.round(q_true,2)}")
        print(f"  q_ik(IDW)     = {np.round(q_ik,2)}")
        print(f"  pos_true      = {np.round(pos_true,3)}")
        print(f"  pos_fk_from_ik= {np.round(pos_fk_from_ik,3)}")
        print(f"  quat_true     = {np.round(quat_true,4)}")
        print(f"  quat_fk_from_ik= {np.round(quat_fk_from_ik,4)}")
        print()

    return records


if __name__ == "__main__":
    # cross_check_top_errors(
    #     "canfield_lookup_pose_0p4deg_full.npz",
    #     "canfield_ik_0p4deg_lookup.npz",
    #     n_tests=50000,
    #     top_k=10,
    #     K=36,          # 近邻数量
    #     tau0=1.2,      # 初始同伦阈值
    #     tau_step=0.4,  # 放宽步长
    #     tau_max=8.0,  # 最大放宽
    #     sigma_p=1.0    # 位置归一化（mm）
    # )
    tree, angles, pos_table, quat_table = load_ik_npz("canfield_ik_0p4deg_lookup.npz")
    q_ik, _, _, meta = ik_with_homotopy_idw(
            tree, angles, pos_table, quat_table,
            (  1.01481346 , -1.30911208 ,114.93005727), (127.89, 128.51, 126.84),
                K=36,          # 近邻数量
                tau0=1.2,      # 初始同伦阈值
                tau_step=0.4,  # 放宽步长
                tau_max=8.0,  # 最大放宽
                sigma_p=1.0    # 位置归一化（mm）
            )
    print(q_ik)
    arr, q1r, q2r, q3r, step = load_fk_npz("canfield_lookup_pose_0p4deg_full.npz")
    pos_fk, quat_fk = fk_lookup_interp((127.89, 128.51 ,126.84), arr, q1r, q2r, q3r, step)
    print(pos_fk)

# === Loop Step ===
# 视觉末端 (mm): [ -1.42032968   5.56750545 190.37761087]
# 模型中心 (mm): [  0.99379697  -0.9807368  115.99893782]
# 目标点 (mm): [  0.   0. 170.]
# PID 输出 (mm): [ 0.02101648 -0.32837527 -1.06888054]
# p_IK_W_mm (mm): [  1.01481346  -1.30911208 114.93005727]
# 当前角度 (deg): [127.89 128.51 126.84]
# [PC] Sent: sync 41.2 42.0 40.4