import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from numba import njit, prange

# ========= Numba 加速算子 =========
@njit(cache=True, fastmath=True)
def _A1_fast(q1):
    ct, st = np.cos(q1), np.sin(q1)
    return np.array([[ct, 0.0,  st, 0.0],
                     [st, 0.0, -ct, 0.0],
                     [0.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])

@njit(cache=True, fastmath=True)
def _A2d_fast(q2, d):
    ct, st = np.cos(q2), np.sin(q2)
    return np.array([[ct, 0.0, -st, 0.0],
                     [st, 0.0,  ct, 0.0],
                     [0.0,-1.0, 0.0,   d],
                     [0.0, 0.0, 0.0, 1.0]])

@njit(cache=True, fastmath=True)
def _A2a_fast(L3):
    return np.array([[1.0, 0.0, 0.0, L3],
                     [0.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])

@njit(cache=True, fastmath=True)
def _A3_fast(q3, L3):
    ct, st = np.cos(q3), np.sin(q3)
    return np.array([[ct, 0.0, -st, L3*ct],
                     [st, 0.0,  ct, L3*st],
                     [0.0,-1.0, 0.0,    0.0],
                     [0.0, 0.0, 0.0,    1.0]])

@njit(cache=True, fastmath=True)
def _A4_fast(q4, d):
    ct, st = np.cos(q4), np.sin(q4)
    return np.array([[ct, 0.0,  st, 0.0],
                     [st, 0.0, -ct, 0.0],
                     [0.0, 1.0, 0.0,   d],
                     [0.0, 0.0, 0.0, 1.0]])

@njit(cache=True, fastmath=True)
def _make_T0s(side, h, axis_to_center_deg):
    T0s = np.empty((3, 4, 4))
    R_ = side / np.sqrt(3.0)
    thetas = np.deg2rad(np.array([0.0, 120.0, 240.0]))
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

@njit(cache=True, fastmath=True)
def _chain_end_fast(T0, q1, q2, q3, q4, L1, L2, L3):
    d = L1 + L2
    T1  = T0 @ _A1_fast(q1)
    T2d = T1 @ _A2d_fast(q2, d)
    T2a = T2d @ _A2a_fast(L3)
    T3  = T2a @ _A3_fast(q3, L3)
    T4  = T3 @ _A4_fast(q4, d)
    B = T4[:3, 3]
    z = T4[:3, 2]
    return B, z

# ========= 批量计算 C 与 R_top =========
@njit(parallel=True, fastmath=True)
def batch_center_pose(q_act_deg_arr, q_pass_deg_arr,
                      side, axis_to_center_deg, L1, L2, L3, h):
    n = q_act_deg_arr.shape[0]
    C_out = np.zeros((n, 3))
    R_out = np.zeros((n, 3, 3))

    T0s = _make_T0s(side, h, axis_to_center_deg)

    for idx in prange(n):
        q_act_rad  = np.deg2rad(q_act_deg_arr[idx])
        q_pass_rad = np.deg2rad(q_pass_deg_arr[idx])

        B = np.zeros((3,3))
        for i in range(3):
            q1 = q_act_rad[i]
            q2, q3, q4 = q_pass_rad[3*i:3*i+3]
            Bi, _ = _chain_end_fast(T0s[i], q1, q2, q3, q4, L1, L2, L3)
            B[i,:] = Bi

        B1, B2, B3 = B
        C = (B1 + B2 + B3) / 3.0

        nvec = np.cross(B2 - B1, B3 - B1)
        nvec /= (np.linalg.norm(nvec) + 1e-12)
        if nvec[2] < 0:
            nvec = -nvec

        x_axis = (B1 - C)
        x_axis -= np.dot(x_axis, nvec) * nvec
        x_axis /= (np.linalg.norm(x_axis) + 1e-12)
        y_axis = np.cross(nvec, x_axis)
        R_top = np.column_stack((x_axis, y_axis, nvec))

        C_out[idx,:]   = C
        R_out[idx,:,:] = R_top

    return C_out, R_out

# ========= I/O 与拼装 =========
if __name__ == "__main__":
    df = pd.read_csv("canfield_lookup_0p4deg_unique.csv")

    # 仅保留被动角都存在的行，避免错位
    cols_pass = ["q2_1_deg","q3_1_deg","q2_2_deg","q3_2_deg","q2_3_deg","q3_3_deg"]
    mask = df[cols_pass].notna().all(axis=1)
    dfv = df.loc[mask].reset_index(drop=True)

    # 主动角 (q1,q2,q3) 这里名字来自原表，但它们是三舵机主动角
    q_act_deg_arr = dfv[["q1_deg","q2_deg","q3_deg"]].to_numpy(dtype=np.float64)

    # 三条支链分别用各自的 (q2_k, q3_k, q4_k=-q2_k)
    q_pass_deg_arr = np.stack([
        dfv["q2_1_deg"].to_numpy(np.float64),
        dfv["q3_1_deg"].to_numpy(np.float64),
        -dfv["q2_1_deg"].to_numpy(np.float64),
        dfv["q2_2_deg"].to_numpy(np.float64),
        dfv["q3_2_deg"].to_numpy(np.float64),
        -dfv["q2_2_deg"].to_numpy(np.float64),
        dfv["q2_3_deg"].to_numpy(np.float64),
        dfv["q3_3_deg"].to_numpy(np.float64),
        -dfv["q2_3_deg"].to_numpy(np.float64),
    ], axis=1)

    # 批量计算
    C_out, R_out = batch_center_pose(
        q_act_deg_arr, q_pass_deg_arr,
        side=93.16, axis_to_center_deg=-16.0,
        L1=39.5, L2=34.1, L3=5, h=10.0
    )

    # 欧拉角（zyx，单位度）
    eul_all = R.from_matrix(R_out).as_euler("zyx", degrees=True)

    # 拼结果，按你先前指定的新表头顺序输出
    # 拼结果，按新表头顺序输出，并保留每条支链的 q2,q3
    df_out = pd.DataFrame({
        # 主动角
        "q1_deg":     q_act_deg_arr[:,0],
        "q2_deg":     q_act_deg_arr[:,1],
        "q3_deg":     q_act_deg_arr[:,2],

        # 被动角（直接从 dfv 取，不要丢）
        "q2_1_deg":   dfv["q2_1_deg"].to_numpy(np.float64),
        "q3_1_deg":   dfv["q3_1_deg"].to_numpy(np.float64),
        "q2_2_deg":   dfv["q2_2_deg"].to_numpy(np.float64),
        "q3_2_deg":   dfv["q3_2_deg"].to_numpy(np.float64),
        "q2_3_deg":   dfv["q2_3_deg"].to_numpy(np.float64),
        "q3_3_deg":   dfv["q3_3_deg"].to_numpy(np.float64),

        # 姿态
        "eul_z_deg":  eul_all[:,0],
        "eul_y_deg":  eul_all[:,1],
        "eul_x_deg":  eul_all[:,2],

        # 末端中心
        "Cx_mm":      C_out[:,0],
        "Cy_mm":      C_out[:,1],
        "Cz_mm":      C_out[:,2],

        # 残差指标
        "success":    (dfv["res_norm"].to_numpy(np.float64) < 1e-3).astype(np.int32),
        "res_norm":   dfv["res_norm"].to_numpy(np.float64),
    })


    df_out.to_csv("canfield_lookup_0p4deg_pose.csv", index=False)
    skipped = len(df) - len(df_out)
    print(f"[done] wrote {len(df_out)} rows → canfield_lookup_pose.csv (skipped {skipped} rows with NaNs)")
