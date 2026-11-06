import pandas as pd
import numpy as np
import os

def detect_step(series, tol=1e-6):
    vals = np.sort(series.unique().astype(np.float64))
    diffs = np.diff(vals)
    diffs = diffs[diffs > tol]
    step = np.median(diffs)
    step = round(step, 3)
    return step

def fix_nan_boundary(arr, q1_range, q2_range, q3_range):
    nan_before = np.isnan(arr).any(axis=-1).sum()
    print(f"[fix] 修复前 NaN = {nan_before}")

    n1, n2, n3, F = arr.shape
    idxs = np.argwhere(np.isnan(arr).any(axis=-1))

    for (i, j, k) in idxs:
        # 可能的邻居候选（往里一格）
        neighbors = []
        if i == 0: neighbors.append((i+1, j, k))
        if i == n1-1: neighbors.append((i-1, j, k))
        if j == 0: neighbors.append((i, j+1, k))
        if j == n2-1: neighbors.append((i, j-1, k))
        if k == 0: neighbors.append((i, j, k+1))
        if k == n3-1: neighbors.append((i, j, k-1))

        # 尝试拷贝第一个非 NaN 邻居
        for ni, nj, nk in neighbors:
            if not np.isnan(arr[ni, nj, nk, :]).any():
                arr[i, j, k, :] = arr[ni, nj, nk, :]
                break
    nan_after = np.isnan(arr).any(axis=-1).sum()
    print(f"[fix] 修复后 NaN = {nan_after}")
    return arr

def csv_to_npz_fast(csv_file):
    dtypes = {
        "q1_deg":"float32","q2_deg":"float32","q3_deg":"float32",
        "q2_1_deg":"float32","q3_1_deg":"float32",
        "q2_2_deg":"float32","q3_2_deg":"float32",
        "q2_3_deg":"float32","q3_3_deg":"float32",
        "eul_z_deg":"float32","eul_y_deg":"float32","eul_x_deg":"float32",
        "Cx_mm":"float32","Cy_mm":"float32","Cz_mm":"float32",
        "success":"float32","res_norm":"float32",
    }
    usecols = list(dtypes.keys())
    df = pd.read_csv(csv_file, usecols=usecols, dtype=dtypes, engine="c")

    step1 = detect_step(df["q1_deg"])
    step2 = detect_step(df["q2_deg"])
    step3 = detect_step(df["q3_deg"])
    if not (step1 == step2 == step3):
        raise ValueError(f"步长不一致: q1={step1}, q2={step2}, q3={step3}")
    step = step1
    print(f"[info] detected step = {step}°")

    q1_min, q1_max = float(df["q1_deg"].min()), float(df["q1_deg"].max())
    q2_min, q2_max = float(df["q2_deg"].min()), float(df["q2_deg"].max())
    q3_min, q3_max = float(df["q3_deg"].min()), float(df["q3_deg"].max())

    q1_range = np.arange(q1_min, q1_max + 1e-9, step, dtype=np.float32)
    q2_range = np.arange(q2_min, q2_max + 1e-9, step, dtype=np.float32)
    q3_range = np.arange(q3_min, q3_max + 1e-9, step, dtype=np.float32)
    n1, n2, n3 = len(q1_range), len(q2_range), len(q3_range)

    features = [
        "q2_1_deg","q3_1_deg","q2_2_deg","q3_2_deg","q2_3_deg","q3_3_deg",
        "eul_z_deg","eul_y_deg","eul_x_deg","Cx_mm","Cy_mm","Cz_mm",
        "success","res_norm"
    ]
    F = len(features)

    arr = np.full((n1, n2, n3, F), np.nan, dtype=np.float32)
    arr_flat = arr.reshape(-1, F)

    idx1 = np.rint((df["q1_deg"].to_numpy() - q1_min) / step).astype(np.int64)
    idx2 = np.rint((df["q2_deg"].to_numpy() - q2_min) / step).astype(np.int64)
    idx3 = np.rint((df["q3_deg"].to_numpy() - q3_min) / step).astype(np.int64)

    flat_idx = (idx1 * n2 + idx2) * n3 + idx3
    vals = df[features].to_numpy(dtype=np.float32, copy=False)
    arr_flat[flat_idx] = vals

    # === 边界补洞 ===
    arr = fix_nan_boundary(arr, q1_range, q2_range, q3_range)


    base, _ = os.path.splitext(csv_file)
    npz_file = base + ".npz"
    np.savez_compressed(
        npz_file,
        arr=arr,
        q1_range=q1_range, q2_range=q2_range, q3_range=q3_range,
        step=np.float32(step),
        features=np.array(features),
    )
    print(f"[done] saved to {npz_file}, shape={arr.shape}, step={step}")

if __name__ == "__main__":
    csv_to_npz_fast("canfield_lookup_pose_0p4deg_full.csv")
