import numpy as np

def compare_npz(file1, file2, atol=1e-6, rtol=1e-6):
    data1 = np.load(file1, allow_pickle=True)
    data2 = np.load(file2, allow_pickle=True)

    # === 1. 基本信息 ===
    print(f"== Comparing {file1} vs {file2} ==")
    keys1, keys2 = set(data1.keys()), set(data2.keys())
    if keys1 != keys2:
        print("⚠️ Keys differ:", keys1.symmetric_difference(keys2))

    # === 2. Step ===
    step1, step2 = float(data1["step"]), float(data2["step"])
    print(f"Step1={step1}, Step2={step2}, Δ={abs(step1-step2)}")

    # === 3. Range ===
    for k in ["q1_range", "q2_range", "q3_range"]:
        r1, r2 = data1[k], data2[k]
        if r1.shape != r2.shape:
            print(f"{k} shape mismatch: {r1.shape} vs {r2.shape}")
        else:
            diff = np.max(np.abs(r1 - r2))
            print(f"{k} max abs diff = {diff:.3e}")

    # === 4. Features ===
    f1, f2 = data1["features"].tolist(), data2["features"].tolist()
    if f1 != f2:
        print("⚠️ Features differ:", f1, f2)
    else:
        print(f"Features identical ({len(f1)})")

    # === 5. 主数组 arr ===
    arr1, arr2 = data1["arr"], data2["arr"]
    if arr1.shape != arr2.shape:
        print(f"⚠️ arr shape mismatch: {arr1.shape} vs {arr2.shape}")
    else:
        mask = ~(np.isnan(arr1) | np.isnan(arr2))  # 两边都非 NaN 的位置
        diff = np.abs(arr1 - arr2)[mask]
        if diff.size == 0:
            print("Both arrays all-NaN (??)")
        else:
            max_diff = diff.max()
            mean_diff = diff.mean()
            n_diff = np.sum(diff > atol + rtol * np.abs(arr1[mask]))
            print(f"arr comparison: max={max_diff:.3e}, mean={mean_diff:.3e}, "
                  f"num_diff={n_diff}/{mask.sum()}")

if __name__ == "__main__":
    # 修改这里：传入两个 npz 文件路径
    compare_npz("canfield_lookup_pose_0p4deg_full.npz",
                "canfield_lookup_pose_0p4deg_full copy.npz")
