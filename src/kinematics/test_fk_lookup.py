import numpy as np

def check_fk_success(npz_file):
    d = np.load(npz_file, allow_pickle=True)
    arr = d["arr"]
    vals = arr.reshape(-1, arr.shape[-1])

    success = vals[:, 12]  # 第12列是 success
    n_total = success.shape[0]
    n_nan = np.isnan(success).sum()
    n_inf = np.isinf(success).sum()

    print(f"文件: {npz_file}")
    print(f"总条目数: {n_total}")
    print(f"NaN 数量: {n_nan}")
    print(f"Inf 数量: {n_inf}")
    print(f"正常数量: {n_total - n_nan - n_inf}")

    if n_nan > 0 or n_inf > 0:
        idx_bad = np.where(~np.isfinite(success))[0][:10]
        print("前10个非法 success 索引:", idx_bad)

if __name__ == "__main__":
    check_fk_success("canfield_lookup_pose_0p4deg_full.npz")
