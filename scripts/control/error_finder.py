import os
import numpy as np
from scipy.spatial import cKDTree

def compute_tracking_error(folder=".", prefix="traj_meas_demo1_final_", file_n=3):
    """
    统计所有轨迹的整体平均误差与最大误差（单位：mm）。
    不绘图，只输出数值。
    """
    all_errors = []

    for i in range(1, file_n + 1):
        fname = os.path.join(folder, f"{prefix}{i}.npz")
        if not os.path.exists(fname):
            print(f"⚠️ 找不到文件 {fname}，跳过。")
            continue

        data = np.load(fname)
        if "traj_meas" not in data or "spline_points" not in data:
            print(f"⚠️ 文件 {fname} 缺少关键数据字段，跳过。")
            continue

        traj = data["traj_meas"]
        spline = data["spline_points"]

        # 建立 KD 树并计算距离误差
        tree = cKDTree(spline)
        dists, _ = tree.query(traj)
        all_errors.append(dists)

        mean_err = np.mean(dists)
        max_err = np.max(dists)
        print(f"✅ {os.path.basename(fname)}: 平均误差 = {mean_err:.3f} mm, 最大误差 = {max_err:.3f} mm")

    # 合并所有误差
    if not all_errors:
        print("❌ 没有有效数据。")
        return None

    all_errors = np.concatenate(all_errors)
    mean_all = np.mean(all_errors)
    max_all = np.max(all_errors)

    print("\n=== 📊 整体误差统计 ===")
    print(f"平均误差（所有轨迹点） = {mean_all:.3f} mm")
    print(f"最大误差（所有轨迹点） = {max_all:.3f} mm")

    return mean_all, max_all


if __name__ == "__main__":
    compute_tracking_error(folder=".", prefix="traj_meas_demo1_final_", file_n=3)
