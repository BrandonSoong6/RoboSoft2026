import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree

# --- 插值工具 (简化版，完整表格直接查表) ---
class WorkspaceInterpolator:
    def __init__(self, df,
                 cols_angles=('q1_deg','q2_deg','q3_deg'),
                 cols_eul=('eul_z_deg','eul_y_deg','eul_x_deg'),
                 cols_pos=('Cx_mm','Cy_mm','Cz_mm')):
        self.df = df.copy()
        self.cols_angles = cols_angles
        self.cols_eul = cols_eul
        self.cols_pos = cols_pos

        # 建 KDTree：以角度空间 (q1,q2,q3) 为索引
        A = df[list(cols_angles)].to_numpy()
        self.kdtree = cKDTree(A)

    def lookup(self, q_act_deg, k=8):
        q = np.asarray(q_act_deg, dtype=float)

        # 查最近邻
        dists, idxs = self.kdtree.query(q, k=k)

        if np.min(dists) < 1e-9:  # 完全命中
            row = self.df.iloc[idxs if np.isscalar(idxs) else idxs[0]]
            C = row[list(self.cols_pos)].to_numpy()
            E = row[list(self.cols_eul)].to_numpy()
            R_end = R.from_euler('zyx', E, degrees=True).as_matrix()
        else:
            weights = 1.0 / dists
            weights /= weights.sum()

            # 插值位置
            C_all = self.df.iloc[idxs][list(self.cols_pos)].to_numpy()
            C = np.average(C_all, axis=0, weights=weights)

            # 插值姿态
            E_all = self.df.iloc[idxs][list(self.cols_eul)].to_numpy()
            R_all = R.from_euler('zyx', E_all, degrees=True)
            R_end = R_all.mean(weights=weights).as_matrix()

        eul = R.from_matrix(R_end).as_euler('zyx', degrees=True)
        return C, eul


# 使用示例
df = pd.read_csv("canfield_lookup_full.csv")
interp = WorkspaceInterpolator(df)

q_act = (41.85 + 90, 34.81 + 90,10.02 + 90)   # 任意三舵机角
C_mm, eul_zyx_deg = interp.lookup(q_act, k=8)

print("C(mm) =", C_mm)
print("eul (zyx,deg) =", eul_zyx_deg)
