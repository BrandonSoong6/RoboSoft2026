import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

class InverseWorkspaceInterpolator:
    def __init__(self, df,
                 cols_angles=('q1_deg','q2_deg','q3_deg'),
                 cols_pos=('Cx_mm','Cy_mm','Cz_mm')):
        self.df = df.copy()
        self.cols_angles = cols_angles
        self.cols_pos = cols_pos

        # 建立 KDTree (位置空间)
        P = df[list(cols_pos)].to_numpy()
        self.kdtree = cKDTree(P)

    def lookup_pos(self, C_target, k=8):
        C_target = np.asarray(C_target, dtype=float)

        # 查最近邻
        dists, idxs = self.kdtree.query(C_target, k=k)

        if np.min(dists) < 1e-9:  # 完全命中
            row = self.df.iloc[idxs if np.isscalar(idxs) else idxs[0]]
            q = row[list(self.cols_angles)].to_numpy()
        else:
            weights = 1.0 / dists
            weights /= weights.sum()

            Q_all = self.df.iloc[idxs][list(self.cols_angles)].to_numpy()
            q = np.average(Q_all, axis=0, weights=weights)

        return q
    
# 加载完整表格（展开过的版本）
df = pd.read_csv("canfield_inverse_lookup.csv")

# 构造逆向插值器
inv_interp = InverseWorkspaceInterpolator(df)

# 输入一个末端目标位置 (mm)
C_target = (-13.18700715,  -9.78680136, 96.30910521)
q_deg = inv_interp.lookup_pos(C_target, k=8)

print("目标位置:", C_target)
print("插值得到的舵机角度 (deg):", q_deg)
