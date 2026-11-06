import numpy as np
from scipy.spatial.transform import Rotation as R

# ---- 数据 ----
cases = {
    "case0": ((100,122,155), [-2.058,  37.583, 92.028], [-0.987,  -2.283, -46.734]),
    "case5": ((155,122,100), [ -9.352, -45.936, 87.023], [ 5.608,  -9.898,  58.987]),
}

def Rz(deg): 
    return R.from_euler("z", deg, degrees=True).as_matrix()

def reflect_vertical(phi_deg):
    """关于包含 z 轴、方位角为 phi 的竖直平面反射"""
    phi = np.deg2rad(phi_deg)
    n = np.array([np.cos(phi), np.sin(phi), 0.0])
    H = np.eye(3) - 2.0*np.outer(n, n)
    return H

def mat_residual_angle(M):
    U, _, Vt = np.linalg.svd(M)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        Vt[-1,:] *= -1
        Rm = U @ Vt
    ang = np.linalg.norm(R.from_matrix(Rm).as_rotvec())
    return np.rad2deg(ang)

# ---- 拿Case0做变换 ----
(_, C0, eul0) = cases["case0"]
C0 = np.array(C0, float)
R0 = R.from_euler("zyx", eul0, degrees=True).as_matrix()

(_, C5, eul5) = cases["case5"]
C5 = np.array(C5, float)
R5 = R.from_euler("zyx", eul5, degrees=True).as_matrix()

# 先关于 φ=180° 镜像
H = reflect_vertical(180)
C_tmp = H @ C0
R_tmp = H @ R0

# 再绕 z 180°
Rz180 = Rz(180)
C_new = Rz180 @ C_tmp
R_new = Rz180 @ R_tmp

# 误差
pos_err = np.linalg.norm(C_new[:2] - C5[:2])
ang_err = mat_residual_angle(R5 @ R_new.T)

print("After Case0 -> mirror(φ=180) -> Rz(180):")
print(f"  pos_xy_err ≈ {pos_err:.3f} mm")
print(f"  rot_err ≈ {ang_err:.3f}°")
print("Target Case5 C:", C5, "Eul:", eul5)
