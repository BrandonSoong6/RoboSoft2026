import numpy as np
from scipy.spatial.transform import Rotation as R

def build_ik_npz(fk_npz, out_npz="ik_lookup.npz"):
    d = np.load(fk_npz, allow_pickle=True)
    arr = d["arr"]
    q1r, q2r, q3r = d["q1_range"], d["q2_range"], d["q3_range"]

    Q1, Q2, Q3 = np.meshgrid(q1r, q2r, q3r, indexing="ij")
    q_flat = np.stack([Q1, Q2, Q3], axis=-1).reshape(-1, 3)

    vals = arr.reshape(-1, arr.shape[-1])
    pos = vals[:, 9:12]   # (Cx,Cy,Cz)
    eul = vals[:, 6:9]
    quat = R.from_euler("zyx", eul, degrees=True).as_quat()

    success = vals[:, 12]
    res_norm = vals[:, 13]

    # ===== 过滤非法点 =====
    mask = np.isfinite(pos).all(axis=1) & np.isfinite(quat).all(axis=1)
    mask &= (success > 0.5)   # 只保留成功解
    pos = pos[mask]
    quat = quat[mask]
    q_flat = q_flat[mask]
    success = success[mask]
    res_norm = res_norm[mask]

    np.savez_compressed(out_npz,
        pos=pos.astype(np.float32),
        quat=quat.astype(np.float32),
        angles=q_flat.astype(np.float32),
        success=success.astype(np.uint8),
        res_norm=res_norm.astype(np.float32)
    )
    print(f"[OK] IK lookup file saved: {out_npz}, samples = {pos.shape[0]}")


build_ik_npz("canfield_lookup_pose_0p4deg_full.npz", "canfield_ik_0p4deg_lookup.npz")