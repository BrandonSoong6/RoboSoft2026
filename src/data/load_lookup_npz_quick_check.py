# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import animation
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# import time
# from canfield_plot import chain_Ts_for_arm_manhattan
# from scipy.spatial.transform import Rotation as R

# # ---------- 数据加载 ----------
# def load_lookup_npz(npz_file):
#     d = np.load(npz_file, allow_pickle=True)
#     return d["arr"], d["q1_range"], d["q2_range"], d["q3_range"], float(d["step"]), d["features"].tolist()

# # ---------- O(1) 插值：逐轴 lerp，避免临时 2x2x2 分配 ----------
# def interpolate_lookup(q_new, arr, q1_range, q2_range, q3_range, step):
#     q1, q2, q3 = q_new
#     def idx_frac(q, q0, step, n):
#         i = (q - q0) / step
#         j = int(np.floor(i))
#         if j < 0: j, t = 0, 0.0
#         elif j > n-2: j, t = n-2, 1.0
#         else: t = i - j
#         return j, t
#     j1, t1 = idx_frac(q1, q1_range[0], step, len(q1_range))
#     j2, t2 = idx_frac(q2, q2_range[0], step, len(q2_range))
#     j3, t3 = idx_frac(q3, q3_range[0], step, len(q3_range))

#     V000 = arr[j1  , j2  , j3  ]
#     V100 = arr[j1+1, j2  , j3  ]
#     V010 = arr[j1  , j2+1, j3  ]
#     V001 = arr[j1  , j2  , j3+1]
#     V110 = arr[j1+1, j2+1, j3  ]
#     V101 = arr[j1+1, j2  , j3+1]
#     V011 = arr[j1  , j2+1, j3+1]
#     V111 = arr[j1+1, j2+1, j3+1]

#     def lerp(t, a, b): return a + t*(b - a)
#     c00 = lerp(t1, V000, V100); c01 = lerp(t1, V001, V101)
#     c10 = lerp(t1, V010, V110); c11 = lerp(t1, V011, V111)
#     c0  = lerp(t2, c00, c10);   c1  = lerp(t2, c01, c11)
#     val = lerp(t3, c0, c1)      # shape: (F,)

#     # 返回 dict（含主动角）
#     return {
#         "q1_deg": q1, "q2_deg": q2, "q3_deg": q3,
#         "q2_1_deg": float(val[0]), "q3_1_deg": float(val[1]),
#         "q2_2_deg": float(val[2]), "q3_2_deg": float(val[3]),
#         "q2_3_deg": float(val[4]), "q3_3_deg": float(val[5]),
#         "eul_z_deg": float(val[6]), "eul_y_deg": float(val[7]), "eul_x_deg": float(val[8]),
#         "Cx_mm": float(val[9]), "Cy_mm": float(val[10]), "Cz_mm": float(val[11]),
#         "success": float(val[12]), "res_norm": float(val[13]),
#     }

# # ---------- 动画类：预创建 Artist，循环 set_data 更新 ----------
# class CanfieldAnimator:
#     def __init__(self, arr, q1_range, q2_range, q3_range, step):
#         self.arr, self.q1r, self.q2r, self.q3r, self.step = arr, q1_range, q2_range, q3_range, step

#         self.fig = plt.figure()
#         self.ax  = self.fig.add_subplot(111, projection="3d")
#         ax = self.ax

#         # 轴域 & 外观一次性设置
#         ax.set_xlim(-100, 100); ax.set_ylim(-100, 100); ax.set_zlim(0, 200)
#         ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
#         ax.set_title("Canfield Joint — 插值动态演示（预创建 Artist）")
#         ax.set_box_aspect([1,1,1])
#         ax.grid(False)

#         # 3 个支链，每条支链 4 段线（预创建）
#         colors = ["red","green","blue","purple"]
#         self.lines = []  # 3 * 4
#         for _ in range(3):
#             segs = []
#             for c in colors:
#                 line, = ax.plot([], [], [], color=c, marker="o")
#                 segs.append(line)
#             self.lines.append(segs)

#         # 上/下板三角形（预创建）
#         self.polyA = Poly3DCollection([np.zeros((3,3))], facecolors="black", alpha=0.15, edgecolors="black")
#         self.polyB = Poly3DCollection([np.zeros((3,3))], facecolors="red",   alpha=0.30, edgecolors="red")
#         ax.add_collection3d(self.polyA)
#         ax.add_collection3d(self.polyB)

#         # 法向量线（预创建，用简单线代替箭头以提速）
#         self.n_line, = ax.plot([], [], [], "b-")

#         # HUD（一次创建，反复 set_text）
#         self.hud = ax.text2D(0.02, 0.95, "", transform=ax.transAxes,
#                              fontsize=10, color="black",
#                              bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

#         # 计时
#         self.t_prev = time.perf_counter()
#         self.fps = 0.0

#     def update(self, frame):
#         t0 = time.perf_counter()
#         q_new = np.random.uniform(90, 160, 3).astype(np.float32)

#         # 只测插值耗时
#         row = interpolate_lookup(q_new, self.arr, self.q1r, self.q2r, self.q3r, self.step)
#         t1 = time.perf_counter()

#         # —— 绘制，纯更新 Artist，不再 cla() —— #
#         q_act_deg = [row["q1_deg"], row["q2_deg"], row["q3_deg"]]
#         q1s = np.deg2rad(q_act_deg)

#         # 组装被动角（每条支链 q2,q3;q4=-q2）
#         q_pass = np.deg2rad([
#             row["q2_1_deg"], row["q3_1_deg"], -row["q2_1_deg"],
#             row["q2_2_deg"], row["q3_2_deg"], -row["q2_2_deg"],
#             row["q2_3_deg"], row["q3_3_deg"], -row["q2_3_deg"],
#         ])

#         A_pts, B_pts = [], []
#         for i in range(3):
#             q1 = q1s[i]
#             q2, q3, q4 = q_pass[3*i:3*i+3]
#             T0,T1,T2d,T2a,T3,T4 = chain_Ts_for_arm_manhattan(
#                 i, q1, q2, q3, q4, side=93.16, axis_to_center_deg=-16.0, link_params=(39.5,34.1,5), h=10.0
#             )
#             A_pts.append(T0[:3,3]); B_pts.append(T4[:3,3])

#             # 更新 4 段线的数据
#             segs = self.lines[i]
#             # seg1: T0->T1
#             segs[0].set_data_3d([T0[0,3], T1[0,3]],[T0[1,3], T1[1,3]],[T0[2,3], T1[2,3]])
#             # seg2: T1->T2d->T2a
#             segs[1].set_data_3d([T1[0,3], T2d[0,3], T2a[0,3]],
#                                  [T1[1,3], T2d[1,3], T2a[1,3]],
#                                  [T1[2,3], T2d[2,3], T2a[2,3]])
#             # seg3: T2a->T3
#             segs[2].set_data_3d([T2a[0,3], T3[0,3]],[T2a[1,3], T3[1,3]],[T2a[2,3], T3[2,3]])
#             # seg4: T3->T4
#             segs[3].set_data_3d([T3[0,3], T4[0,3]],[T3[1,3], T4[1,3]],[T3[2,3], T4[2,3]])

#         A_loop = np.vstack([A_pts[0], A_pts[1], A_pts[2]])
#         B_loop = np.vstack([B_pts[0], B_pts[1], B_pts[2]])
#         self.polyA.set_verts([A_loop])
#         self.polyB.set_verts([B_loop])

#         C = (B_pts[0] + B_pts[1] + B_pts[2]) / 3.0
#         n_top = np.cross(B_pts[1]-B_pts[0], B_pts[2]-B_pts[0])
#         n_top /= np.linalg.norm(n_top) + 1e-12
#         if n_top[2] < 0: n_top = -n_top
#         L = 15.0
#         self.n_line.set_data_3d([C[0], C[0]+n_top[0]*L],
#                                 [C[1], C[1]+n_top[1]*L],
#                                 [C[2], C[2]+n_top[2]*L])

#         # 计时 & HUD（FPS 用滑动平均更稳）
#         t2 = time.perf_counter()
#         dt_frame = t2 - self.t_prev
#         self.t_prev = t2
#         inst_fps = 1.0/dt_frame if dt_frame > 0 else 0.0
#         self.fps = 0.9*self.fps + 0.1*inst_fps

#         lookup_ms = (t1 - t0)*1000.0
#         draw_ms   = (t2 - t1)*1000.0
#         self.hud.set_text(f"FPS: {self.fps:.1f}\nLookup: {lookup_ms:.2f} ms\nDraw: {draw_ms:.2f} ms")
#         return []

# if __name__ == "__main__":
#     arr, q1r, q2r, q3r, step, _ = load_lookup_npz("canfield_lookup_pose_0p4deg_full.npz")
#     app = CanfieldAnimator(arr, q1r, q2r, q3r, step)
#     ani = animation.FuncAnimation(app.fig, app.update, interval=1, blit=False, cache_frame_data=False)
#     plt.show()
