import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares

# ================================================================
# 通用工具函数：将样条整体平移到指定中心
# ================================================================
def shift_points(points, center):
    minv = np.min(points, axis=0)
    maxv = np.max(points, axis=0)
    bbox_center = (minv + maxv) / 2.0
    offset = np.array(center) - bbox_center
    points[:, :3] += offset
    return points


# ================================================================
# N 轨迹
# ================================================================
def line_from_points(p1, p2):
    """ax + by + c = 0"""
    a = p2[1] - p1[1]
    b = p1[0] - p2[0]
    c = -(a * p1[0] + b * p1[1])
    return np.array([a, b, c], dtype=float)

def eval_line(L, P):
    a, b, c = L
    return a*P[0] + b*P[1] + c

def proj_foot_on_line(L, C):
    """返回点C到直线L的垂足"""
    a, b, c = L
    den = a*a + b*b
    t = eval_line(L, C) / den
    return np.array([C[0] - a*t, C[1] - b*t], dtype=float)

def wrap_to_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def solve_circle_three_lines(LA, LB, LC, P_inside):
    """
    用“偏移直线交点 + 第三线闭合”解析求解与三条线相切的圆
    - LA/LB/LC: ax+by+c=0
    - P_inside: 希望圆心所在的“区域内参考点”——用于判定“在每条线的哪一侧”
      规则：对每条线取 side = -sign( eval_line(L, P_inside) )
    返回：(C=(cx,cy), r, sideA, sideB, sideC)
    """
    a1,b1,c1 = LA; n1 = np.hypot(a1,b1)
    a2,b2,c2 = LB; n2 = np.hypot(a2,b2)
    a3,b3,c3 = LC; n3 = np.hypot(a3,b3)

    # 选择三条线的侧（确保圆心落在我们想要的角区/内区）
    s1 = -np.sign(eval_line(LA, P_inside));  s1 = 1.0 if s1>=0 else -1.0
    s2 = -np.sign(eval_line(LB, P_inside));  s2 = 1.0 if s2>=0 else -1.0
    s3 = -np.sign(eval_line(LC, P_inside));  s3 = 1.0 if s3>=0 else -1.0

    # 交解：  [a1 b1] [cx] = -c1 - s1*r*n1
    #         [a2 b2] [cy]   -c2 - s2*r*n2
    M = np.array([[a1, b1],
                  [a2, b2]], dtype=float)
    vC = np.array([c1, c2], dtype=float)
    vR = np.array([s1*n1, s2*n2], dtype=float)

    # p0 + r*p1 的形式
    p0 = np.linalg.solve(M, -vC)
    p1 = -np.linalg.solve(M, vR)

    # 第三条线闭合：a3*(p0x+r*p1x) + b3*(p0y+r*p1y) + c3 + s3*r*n3 = 0
    num = a3*p0[0] + b3*p0[1] + c3
    den = a3*p1[0] + b3*p1[1] + s3*n3
    r = - num / den
    C = p0 + r*p1

    return C, float(abs(r)), s1, s2, s3

def sample_short_arc(C, r, P_start, P_end, step_mm):
    """按“短弧”从 P_start→P_end 采样"""
    v1 = P_start - C
    v2 = P_end   - C
    th1 = np.arctan2(v1[1], v1[0])
    th2 = np.arctan2(v2[1], v2[0])
    dth = wrap_to_pi(th2 - th1)         # 取短弧
    L_arc = abs(dth) * r
    n = max(2, int(np.ceil(L_arc / step_mm)) + 1)
    thetas = np.linspace(th1, th1 + dth, n)
    arc = np.stack([C[0] + r*np.cos(thetas),
                    C[1] + r*np.sin(thetas)], axis=1)
    return arc

def generate_spline_N_centered(step_mm=0.1,
                               height=100.0,
                               base_ratio=0.5,
                               chamfer_mm=20.0,   # 用来“撑角”的延伸量 x（非直接半径）
                               center=(0.0, 0.0, 220.0),
                               save_path="spline_N_centered.npz"):
    """
    几何逻辑：
      1) 左竖向上延伸 x，右下向下延伸 x，得到斜线端点。这样把角“支出来”。
      2) 上角倒角：与 {左竖, 斜线, 顶边 y=+H/2} 三线相切的圆；圆弧替代尖角。
      3) 下角倒角：与 {右竖, 斜线, 底边 y=-H/2} 三线相切的圆；圆弧替代尖角。
      4) 路径：左竖(下→上倒角切点) → 上倒角圆弧 → 斜线(上切点→下切点) → 下倒角圆弧 → 右竖(下倒角切点→右上)
    """
    H = float(height)
    W = float(height * base_ratio)
    x = float(chamfer_mm)
    z_base = float(center[2])

    # 基本四角（未延伸）
    p_left_bottom  = np.array([-W/2, -H/2], dtype=float)
    p_left_top     = np.array([-W/2,  H/2], dtype=float)
    p_right_bottom = np.array([+W/2, -H/2], dtype=float)
    p_right_top    = np.array([+W/2,  H/2], dtype=float)

    # 延伸两端点
    p_top_ext    = np.array([-W/2,  H/2 + x], dtype=float)
    p_bottom_ext = np.array([+W/2, -H/2 - x], dtype=float)

    # 直线：左竖、右竖、斜线、顶边、底边
    L_left   = np.array([1.0, 0.0,  W/2])        # x + W/2 = 0 → x = -W/2
    L_right  = np.array([1.0, 0.0, -W/2])        # x - W/2 = 0 → x = +W/2
    L_diag   = line_from_points(p_top_ext, p_bottom_ext)
    L_top    = np.array([0.0, 1.0, -H/2])        # y - H/2 = 0 → y = +H/2
    L_bottom = np.array([0.0, 1.0,  H/2])        # y + H/2 = 0 → y = -H/2

    # —— 上角倒角（三线：左竖、斜线、顶边），区域内参考点取“字内靠角”的点 ——
    Pin_top = np.array([-W/2 + 1e-3, H/2 - 1e-3], dtype=float)
    C1, r1, _, _, _ = solve_circle_three_lines(L_left, L_diag, L_top, Pin_top)

    # 切点：投影到对应直线（圆与直线相切的切点就是垂足）
    P1_vert = proj_foot_on_line(L_left,  C1)   # 左竖切点
    P1_diag = proj_foot_on_line(L_diag,  C1)   # 斜线切点（上）

    # —— 下角倒角（三线：右竖、斜线、底边） ——
    Pin_bot = np.array([+W/2 - 1e-3, -H/2 + 1e-3], dtype=float)
    C2, r2, _, _, _ = solve_circle_three_lines(L_right, L_diag, L_bottom, Pin_bot)

    P2_diag = proj_foot_on_line(L_diag,  C2)   # 斜线切点（下）
    P2_vert = proj_foot_on_line(L_right, C2)   # 右竖切点

    # ========== 各段采样（保证方向一致且不回头） ==========
    # 左竖：从左下 → P1_vert
    seg1_len = np.linalg.norm(P1_vert - p_left_bottom)
    n1 = max(2, int(np.ceil(seg1_len/step_mm)) + 1)
    seg1_2d = np.linspace(p_left_bottom, P1_vert, n1)

    # 上倒角圆弧：从 P1_vert → P1_diag（短弧）
    arc1_2d = sample_short_arc(C1, r1, P1_vert, P1_diag, step_mm)

    # 斜线：从 P1_diag → P2_diag
    seg2_len = np.linalg.norm(P2_diag - P1_diag)
    n2 = max(2, int(np.ceil(seg2_len/step_mm)) + 1)
    seg2_2d = np.linspace(P1_diag, P2_diag, n2)

    # 下倒角圆弧：从 P2_diag → P2_vert（短弧）
    arc2_2d = sample_short_arc(C2, r2, P2_diag, P2_vert, step_mm)

    # 右竖：从 P2_vert → 右上
    seg3_len = np.linalg.norm(p_right_top - P2_vert)
    n3 = max(2, int(np.ceil(seg3_len/step_mm)) + 1)
    seg3_2d = np.linspace(P2_vert, p_right_top, n3)

    # 组合，并添加 z
    xy = np.vstack([seg1_2d,
                    arc1_2d[1:],     # 去重合点
                    seg2_2d[1:],
                    arc2_2d[1:],
                    seg3_2d[1:]])
    points = np.column_stack([xy[:,0], xy[:,1], np.full(len(xy), z_base, dtype=float)])

    # 平移到指定中心（沿用你原有的 shift_points 语义）
    points = shift_points(points, center)

    np.savez(save_path, points=points)
    print(f"[INFO] Saved chamfered 'N' at center={center} → {save_path} ({len(points)} pts)")
    return points

# ================================================================
# U 轨迹
# ================================================================
def generate_spline_U_centered(step_mm=0.1,
                               height=100.0,
                               base_ratio=0.5,
                               radius_ratio=0.25,
                               center=(0.0, 0.0, 200.0),
                               save_path="spline_U_centered.npz"):
    H = float(height)
    W = float(height * base_ratio)
    R = float(height * radius_ratio)
    z = center[2]
    half_h = H / 2.0

    top_y = half_h
    arc_top_y = -half_h + R
    top_left  = np.array([-W/2, top_y, z])
    top_right = np.array([ W/2, top_y, z])
    left_arc_start  = np.array([-W/2, arc_top_y, z])
    right_arc_start = np.array([ W/2, arc_top_y, z])

    n1 = int(np.ceil((H - R) / step_mm)) + 1
    seg1 = np.linspace(top_left, left_arc_start, n1)

    cx = 0.0
    cy = -half_h + R
    n2 = int(np.ceil(np.pi * R / step_mm)) + 1
    theta = np.linspace(np.pi, 0.0, n2)
    seg2 = np.zeros((n2, 3))
    seg2[:, 0] = cx + R * np.cos(theta)
    seg2[:, 1] = cy - R * np.sin(theta)
    seg2[:, 2] = z

    n3 = int(np.ceil((H - R) / step_mm)) + 1
    seg3 = np.linspace(right_arc_start, top_right, n3)

    points = np.vstack([seg1, seg2[1:], seg3[1:]])
    points = shift_points(points, center)

    np.savez(save_path, points=points)
    print(f"[INFO] Saved 'U' at center={center} → {save_path} ({len(points)} pts)")
    return points

# ================================================================
# S 轨迹
# ================================================================
def generate_spline_S_centered(step_mm=0.1,
                               height=100.0,
                               base_ratio=0.5,
                               center=(0.0, 0.0, 180.0),
                               save_path="spline_S_centered.npz"):
    H = float(height)
    W = float(height * base_ratio)
    R = W / 2.0
    z = center[2]

    cx1, cy1 = 0.0, H / 4.0
    L1 = (3.0 * np.pi / 2.0) * R
    n1 = int(np.ceil(L1 / step_mm)) + 1
    theta1 = np.linspace(0.0, 3.0 * np.pi / 2.0, n1)
    arc1 = np.zeros((n1, 3))
    arc1[:, 0] = cx1 + R * np.cos(theta1)
    arc1[:, 1] = cy1 + R * np.sin(theta1)
    arc1[:, 2] = z

    cx2, cy2 = 0.0, -H / 4.0
    L2 = (3.0 * np.pi / 2.0) * R
    n2 = int(np.ceil(L2 / step_mm)) + 1
    theta2 = np.linspace(np.pi / 2.0, -np.pi, n2)
    arc2 = np.zeros((n2, 3))
    arc2[:, 0] = cx2 + R * np.cos(theta2)
    arc2[:, 1] = cy2 + R * np.sin(theta2)
    arc2[:, 2] = z

    points = np.vstack([arc1, arc2[1:]])
    points = shift_points(points, center)

    np.savez(save_path, points=points)
    print(f"[INFO] Saved 'S' at center={center} → {save_path} ({len(points)} pts)")
    return points


# ================================================================
# 绘图函数
# ================================================================
def plot_all_splines(ptsN, ptsU, ptsS):
    """直接绘制输入的 N/U/S 三个样条，不再读文件"""
    data = [
        (ptsN, "N", "tab:blue"),
        (ptsU, "U", "tab:green"),
        (ptsS, "S", "tab:red"),
    ]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3D Centered N-U-S Trajectories (In-Memory)", fontsize=14)
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")

    # 绘制每个轨迹
    for pts, label, color in data:
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                '-', color=color, linewidth=1.5, label=label)
        ax.text(pts[0, 0], pts[0, 1], pts[0, 2], f"{label}_start", color=color)
        ax.text(pts[-1, 0], pts[-1, 1], pts[-1, 2], f"{label}_end", color=color)

    # 自动调整显示范围
    all_pts = np.vstack([ptsN, ptsU, ptsS])
    max_range = np.ptp(all_pts, axis=0).max() / 2.0
    mid_x, mid_y, mid_z = np.mean(all_pts[:, 0]), np.mean(all_pts[:, 1]), np.mean(all_pts[:, 2])
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.scatter([0], [0], [0], s=40, marker='x', color='k')
    ax.legend()
    plt.tight_layout()
    plt.show()

# ================================================================
# 主函数
# ================================================================
if __name__ == "__main__":
    # 自定义每个字母的中心位置
    ptsN = generate_spline_N_centered(height = 80.0, center=(-20.0, 0.0, 220.0),save_path="spline_trajectory1.npz")
    ptsU = generate_spline_U_centered(height = 80.0, center=(0.0, 0.0, 200.0),save_path="spline_trajectory2.npz")
    ptsS = generate_spline_S_centered(height = 80.0, center=(20.0, 0.0, 180.0),save_path="spline_trajectory3.npz")

    plot_all_splines(ptsN, ptsU, ptsS)
