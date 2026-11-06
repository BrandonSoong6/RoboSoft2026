import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

SHOW = {"link0": True, "link1": True, "link2": True, "link3": True, "link4": True}
Q_FIXED_DEG = [90, 0, 0, 0]

# ---------- DH & 基变换 ----------
def dh_transform(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d],
        [0,   0,      0,     1]
    ])

def make_T0(i, side, h, axis_to_center_deg):
    R = side / np.sqrt(3.0)
    theta0 = np.deg2rad([0, 120, 240][i])
    beta = theta0 + np.deg2rad(180.0 - axis_to_center_deg)
    xi, yi = R*np.cos(theta0), R*np.sin(theta0)
    return np.array([
        [-np.sin(beta), 0,  np.cos(beta), xi],
        [ np.cos(beta), 0,  np.sin(beta), yi],
        [ 0,            1,  0,            h ],
        [ 0,            0,  0,            1 ]
    ])

# ---------- 链条 ----------
def chain_Ts_for_arm_manhattan(i, q1,q2,q3,q4, side, axis_to_center_deg, link_params, h=10.0):
    L1,L2,L3 = link_params
    T0 = make_T0(i, side, h, axis_to_center_deg)
    A1  = dh_transform(q1, 0, 0, +np.pi/2)
    A2d = dh_transform(q2, L2+L1, 0, -np.pi/2)
    A2a = dh_transform(0, 0, L3, 0)
    A3  = dh_transform(q3, 0,     L3, -np.pi/2)
    A4  = dh_transform(q4, L2+L1, 0,  +np.pi/2)
    T1   = T0 @ A1
    T2d  = T1 @ A2d
    T2a  = T2d @ A2a
    T3   = T2a @ A3
    T4   = T3 @ A4
    return [("T0",T0),("T1",T1),("T2d",T2d),("T2a",T2a),("T3",T3),("T4",T4)]

# ---------- 绘制小工具 ----------
def _draw_axis_arrow(ax, o, v, color, label, length=8.0):
    v = v / (np.linalg.norm(v)+1e-12)
    ax.quiver(o[0], o[1], o[2], v[0], v[1], v[2],
              length=length, normalize=True, color=color)
    tip = o + v * length
    ax.text(tip[0], tip[1], tip[2], label, fontsize=7, color=color)

def _draw_line_with_label(ax, p, q, color, label):
    ax.plot([p[0],q[0]],[p[1],q[1]],[p[2],q[2]], color=color, linewidth=1.5)
    mid = 0.5*(p+q)
    ax.text(mid[0], mid[1], mid[2], label, fontsize=7, color=color)

# ---------- 主绘制 ----------
def plot_fixed_geometry(side=93.16, axis_to_center_deg=164.0, link_params=(39.5,34.1,5), h=10.0):
    q_fixed = np.deg2rad(Q_FIXED_DEG)
    chains=[]; A_pts=[]; B_pts=[]
    nodes=[]

    for i in range(3):
        Ts = chain_Ts_for_arm_manhattan(i, *q_fixed, side, axis_to_center_deg, link_params, h)
        pts=[Ts[0][1][:3,3]]
        if SHOW["link1"]: pts.append(Ts[1][1][:3,3])
        if SHOW["link2"]: pts+=[Ts[2][1][:3,3], Ts[3][1][:3,3]]
        if SHOW["link3"]: pts.append(Ts[4][1][:3,3])
        if SHOW["link4"]: pts.append(Ts[5][1][:3,3])
        chains.append(np.vstack(pts))
        A_pts.append(Ts[0][1][:3,3])
        B_pts.append(Ts[-1][1][:3,3])
        for tag,T in Ts: nodes.append((i,tag,T))

    fig=plt.figure(); ax=fig.add_subplot(111,projection='3d')

    # 底板 & 上板
    A_loop=np.vstack([A_pts,[A_pts[0]]])
    B_loop=np.vstack([B_pts,[B_pts[0]]])
    ax.plot(A_loop[:,0],A_loop[:,1],A_loop[:,2],'k-')
    ax.plot(B_loop[:,0],B_loop[:,1],B_loop[:,2],'gray')

    # 三条链
    for pts in chains:
        ax.plot(pts[:,0],pts[:,1],pts[:,2],marker='o')

    # 圆心 & 径向
    O=np.mean(A_pts,axis=0); C=np.mean(B_pts,axis=0)
    for i,Ai in enumerate(A_pts): _draw_line_with_label(ax,Ai,O,"cyan",f"A{i+1}→O")
    for i,Bi in enumerate(B_pts): _draw_line_with_label(ax,Bi,C,"magenta",f"B{i+1}→C")

    # ---------- 每个 link 的坐标系 ----------
    for i, tag, T in nodes:
        o = T[:3,3]; x,y,z = T[:3,0],T[:3,1],T[:3,2]

        # 默认绘制点
        o_draw = o.copy()

        # 如果是 link1 (T1)，就偏移一下
        if tag == "T1":
            # 这里选用全局Z方向抬升，你也可以改成其他方向
            offset = np.array([20.0,0,-15.0])   # 偏移距离 (mm)
            o_draw = o + offset
            # 用半透明灰线指回真实位置
            ax.plot([o[0],o_draw[0]],[o[1],o_draw[1]],[o[2],o_draw[2]],
                    color="0.5",alpha=0.35,linestyle="--")

        _draw_axis_arrow(ax,o_draw,x,'r',f"{i+1}:{tag}-x")
        _draw_axis_arrow(ax,o_draw,y,'g',f"{i+1}:{tag}-y")
        _draw_axis_arrow(ax,o_draw,z,'b',f"{i+1}:{tag}-z")

    # 视域
    all_pts=np.vstack(A_pts+B_pts)
    mins,maxs=all_pts.min(0),all_pts.max(0)
    rng=(maxs-mins).max(); center=(maxs+mins)/2
    ax.set_xlim(center[0]-rng/2,center[0]+rng/2)
    ax.set_ylim(center[1]-rng/2,center[1]+rng/2)
    ax.set_zlim(center[2]-rng/2,center[2]+rng/2)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    ax.set_title("Canfield Joint — 坐标系 + 三角形 + 径向")
    plt.show()

if __name__=="__main__":
    plot_fixed_geometry()
