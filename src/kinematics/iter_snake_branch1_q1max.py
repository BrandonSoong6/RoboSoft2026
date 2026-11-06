import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ===== 贴合版蛇形迭代器（与之前一致）=====
def iter_snake_branch1_q1max(q_min=90, q_max=160, step=1, strict=False):
    def admissible(q1, q2, q3):
        return (q1 > q2 and q1 > q3) if strict else (q1 >= q2 and q1 >= q3)

    Q = np.arange(q_min, q_max + 1, step, dtype=int)
    last_q = None
    for q1 in Q:
        q2_range = list(range(q_min, q1 + 1, step))
        for i_q2, q2 in enumerate(q2_range):
            q3_line = list(range(q_min, q1 + 1, step))
            if i_q2 % 2 == 1:
                q3_line.reverse()

            for q3 in q3_line:
                if not admissible(q1, q2, q3):
                    continue
                # 层间贴合：第一行第一个点尽量对齐上一层终点
                if last_q is not None and q1 > last_q[0] and i_q2 == 0 and q3 == q3_line[0]:
                    q2 = min(last_q[1], q1 if not strict else q1 - 1)
                    q3 = min(last_q[2], q1 if not strict else q1 - 1)
                    if not admissible(q1, q2, q3):
                        continue
                yield (float(q1), float(q2), float(q3))
                last_q = (q1, q2, q3)

# ===== 把相邻关键点“展开”为逐度相邻的小步，避免跳变 =====
def expand_path_monotone(points, step=1, strict=False):
    """
    对于每对相邻关键点 A=(q1a,q2a,q3a), B=(q1b,q2b,q3b)：
    先把 q1 以 1° 步进到 q1b，再把 q2 以 1° 逼近 q2b（保持 q2<=q1），再把 q3 逼近 q3b（保持 q3<=q1）。
    保证中间所有中间点都落在 1/3 区域（strict=False 为 >=；strict=True 则 >，会稍微向内收）。
    """
    def clamp(q1, q2, q3):
        if strict:
            q2 = min(q2, q1 - step)
            q3 = min(q3, q1 - step)
        else:
            q2 = min(q2, q1)
            q3 = min(q3, q1)
        return q1, q2, q3

    out = [points[0]]
    for (q1a,q2a,q3a), (q1b,q2b,q3b) in zip(points[:-1], points[1:]):
        q1,q2,q3 = q1a,q2a,q3a
        # 1) 先调 q1（通常是递增）
        while abs(q1 - q1b) >= 1e-9:
            q1 += np.sign(q1b - q1) * step
            q1,q2,q3 = clamp(q1,q2,q3)
            out.append((q1,q2,q3))
        # 2) 再调 q2
        while abs(q2 - q2b) >= 1e-9:
            q2 += np.sign(q2b - q2) * step
            q1,q2,q3 = clamp(q1,q2,q3)
            out.append((q1,q2,q3))
        # 3) 最后调 q3
        while abs(q3 - q3b) >= 1e-9:
            q3 += np.sign(q3b - q3) * step
            q1,q2,q3 = clamp(q1,q2,q3)
            out.append((q1,q2,q3))
    return out

# ===== 生成关键点，并展开为无跳变的小步 =====
key_points = list(iter_snake_branch1_q1max(90, 160, 1, strict=False))
smooth_points = expand_path_monotone(key_points, step=1, strict=False)

# （可选）跳帧播放以更快：stride=1 表示每帧一步；stride=5 表示每 5 步取一帧
stride = 1
frames = list(range(0, len(smooth_points), stride))

# ===== 动画：滑动窗口 10 个点 =====
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(90, 160); ax.set_ylim(90, 160); ax.set_zlim(90, 160)
ax.set_xlabel("q1"); ax.set_ylabel("q2"); ax.set_zlabel("q3")
ax.set_title("Snake Sampling Path (sliding window of 10 points, no jumps)")

scat = ax.scatter([], [], [], marker='o', s=10)      # 蛇身（蓝，默认色）
head = ax.scatter([], [], [], c='r', marker='o', s=30)  # 蛇头（红）

WINDOW = 10

def update(fi):
    idx = frames[fi]
    w = smooth_points[max(0, idx-WINDOW+1): idx+1]
    V = np.array(w)
    scat._offsets3d = (V[:,0], V[:,1], V[:,2])
    head._offsets3d = ([V[-1,0]],[V[-1,1]],[V[-1,2]])
    return scat, head

# interval=1ms → 很快；把 blit=False，避免 3D 绘图的 blit 兼容性问题
ani = FuncAnimation(fig, update, frames=len(frames), interval=1, blit=False, repeat=False)
plt.show()
