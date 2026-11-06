from pathlib import Path
import numpy as np
import sys, os

def locate_npz(target_name_or_path: str, max_up=5) -> Path | None:
    # 传入路径就按路径；传入文件名就从脚本所在目录开始搜
    p = Path(target_name_or_path)
    if p.is_absolute() and p.exists():
        return p

    # 1) 先看相对路径是否刚好存在（以当前工作目录为基准）
    if p.exists():
        return p.resolve()

    # 2) 到脚本所在目录找同名文件
    script_dir = Path(__file__).resolve().parent
    candidate = (script_dir / p.name)
    if candidate.exists():
        return candidate

    # 3) 向上递归搜索几层
    for base in [script_dir, *list(script_dir.parents)[:max_up]]:
        for hit in base.rglob(p.name):
            return hit.resolve()
    return None

def read_npz(npz_path: Path):
    print(f"\n使用文件: {npz_path}")
    data = np.load(npz_path)  # allow_pickle=False 默认即可
    print("包含的键：", data.files)
    for k in data.files:
        arr = data[k]
        print(f"\n=== {k} ===")
        print("形状:", arr.shape, "  dtype:", arr.dtype)
        print(arr)

if __name__ == "__main__":
    # 诊断信息：看看你实际从哪运行
    print("当前工作目录 cwd      :", Path.cwd())
    print("脚本所在目录 scriptdir:", Path(__file__).resolve().parent)

    target = sys.argv[1] if len(sys.argv) > 1 else "marker_to_center.npz"
    path = locate_npz(target)
    if path is None:
        print(f"\n没有找到文件：{target}")
        print("排查：")
        print("  1) 确认真实文件名与后缀（不是 .npz.zip）。")
        print("  2) 如果在 Windows 路径里有反斜杠，请用原始字符串 r'C:\\path\\file.npz' 或改用正斜杠。")
        print("  3) 右键复制*绝对路径*后这样运行：")
        print("       python read_npz.py '绝对路径/marker_to_center.npz'")
        raise SystemExit(1)

    read_npz(path)
