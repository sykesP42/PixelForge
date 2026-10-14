"""裁切框几何运算（纯函数，无 Qt 依赖）。

图像工具坊（ImageLab）与动画播放器（AnimPlayer）的手动裁切共用；
宿主各自负责画布坐标换算、面板同步与应用动作。
"""
from __future__ import annotations


def normalize_rect(x0: int, y0: int, x1: int, y1: int,
                   max_w: int, max_h: int, min_size: int = 2) -> list[int]:
    """排序 + clamp 到 [0,max_w]x[0,max_h] + 最小边距，返回 [x0,y0,x1,y1]。"""
    x0, x1 = sorted((int(x0), int(x1)))
    y0, y1 = sorted((int(y0), int(y1)))
    x0 = max(0, min(x0, max_w - min_size))
    x1 = max(x0 + min_size, min(x1, max_w))
    y0 = max(0, min(y0, max_h - min_size))
    y1 = max(y0 + min_size, min(y1, max_h))
    return [x0, y0, x1, y1]


def hit_test(rect: list[int], x: int, y: int, tol: int) -> str | None:
    """命中测试：四角优先，其次四边中点，再框内移动，框外 None（新建）。"""
    x0, y0, x1, y1 = rect
    mx, my = (x0 + x1) // 2, (y0 + y1) // 2
    if abs(x - x0) <= tol and abs(y - y0) <= tol:
        return "nw"
    if abs(x - x1) <= tol and abs(y - y0) <= tol:
        return "ne"
    if abs(x - x0) <= tol and abs(y - y1) <= tol:
        return "sw"
    if abs(x - x1) <= tol and abs(y - y1) <= tol:
        return "se"
    if (abs(x - x0) <= tol or abs(x - x1) <= tol
            or abs(y - y0) <= tol or abs(y - y1) <= tol):
        if abs(x - mx) <= tol and abs(y - my) <= tol:
            return "move"
        if abs(x - x0) <= tol:
            return "left"
        if abs(x - x1) <= tol:
            return "right"
        if abs(y - y0) <= tol:
            return "top"
        return "bottom"
    if x0 < x < x1 and y0 < y < y1:
        return "move"
    return None


def corner_rect(fixed: tuple[int, int], moving: tuple[int, int],
                ratio: tuple[int, int] | None, max_w: int, max_h: int,
                min_size: int = 2) -> list[int]:
    """四角缩放：对角 fixed 固定不动；ratio=(rw,rh) 时短边向长边看齐。

    锁比例顺序：宽 -> 高（按比例）-> clamp 高度 -> 回算宽（保比例）。
    """
    fx, fy = fixed
    nx, ny = moving
    nw = max(min_size, abs(nx - fx))
    nh = max(min_size, abs(ny - fy))
    if ratio:
        rw, rh = ratio
        nh = max(min_size, round(nw * rh / rw))
        nh = min(nh, max_h)
        nw = max(min_size, round(nh * rw / rh))
    sx = 1 if nx >= fx else -1
    sy = 1 if ny >= fy else -1
    return normalize_rect(fx, fy, fx + sx * nw, fy + sy * nh,
                          max_w, max_h, min_size)


def fit_edges(rect: list[int], axis: str, ratio: tuple[int, int] | None,
              max_w: int, max_h: int, min_size: int = 2) -> list[int]:
    """边手柄锁比例：以已拖边为基准，垂直方向按中心对称外扩。

    axis="h"：宽已定，修高；axis="w"：高已定，修宽。
    """
    r = list(rect)
    if not ratio:
        return r
    rw, rh = ratio
    if axis == "h":
        nw = r[2] - r[0]
        nh = min(max(min_size, round(nw * rh / rw)), max_h)
        cy = (r[1] + r[3]) / 2
        r[1] = max(0, int(cy - nh / 2))
        r[3] = min(max_h, r[1] + nh)
        r[1] = max(0, r[3] - nh)
    else:
        nh = r[3] - r[1]
        nw = min(max(min_size, round(nh * rw / rh)), max_w)
        cx = (r[0] + r[2]) / 2
        r[0] = max(0, int(cx - nw / 2))
        r[2] = min(max_w, r[0] + nw)
        r[0] = max(0, r[2] - nw)
    return r
