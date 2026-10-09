"""帧间连贯性度量：把「动作连贯 / 角色一致」从肉眼判断变成数字。

三个原始指标（a、b 为同尺寸 BGRA 帧）：
- alpha_iou     不透明区域的交并比（形状/位置一致性）
- rgb_mae       共同不透明区域的 RGB 平均绝对差（颜色/细节一致性）
- centroid_shift 不透明质心位移像素数（整体运动幅度）

coherence_score 综合三者为 0~1 分，越高越连贯：
    0.5*alpha_iou + 0.3*(1-mae/64) + 0.2*(1-shift/32)，各项截断到 [0,1]。

经验阈值（Qwen 像素帧实测）：相邻帧 >0.62 正常，<0.45 大概率跳帧/崩帧。
"""
from __future__ import annotations

import cv2
import numpy as np


def _norm_size(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """尺寸不同时把 b 对齐到 a（最近邻），保证可比。"""
    if b.shape[:2] != a.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]),
                       interpolation=cv2.INTER_NEAREST)
    return a, b


def frame_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = _norm_size(a, b)
    ma = a[:, :, 3] > 127
    mb = b[:, :, 3] > 127
    inter = float((ma & mb).sum())
    union = float((ma | mb).sum())
    alpha_iou = inter / union if union else 1.0
    if inter > 0:
        mae = float(np.abs(a[:, :, :3][ma & mb].astype(np.int16)
                           - b[:, :, :3][ma & mb].astype(np.int16)).mean())
    else:
        mae = 255.0
    if ma.any() and mb.any():
        ca = np.column_stack(np.nonzero(ma)).mean(axis=0)
        cb = np.column_stack(np.nonzero(mb)).mean(axis=0)
        shift = float(np.hypot(*(ca - cb)))
    else:
        shift = 999.0
    return {"alpha_iou": round(alpha_iou, 4),
            "rgb_mae": round(mae, 2),
            "centroid_shift": round(shift, 2)}


def coherence_score(a: np.ndarray, b: np.ndarray) -> float:
    m = frame_metrics(a, b)
    s = (0.5 * m["alpha_iou"]
         + 0.3 * max(0.0, 1.0 - m["rgb_mae"] / 64.0)
         + 0.2 * max(0.0, 1.0 - m["centroid_shift"] / 32.0))
    return round(max(0.0, min(1.0, s)), 3)


def neighbor_blend(cur: np.ndarray, prev: np.ndarray | None,
                   nxt: np.ndarray | None) -> np.ndarray:
    """重抽底图合成：前后帧 50% 混合（过渡姿态优先）。

    两侧都在 -> (prev + nxt) / 2；只有一侧 -> (邻帧 + cur) / 2；
    都没有 -> cur 原样。
    """
    def mix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x, y = _norm_size(x, y)
        return ((x.astype(np.float32) + y.astype(np.float32)) / 2.0
                ).astype(np.uint8)

    if prev is not None and nxt is not None:
        return mix(prev, nxt)
    if prev is not None:
        return mix(cur, prev)
    if nxt is not None:
        return mix(cur, nxt)
    return cur


# ---------------- 角色身份一致性（与姿态无关） ----------------

def _hsv_hist(bgr_px: np.ndarray) -> np.ndarray:
    """不透明像素的 H-S 2D 直方图（归一化）。"""
    hsv = cv2.cvtColor(bgr_px.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def _dominant_colors(bgr_px: np.ndarray, k: int = 8) -> np.ndarray:
    """主色提取（KMeans 中心，BGR）。"""
    data = bgr_px.reshape(-1, 3).astype(np.float32)
    if data.shape[0] > 20000:
        idx = np.random.default_rng(0).choice(data.shape[0], 20000,
                                              replace=False)
        data = data[idx]
    k = max(1, min(k, int(np.unique(data, axis=0).shape[0])))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, _, centers = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    return centers


def _palette_overlap(pa: np.ndarray, pb: np.ndarray, thresh: float = 48.0
                     ) -> float:
    """双向主色重合度：一方主色在另一方最近邻距离 < thresh 的比例。"""
    d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)
    hit_a = (d.min(axis=1) < thresh).mean()
    hit_b = (d.min(axis=0) < thresh).mean()
    return (hit_a + hit_b) / 2


def identity_score(a: np.ndarray, b: np.ndarray) -> float:
    """角色身份一致性（0~1，与姿态/角度无关）。

    度量「配色与明暗分布是否同一角色」：
    0.6 * HSV 直方图相关性 + 0.4 * 主色双向重合度。
    校准基准（像素角色立绘）：同一角色不同姿态/角度 >= 0.8；
    不同角色通常 < 0.6。
    """
    a, b = _norm_size(a, b)
    ma = a[:, :, 3] > 127
    mb = b[:, :, 3] > 127
    if ma.sum() < 50 or mb.sum() < 50:
        return 0.0
    ha = _hsv_hist(a[ma][:, :3])
    hb = _hsv_hist(b[mb][:, :3])
    corr = cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL)
    corr = max(0.0, corr)
    overlap = _palette_overlap(_dominant_colors(a[ma][:, :3]),
                               _dominant_colors(b[mb][:, :3]))
    return round(0.6 * corr + 0.4 * overlap, 3)


def identity_report(frames: list[np.ndarray]) -> dict:
    """帧序列的两两身份一致性矩阵（对角线为 1）。"""
    n = len(frames)
    mat = [[1.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = identity_score(frames[i], frames[j])
            mat[i][j] = mat[j][i] = s
    return {"matrix": mat, "min": min(min(r) for r in mat),
            "avg": round(sum(sum(r) for r in mat) / (n * n), 3) if n else 0.0}


# ---------------- 画风相似度（与角色身份、色相无关） ----------------

def _gray(bgr_px: np.ndarray) -> np.ndarray:
    bgr = bgr_px.reshape(-1, 3).astype(np.uint8).reshape(-1, 1, 3)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).reshape(-1)


def _edge_density(bgra: np.ndarray) -> float:
    """边缘密度：Canny 边缘像素占比（线条粗细/细节密度特征）。"""
    g = cv2.cvtColor(bgra[:, :, :3], cv2.COLOR_BGR2GRAY)
    m = bgra[:, :, 3] > 127
    if m.sum() < 100:
        return 0.0
    edges = cv2.Canny(g, 80, 160)
    return float(edges[m].mean()) / 255.0


def _gray_hist_corr(bgra_a: np.ndarray, bgra_b: np.ndarray) -> float:
    ga = cv2.cvtColor(bgra_a[:, :, :3], cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(bgra_b[:, :, :3], cv2.COLOR_BGR2GRAY)
    ma, mb = bgra_a[:, :, 3] > 127, bgra_b[:, :, 3] > 127
    if ma.sum() < 100 or mb.sum() < 100:
        return 0.0
    ha = cv2.calcHist([ga[ma].reshape(-1, 1)], [0], None, [32], [0, 256])
    hb = cv2.calcHist([gb[mb].reshape(-1, 1)], [0], None, [32], [0, 256])
    cv2.normalize(ha, ha)
    cv2.normalize(hb, hb)
    return max(0.0, cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))


def _sat_hist_corr(bgra_a: np.ndarray, bgra_b: np.ndarray) -> float:
    sa = cv2.cvtColor(bgra_a[:, :, :3], cv2.COLOR_BGR2HSV)[:, :, 1]
    sb = cv2.cvtColor(bgra_b[:, :, :3], cv2.COLOR_BGR2HSV)[:, :, 1]
    ma, mb = bgra_a[:, :, 3] > 127, bgra_b[:, :, 3] > 127
    if ma.sum() < 100 or mb.sum() < 100:
        return 0.0
    ha = cv2.calcHist([sa[ma].reshape(-1, 1)], [0], None, [24], [0, 256])
    hb = cv2.calcHist([sb[mb].reshape(-1, 1)], [0], None, [24], [0, 256])
    cv2.normalize(ha, ha)
    cv2.normalize(hb, hb)
    return max(0.0, cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))


def style_score(a: np.ndarray, b: np.ndarray) -> float:
    """画风相似度（0~1）：度量「渲染技法是否同一画风」。

    与角色身份（配色/形状）和姿态无关——用于验收
    「同一画风应用到不同角色」的场景（Style Forge 设定图 QC）。

    构成：0.35*边缘密度相似（线条粗细）+ 0.25*灰度分布相关（明暗调性）
         + 0.2*饱和度分布相关（上色浓淡）+ 0.2*亮度均值接近度（整体调性）。
    """
    a, b = _norm_size(a, b)
    ma, mb = a[:, :, 3] > 127, b[:, :, 3] > 127
    if ma.sum() < 100 or mb.sum() < 100:
        return 0.0
    # 边缘密度相似度：两者密度差的归一化接近度
    ea, eb = _edge_density(a), _edge_density(b)
    edge_sim = max(0.0, 1.0 - abs(ea - eb) / max(ea, eb, 0.01))
    gray_corr = _gray_hist_corr(a, b)
    sat_corr = _sat_hist_corr(a, b)
    va = float(a[:, :, :3][ma].mean())
    vb = float(b[:, :, :3][mb].mean())
    value_sim = max(0.0, 1.0 - abs(va - vb) / 128.0)
    return round(0.35 * edge_sim + 0.25 * gray_corr + 0.2 * sat_corr
                 + 0.2 * value_sim, 3)


def style_report(frames: list[np.ndarray]) -> dict:
    """帧序列两两画风相似度矩阵。"""
    n = len(frames)
    mat = [[1.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = style_score(frames[i], frames[j])
            mat[i][j] = mat[j][i] = s
    return {"matrix": mat, "min": min(min(r) for r in mat),
            "avg": round(sum(sum(r) for r in mat) / (n * n), 3) if n else 0.0}
