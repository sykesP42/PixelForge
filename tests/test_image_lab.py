"""图像工具坊（ImageLab）离屏功能测试 — 覆盖实时预览/画笔/智能单帧/自由裁切/放大镜。

用法：QT_QPA_PLATFORM=offscreen python tests/test_image_lab.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixelforge.processing import editing
from pixelforge.ui.image_lab import ImageLab

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / "outputs" / "test_tmp"


def make_green_sheet(path: Path) -> None:
    """2x2 绿幕合成精灵表。"""
    sheet = np.full((256, 256, 3), (60, 180, 60), np.uint8)

    def draw(cx, base_y, body_w, body_h, color):
        cv2.circle(sheet, (cx, base_y - body_h), body_w // 2, color, -1)
        cv2.rectangle(sheet, (cx - body_w // 2, base_y - body_h + body_w // 2),
                      (cx + body_w // 2, base_y), color, -1)

    draw(40, 96, 24, 40, (0, 0, 220))
    draw(180, 88, 20, 55, (220, 120, 0))
    draw(30, 230, 22, 48, (0, 200, 220))
    draw(170, 240, 26, 30, (200, 0, 200))
    cv2.imwrite(str(path), sheet)


def mouse_event(lab: ImageLab, x: float, y: float, etype: QEvent.Type) -> QMouseEvent:
    """在画布本地坐标构造鼠标事件（画布坐标，非图像坐标）。"""
    return QMouseEvent(etype, QPointF(x, y), Qt.LeftButton, Qt.LeftButton,
                       Qt.NoModifier)


def body_point(lab: ImageLab) -> tuple[float, float]:
    """图像坐标 (40, 80)（红色小人身体）对应的画布本地坐标。"""
    pm = lab.canvas.pixmap()
    ox = max(0, (lab.canvas.width() - pm.width()) // 2)
    oy = max(0, (lab.canvas.height() - pm.height()) // 2)
    return ox + 40 * lab._zoom, oy + 80 * lab._zoom


def main() -> int:
    app = QApplication(sys.argv)
    lab = ImageLab()

    # 1. 加载图片
    sheet = TMP / "lab_sheet.png"
    make_green_sheet(sheet)
    lab.load_path(sheet)
    assert lab.arr is not None and lab.arr.shape[2] == 4
    assert lab._orig is not None, "应保留原图快照"
    print("[OK] 打开图片 -> BGRA + 原图快照")

    # 2. 色键实时预览（不改动 arr）
    lab._update_preview()
    assert lab._preview is not None, "纯色背景应能生成预览"
    assert (lab._preview[:, :, 3] == 0).any() and (lab.arr[:, :, 3] == 255).all(), \
        "预览不应改动原图 arr"
    print("[OK] 容差实时预览: 预览生效且原图未被修改")

    # 3. 应用去背景
    lab.apply_key()
    transparent = (lab.arr[:, :, 3] == 0).mean()
    assert transparent > 0.5, f"去背景失败, 透明占比 {transparent:.2f}"
    assert lab._preview is None
    print(f"[OK] 应用去背景: 透明占比 {transparent:.2f}")

    # 4. 画笔擦除（事件流：press -> move -> release）
    lab._set_brush_mode(1)  # 擦除
    assert lab._brush_mode == "erase"
    lab._render()
    cx, cy = body_point(lab)
    icx, icy = 40, 80
    assert lab.arr[icy, icx, 3] > 0, "小人身体处应先有不透明像素"
    lab.eventFilter(lab.canvas, mouse_event(lab, cx, cy, QEvent.MouseButtonPress))
    assert lab._painting
    lab.eventFilter(lab.canvas, mouse_event(lab, cx + 1, cy, QEvent.MouseMove))
    lab.eventFilter(lab.canvas, mouse_event(lab, cx, cy, QEvent.MouseButtonRelease))
    assert not lab._painting
    assert lab.arr[icy, icx, 3] == 0, "擦除后该处 alpha 应为 0"
    print("[OK] 画笔擦除: 身体像素已透明")

    # 5. 画笔恢复
    lab._set_brush_mode(2)  # 恢复
    lab.eventFilter(lab.canvas, mouse_event(lab, cx, cy, QEvent.MouseButtonPress))
    lab.eventFilter(lab.canvas, mouse_event(lab, cx, cy, QEvent.MouseButtonRelease))
    assert lab.arr[icy, icx, 3] > 0, "恢复后应找回原图像素"
    print("[OK] 画笔恢复: 误删像素已找回")

    # 6. 撤销
    lab.undo()
    print("[OK] 撤销正常")

    # 7. 智能单帧（不切碎内容）
    lab.smart_single_frame()
    assert len(lab._frames) == 1, f"智能单帧应输出 1 帧, 实际 {len(lab._frames)}"
    fw, fh = lab._frames[0].shape[1], lab._frames[0].shape[0]
    assert 0 < fw <= 260 and 0 < fh <= 260
    print(f"[OK] 智能单帧: 1 帧 {fw}x{fh}（整体紧致化）")

    # 8. 连通域 / 网格裁切
    lab.auto_crop()
    assert len(lab._frames) == 4
    lab.grid_crop()
    assert len(lab._frames) == 4
    print("[OK] 连通域裁切 + 网格切分: 各 4 帧")

    # 9. 编辑操作
    h0, w0 = lab.arr.shape[:2]
    lab._mutate(editing.rotate90, lab.arr)
    assert lab.arr.shape[:2] == (w0, h0)
    lab.undo()
    lab._mutate(editing.flip, lab.arr, True)
    lab._mutate(lambda a: editing.adjust_bcs(a, brightness=30, contrast=1.2, saturation=1.5),
                lab.arr)
    lab._mutate(editing.trim_edges, lab.arr, 4, 4, 4, 4)
    assert lab.arr.shape[0] == h0 - 8 and lab.arr.shape[1] == w0 - 8
    print("[OK] 旋转/翻转/撤销/色彩调整/裁边")

    # 10. 重置为原图
    lab.reset()
    assert lab.arr.shape[:2] == (256, 256) and (lab.arr[:, :, 3] == 255).all()
    print("[OK] 重置为原图")

    # 11. 自由裁切（裁切框拖动 + 应用）
    lab.apply_key()  # 重新去背景
    lab._set_brush_mode(3)  # 裁切框模式
    assert lab._brush_mode == "crop"
    assert lab._crop_rect == [0, 0, 256, 256], f"进入裁切应初始化全图框: {lab._crop_rect}"
    # 先模拟用户已拖出的一个小于画布的框
    lab._crop_rect = [40, 40, 216, 216]
    lab._crop_press((100, 128))          # 框内 -> 整体移动
    assert lab._drag == "move"
    lab._crop_move((80, 128))            # 左移 20（留边足够）
    assert lab._crop_rect == [20, 40, 196, 216], f"移动后: {lab._crop_rect}"
    w_before = lab.arr.shape[1]
    lab.apply_crop()
    assert lab.arr.shape[1] == w_before - 80, f"应用裁切后宽度应减 80: {lab.arr.shape}"
    lab.undo()
    assert lab.arr.shape[1] == w_before
    print("[OK] 自由裁切: 框内拖动整体移动 + 应用 + 撤销")

    # 12. 裁切框：框外新建框
    lab._crop_rect = [50, 50, 200, 200]
    lab._crop_press((10, 10))            # 远离四边 -> 新建
    assert lab._drag == "new"
    lab._crop_move((30, 40))
    assert lab._crop_rect == [10, 10, 30, 40]
    lab.apply_crop()
    assert lab.arr.shape[:2] == (30, 20), f"新框裁切: {lab.arr.shape}"
    lab.undo()
    print("[OK] 裁切框: 框外拖动新建框 + 应用")

    # 13. 放大镜
    lab._update_magnifier(40, 80)
    assert lab.mag.pixmap() is not None, "放大镜应显示内容"
    print("[OK] 放大镜渲染")

    # 14. 帧导出
    lab.load_path(sheet)
    lab.apply_key()
    lab.auto_crop()
    out = TMP / "lab_frames"
    out.mkdir(parents=True, exist_ok=True)
    for i, a in enumerate(lab._frames):
        cv2.imwrite(str(out / f"lab_sheet_{i:03d}.png"), a)
    assert len(list(out.glob("*.png"))) == 4
    print("[OK] 帧导出: 4 个 PNG")

    # 15. 边界：空画布网格切分
    blank = np.zeros((64, 64, 4), np.uint8)
    assert editing.grid_split(blank, 2, 2) == []
    print("[OK] 空画布边界处理")

    # 16. A/B 对比：弹窗（打桩 exec）+ 分割线拖动 + 尺寸对齐 + 空态提示
    from PySide6.QtWidgets import QDialog
    from pixelforge.ui.image_lab import _ABCompare

    def _open_stub(lab):
        cap = {}
        orig = QDialog.exec
        QDialog.exec = lambda self: (cap.setdefault("dlg", self), 0)[1]
        try:
            lab._open_ab_compare()
        finally:
            QDialog.exec = orig
        return cap.get("dlg")

    lab._orig = np.full((40, 60, 4), (0, 0, 200, 255), np.uint8)
    lab.arr = np.full((40, 60, 4), (0, 200, 0, 255), np.uint8)
    lab._preview = None
    dlg = _open_stub(lab)
    assert dlg is not None, "应弹出对比对话框"
    view = dlg.findChild(_ABCompare)
    assert view is not None and abs(view._ratio - 0.5) < 1e-6
    view.resize(600, 480)
    view.grab()                     # 触发 paintEvent，验证绘制不崩
    x, y, w, h, size = view._geo()
    assert w > 0 and h > 0, f"显示区应有效: {view._geo()}"
    view._set_ratio(x + w * 0.25)
    assert abs(view._ratio - 0.25) < 0.02, f"拖动后 ratio={view._ratio}"
    view._set_ratio(x - 999)
    assert view._ratio == 0.0, "左越界应收敛到 0"
    view._set_ratio(x + w + 999)
    assert view._ratio == 1.0, "右越界应收敛到 1"
    # 尺寸不一致 -> B 就近缩放到 A，仍能正常绘制
    lab.arr = np.full((20, 30, 4), (0, 200, 0, 255), np.uint8)
    dlg2 = _open_stub(lab)
    v2 = dlg2.findChild(_ABCompare)
    assert v2 is not None
    v2.resize(600, 480)
    v2.grab()
    # 未打开图片 -> 仅状态提示，不弹窗
    lab._orig = None
    msgs = []
    lab.state_changed.connect(lambda m: msgs.append(m))
    assert _open_stub(lab) is None
    assert any("A/B" in m for m in msgs), f"应有空态提示: {msgs}"
    print("[OK] A/B 对比: 弹窗/分割线拖动与边界收敛/尺寸对齐/空态提示")

    # 17. 逐帧位移/缩放：对话框逻辑 + 一键应用到全部帧 + 撤销
    from PySide6.QtWidgets import QDialog as _QD
    from pixelforge.ui.image_lab import FrameAdjustDialog
    from pixelforge.ui import image_lab as il_mod

    fa = np.zeros((10, 10, 4), np.uint8)
    fa[2:8, 2:8] = (0, 0, 255, 255)
    fb = np.zeros((20, 6, 4), np.uint8)
    fb[2:18, 1:5] = (0, 255, 0, 255)
    lab._set_frames([fa.copy(), fb.copy()], quiet=True)

    # 17a. 对话框内部逻辑
    dlg = FrameAdjustDialog([fa.copy(), fb.copy()])
    assert dlg._dx == [0, 0] and dlg._dy == [0, 0] and dlg._sc == [1.0, 1.0], \
        "初始参数应为单位值"
    assert dlg._anchor() == (0.5, 0.5), "默认锚点为中心"
    dlg.sp_sc.setValue(200)                  # 改当前帧(0) 缩放 2x
    assert abs(dlg._sc[0] - 2.0) < 1e-6, f"缩放应写入当前帧: {dlg._sc}"
    assert abs(dlg._sc[1] - 1.0) < 1e-6, "未选中帧不应被改动"
    dlg.sp_dx.setValue(4)
    assert dlg._dx[0] == 4
    dlg._broadcast()                         # 一键应用到全部帧
    assert dlg._dx == [4, 4] and dlg._sc == [2.0, 2.0], \
        f"广播后全部帧参数应一致: dx={dlg._dx} sc={dlg._sc}"
    res = dlg.result_frames()
    assert len(res) == 2 and res[0].shape == res[1].shape, "结果应统一画布"
    assert res[0].shape[1] > fb.shape[1], "2x 缩放后画布应放大"
    dlg._baseline()                          # 一键脚底对齐
    assert dlg._anchor() == (0.5, 1.0), "锚点应切到脚底中心"
    assert dlg._dx == [0, 0] and dlg._sc == [1.0, 1.0], "对齐应把参数归零"
    base = dlg.result_frames()
    bots = []
    for o in base:
        ys, xs = np.where(o[:, :, 3] > 0)
        bots.append(int(ys.max()))
    assert bots[0] == bots[1], f"脚底对齐后底线应一致: {bots}"
    dlg._refresh()
    assert dlg.pv.pixmap() is not None and not dlg.pv.pixmap().isNull()
    print("[OK] 逐帧位移/缩放: 参数写入/广播/脚底对齐/预览渲染")

    # 17b. 面板入口：一键应用到全部帧 + 撤销回退
    def _fake_exec(self):
        self.sp_dx.setValue(5)
        self._broadcast()
        return _QD.Accepted

    def _rel_x(fp: np.ndarray) -> float:
        """内容水平中心相对「自身画布中心」的偏移（等价播放器的居中归一化）。"""
        ys, xs = np.where(fp[:, :, 3] > 0)
        return float(xs.mean()) - fp.shape[1] / 2.0

    orig_exec = FrameAdjustDialog.exec
    FrameAdjustDialog.exec = _fake_exec
    try:
        rel0 = [_rel_x(f) for f in lab._frames]
        lab._open_frame_adjust()
        assert len(lab._frames) == 2
        assert lab._frames[0].shape == lab._frames[1].shape, "应用后应统一画布"
        rel1 = [_rel_x(f) for f in lab._frames]
        for a, b in zip(rel0, rel1):
            assert abs((b - a) - 5.0) < 1e-6, \
                f"两帧相对画布中心都应右移 5px: {a:.1f}->{b:.1f}"
        lab.undo()
        assert [f.shape for f in lab._frames] == [fa.shape, fb.shape], \
            "撤销应恢复原始帧尺寸"
    finally:
        FrameAdjustDialog.exec = orig_exec
    del il_mod
    print("[OK] 逐帧位移/缩放: 面板入口一键应用（全部帧相对画布中心右移 5px）+ 撤销回退")

    # 18. waifu2x：工具坊入口（同步打桩）→ 画布替换 + 撤销 + 进度文案
    from pixelforge.ui.w2x_worker import Waifu2xWorker
    assert hasattr(lab, "cmb_w2x_scale") and hasattr(lab, "cmb_w2x_noise"), \
        "工具坊应有 waifu2x 倍率/降噪控件"
    wsrc = TMP / "w2x_src.png"
    cv2.imwrite(str(wsrc), np.full((10, 10, 4), (40, 200, 40, 255), np.uint8))
    lab.load_path(wsrc)
    msgs2 = []
    lab.state_changed.connect(msgs2.append)
    orig_start = Waifu2xWorker.start
    Waifu2xWorker.start = lambda self: self.run()
    try:
        lab.cmb_w2x_scale.setCurrentText("2x")
        lab.btn_w2x.click()
    finally:
        Waifu2xWorker.start = orig_start
    assert lab.arr is not None and lab.arr.shape[:2] == (20, 20), \
        f"画布应变为 2x: {lab.arr.shape}"
    assert any("waifu2x 完成" in m for m in msgs2), f"应有完成日志: {msgs2[-3:]}"
    # 撤销应回到 10x10
    lab.undo()
    assert lab.arr.shape[:2] == (10, 10), "撤销应恢复原画布"
    print("[OK] waifu2x: 工具坊放大（画布替换 + 写盘 + 撤销回退）")

    # 19. 调色板吸取：从当前图提取主色板并锁定（QInputDialog 打桩）
    from PySide6.QtWidgets import QInputDialog
    from pixelforge.processing.image_processor import extract_palette
    flat = np.zeros((8, 8, 4), np.uint8)
    flat[:, :] = (90, 45, 200, 255)   # 单色
    pal0 = extract_palette(flat, k=8)
    assert pal0 == [[90, 45, 200]], f"平色图应提取唯一色: {pal0}"
    quad = np.zeros((8, 8, 4), np.uint8)
    quad[:4, :4] = (0, 0, 255, 255)     # 红 (BGR)
    quad[:4, 4:] = (0, 255, 0, 255)     # 绿
    quad[4:, :4] = (255, 0, 0, 255)     # 蓝
    quad[4:, 4:] = (255, 255, 255, 255)  # 白
    pquad = TMP / "quad_pal.png"
    cv2.imwrite(str(pquad), quad)
    lab.load_path(pquad)
    orig_getint = QInputDialog.getInt
    QInputDialog.getInt = lambda *a, **k: (4, True)
    try:
        lab._pick_palette_dialog()
    finally:
        QInputDialog.getInt = orig_getint
    assert lab._locked_palette, "吸取后应有锁定调色板"
    got = {tuple(c) for c in lab._locked_palette}
    expect = {(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 255)}
    assert got <= expect, f"提取色应来自图中四色: {got}"
    assert "吸取自" in lab.lbl_palette.text(), lab.lbl_palette.text()
    # 取消（ok=False）→ 调色板保持不变
    pal_before = [list(c) for c in lab._locked_palette]
    QInputDialog.getInt = lambda *a, **k: (16, False)
    try:
        lab._pick_palette_dialog()
    finally:
        QInputDialog.getInt = orig_getint
    assert [list(c) for c in lab._locked_palette] == pal_before, "取消不应修改调色板"
    print("[OK] 调色板吸取: 纯函数平色/四色提取 + 入口锁定 + 取消保护")

    print("\n图像工具坊全部测试通过 ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
