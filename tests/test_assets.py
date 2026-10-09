"""资产库面板测试（离屏）：多选/分类/文件夹/上云/提示词登记与回填。

用法：QT_QPA_PLATFORM=offscreen python tests/test_assets.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QApplication, QInputDialog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixelforge.ui.assets_panel import AssetLibraryPanel


def _frame(color) -> np.ndarray:
    a = np.zeros((24, 24, 4), np.uint8)
    a[4:20, 4:20] = (*color, 255)
    return a


def _rel_of(panel, keyword: str) -> str:
    for a in panel._assets:
        if keyword in a.path.name:
            return panel._rel(a.path)
    raise AssertionError(f"找不到资产 {keyword}")


def main() -> int:
    app = QApplication(sys.argv)

    tmp = Path(tempfile.mkdtemp(prefix="pf_assets_"))
    # 动画资产：2 帧 + atlas.json
    anim = tmp / "knight_idle"
    anim.mkdir()
    for i, c in enumerate([(255, 0, 0), (0, 255, 0)]):
        cv2.imwrite(str(anim / f"knight_idle_{i:03d}.png"), _frame(c))
    (anim / "knight_idle.atlas.json").write_text(
        json.dumps({"name": "knight_idle", "fps": 8, "frame_count": 2,
                    "frames": []}), "utf-8")
    # 两个单图资产
    cv2.imwrite(str(tmp / "poster.png"), _frame((0, 0, 255)))
    cv2.imwrite(str(tmp / "grid_run.png"), _frame((0, 0, 255)))

    panel = AssetLibraryPanel(root=tmp)
    assert len(panel._assets) == 3, f"应扫描到 3 项: {len(panel._assets)}"
    assert panel.lst.count() == 3
    print("[OK] 1. 扫描: 1 动画目录 + 2 单图 -> 3 项")

    # 框选多选基础设施
    from PySide6.QtWidgets import QAbstractItemView
    assert panel.lst.selectionMode() == QAbstractItemView.ExtendedSelection
    assert panel.lst.isSelectionRectVisible()
    print("[OK] 2. ExtendedSelection + 橡皮筋框选已启用")

    # 多选收藏：全选 -> 一次切换
    for i in range(panel.lst.count()):
        panel.lst.item(i).setSelected(True)
    panel._toggle_favorite()
    favs = panel._load_favorites()
    assert len(favs) == 3 and (tmp / "library.json").is_file()
    panel.cmb_filter.setCurrentText("收藏")
    assert panel.lst.count() == 3
    panel.cmb_filter.setCurrentText("全部")
    for i in range(panel.lst.count()):
        panel.lst.item(i).setSelected(True)
    panel._toggle_favorite()
    assert len(panel._load_favorites()) == 0
    print("[OK] 3. 多选收藏往返（library.json v2）")

    # 分类筛选：默认 generated
    rel_poster = _rel_of(panel, "poster")
    idx = panel._load_index()
    idx["categories"][rel_poster] = "local"
    panel._save_index(idx)
    panel.cmb_filter.setCurrentText("生成资产")
    n_gen = panel.lst.count()
    panel.cmb_filter.setCurrentText("本地资产")
    assert panel.lst.count() == 1
    panel.cmb_filter.setCurrentText("全部")
    assert n_gen == 2
    print("[OK] 4. 分类筛选: 本地/生成（云同机制）")

    # 文件夹：QInputDialog 打桩 -> 新建 -> 移动 -> 文件夹筛选
    panel.lst.clearSelection()
    for it in panel.lst.findItems("*", Qt.MatchWildcard):
        pass
    row_poster = next(i for i in range(panel.lst.count())
                      if panel.lst.item(i).data(Qt.UserRole) == rel_poster)
    panel.lst.setCurrentRow(row_poster)
    QInputDialog.getText = staticmethod(
        lambda *a, **k: ("hero", True))
    panel._new_folder()
    idx = panel._load_index()
    assert idx["folders"][rel_poster] == "hero"
    panel.cmb_folder.setCurrentText("hero")
    assert panel.lst.count() == 1
    panel.cmb_folder.setCurrentText("全部文件夹")
    assert panel.lst.count() == 3
    # 多选移动到文件夹
    row_anim = next(i for i in range(panel.lst.count())
                    if "knight" in panel.lst.item(i).text())
    panel.lst.setCurrentRow(row_anim)
    panel.lst.item(row_poster).setSelected(True)
    panel._move_to_folder("props")
    idx = panel._load_index()
    assert idx["folders"][rel_poster] == "props"
    assert panel.cmb_folder.findText("props") >= 0
    panel.cmb_folder.setCurrentText("全部文件夹")
    print("[OK] 5. 文件夹: 新建/移动/筛选")

    # 提示词登记 -> 导入回填
    info = {"prompt": "8-frame idle cycle", "negative_prompt": "blurry",
            "model": "gpt-image-2", "size": "2k"}
    rel_grid = _rel_of(panel, "grid_run")
    panel.register_prompt([str(tmp / "grid_run.png")], info)
    assert panel._load_index()["prompts"][rel_grid] == info
    got = []
    panel.prompt_loaded.connect(lambda d: got.append(d))
    # 打桩掉模态弹窗（离屏测试中 QMessageBox 会永久挂起）
    from PySide6.QtWidgets import QMessageBox
    QMessageBox.information = staticmethod(lambda *a, **k: None)
    panel.lst.clearSelection()
    row_grid = next(i for i in range(panel.lst.count())
                    if panel.lst.item(i).data(Qt.UserRole) == rel_grid)
    panel.lst.setCurrentRow(row_grid)
    panel._import_prompt()
    assert got and got[0]["prompt"] == "8-frame idle cycle", f"got={got}"
    # 未登记的资产 -> 不发信号
    panel.lst.setCurrentRow(row_anim)
    panel._import_prompt()
    assert len(got) == 1
    print("[OK] 6. 提示词: register_prompt 登记 + prompt_loaded 回填信号")

    # 上云：QSettings cloud_asset_dir -> 复制 + 标记 cloud
    cloud = Path(tempfile.mkdtemp(prefix="pf_cloud_"))
    QSettings("PixelForge", "PixelForge").setValue("cloud_asset_dir", str(cloud))
    panel.lst.clearSelection()
    panel.lst.setCurrentRow(row_grid)
    panel._to_cloud()
    assert (cloud / "grid_run.png").is_file()
    assert panel._load_index()["categories"][rel_grid] == "cloud"
    panel.cmb_filter.setCurrentText("云资产")
    assert panel.lst.count() == 1
    QSettings("PixelForge", "PixelForge").setValue("cloud_asset_dir", "")
    print("[OK] 7. 上云: 复制到云目录 + 分类登记 + 云资产筛选")

    # 重载信号（回归）
    got2 = {"review": None, "player": None}
    panel.frames_to_review.connect(lambda f, a: got2.__setitem__("review", (f, a)))
    panel.frames_to_player.connect(lambda fr: got2.__setitem__("player", fr))
    panel.cmb_filter.setCurrentText("全部")
    panel.lst.clearSelection()
    panel.lst.setCurrentRow(row_anim)
    panel._reload_selected()
    files, atlas = got2["review"]
    assert len(files) == 2 and atlas["name"] == "knight_idle"
    assert got2["player"] is not None and len(got2["player"]) == 2
    print("[OK] 8. 重载动画: 审核区 + 播放器信号")

    # 生成素材两路信号：替换 vs 附属（追加）
    got_m, got_x = [], []
    panel.gen_material.connect(lambda ps: got_m.extend(ps))
    panel.gen_material_extra.connect(lambda ps: got_x.extend(ps))
    panel.lst.clearSelection()
    panel.lst.setCurrentRow(row_grid)
    panel._as_gen_material()
    panel._as_gen_material(extra=True)
    assert len(got_m) == 1 and got_m[0].name == "grid_run.png"
    assert len(got_x) == 1 and got_x[0].name == "grid_run.png"
    print("[OK] 9. 生成素材信号: 替换(gen_material) + 附属(gen_material_extra)")

    print("\n资产库面板测试全部通过 ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
