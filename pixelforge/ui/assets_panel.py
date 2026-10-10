"""资产库面板：分类（本地/云/生成）+ 文件夹整理 + 多选 + 右键操作。

资产类型：
- 动画资产：含 *.atlas.json 的目录（帧 PNG + 元数据）-> 重载到审核区/播放器
- 单图资产：outputs 顶层 PNG / GIF -> 重载到工具坊
索引 library.json v2：
  favorites / folders(rel->文件夹名) / categories(rel->local|cloud|generated)
  prompts(rel->生成时用的提示词信息)
云资产：设置中的「云资产目录」（可为网盘同步目录），上云 = 复制过去并登记。
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QHBoxLayout, QInputDialog, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMenu, QMessageBox,
    QPushButton, QVBoxLayout, QWidget,
)

CAT_NAMES = {"local": "本地资产", "cloud": "云资产", "generated": "生成资产"}
UPLOAD_DIR = "uploads"          # 用户上传素材在资产根下的存放子目录


def _to_pixmap(arr: np.ndarray) -> QPixmap:
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGRA)
    elif arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2BGRA)
    rgba = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
    h, w = rgba.shape[:2]
    img = QImage(rgba.data, w, h, w * 4, QImage.Format_RGBA8888)
    return QPixmap.fromImage(img.copy())


class AssetItem:
    """一个可浏览的资产（动画目录或单图文件）。"""

    def __init__(self, path: Path, kind: str, frame_files: list[Path] | None = None,
                 atlas: dict | None = None):
        self.path = path
        self.kind = kind              # "anim" | "image"
        self.frame_files = frame_files or []
        self.atlas = atlas or {}

    def label(self) -> str:
        if self.kind == "anim":
            return f"{self.path.name}  ({len(self.frame_files)} 帧)"
        return self.path.name


class AssetLibraryPanel(QWidget):
    state_changed = Signal(str)

    IMAGE_EXTS = (".png", ".gif", ".jpg", ".jpeg", ".webp", ".bmp")
    frames_to_review = Signal(list, dict)   # [Path], atlas -> 审核区
    frames_to_player = Signal(list)         # [BGRA ndarray] -> 播放器
    image_to_lab = Signal(Path)             # 单图 -> 工具坊
    prompt_loaded = Signal(dict)            # 生成提示词信息 -> 主界面回填
    gen_material = Signal(list)             # [Path] 作为生成素材（参考图）
    gen_material_extra = Signal(list)       # [Path] 附属素材（追加参考图）
    preview_requested = Signal(str)         # 双击资产 -> 中央预览看图器

    def __init__(self, root: Path | None = None, parent=None):
        super().__init__(parent)
        self._root = Path(root) if root else None  # None = 默认 outputs/
        self._assets: list[AssetItem] = []
        self._folder_filter = ""
        self._build_ui()
        self.refresh()

    # ---------- UI ----------

    def _build_ui(self) -> None:
        lay = QVBoxLayout(self)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("分类"))
        self.cmb_filter = QComboBox()
        self.cmb_filter.addItems(
            ["全部", "动画资产", "单图", "收藏",
             CAT_NAMES["local"], CAT_NAMES["cloud"], CAT_NAMES["generated"]])
        self.cmb_filter.currentIndexChanged.connect(lambda _: self._apply_filter())
        bar.addWidget(self.cmb_filter)
        self.ed_search = QLineEdit()
        self.ed_search.setPlaceholderText("🔍 模糊搜索（名称/路径/分类/提示词，空格分词）")
        self.ed_search.setClearButtonEnabled(True)
        self.ed_search.setMinimumWidth(220)
        self.ed_search.setToolTip(
            "模糊搜索：多个关键词用空格分隔，需全部命中（支持顺序子序列，如 “红披风” 可命中 “红色_披风_idle”）")
        self.ed_search.textChanged.connect(lambda _: self._apply_filter())
        bar.addWidget(self.ed_search)
        self.cmb_folder = QComboBox()
        self.cmb_folder.currentTextChanged.connect(self._on_folder_filter)
        bar.addWidget(self.cmb_folder)
        b_newdir = QPushButton("📁 新建文件夹")
        b_newdir.setToolTip("选中资产后，先建文件夹再「移动到文件夹」归类")
        b_newdir.clicked.connect(self._new_folder)
        bar.addWidget(b_newdir)
        b_fav = QPushButton("♥ 收藏")
        b_fav.clicked.connect(self._toggle_favorite)
        bar.addWidget(b_fav)
        b_reload = QPushButton("重载选中")
        b_reload.setProperty("accent", True)
        b_reload.clicked.connect(self._reload_selected)
        bar.addWidget(b_reload)
        b_import = QPushButton("📥 导入")
        b_import.setToolTip(
            "把外部图片/动画导入到当前分类（可多选，动画按文件夹导入）")
        b_import.clicked.connect(self._import_files)
        bar.addWidget(b_import)
        b_refresh = QPushButton("刷新")
        b_refresh.clicked.connect(self.refresh)
        bar.addWidget(b_refresh)
        bar.addStretch(1)
        lay.addLayout(bar)

        self.lst = QListWidget()
        self.lst.setObjectName("assetGrid")
        self.lst.setIconSize(QSize(96, 96))
        self.lst.setViewMode(QListWidget.IconMode)
        self.lst.setResizeMode(QListWidget.Adjust)
        self.lst.setWordWrap(True)
        # 框选多选：ExtendedSelection + 橡皮筋框选
        self.lst.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.lst.setSelectionRectVisible(True)
        self.lst.setContextMenuPolicy(Qt.CustomContextMenu)
        self.lst.customContextMenuRequested.connect(self._context_menu)
        self.lst.itemDoubleClicked.connect(lambda _: self._preview_selected())
        lay.addWidget(self.lst, 1)
        # 工业软件式操作提示行（常驻底部）
        tips = QLabel("双击=预览打开｜右键=重载/AI修图素材/附属素材/导入提示词/"
                      "收藏/分类｜框选可多选｜🔍搜索支持模糊匹配（空格分词）｜切到本页自动刷新")
        tips.setStyleSheet("color:#8a8f98; font-size:11px;")
        tips.setWordWrap(True)
        lay.addWidget(tips)

    # ---------- 索引（library.json v2） ----------

    def _scan_root(self) -> Path:
        return self._root or (Path(__file__).resolve().parent.parent.parent
                              / "outputs")

    def _index_file(self) -> Path:
        return self._scan_root() / "library.json"

    def _load_index(self) -> dict:
        try:
            d = json.loads(self._index_file().read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            d = {}
        for k, dv in (("favorites", []), ("folders", {}),
                      ("categories", {}), ("prompts", {})):
            d.setdefault(k, dv)
        return d

    def _save_index(self, idx: dict) -> None:
        self._index_file().write_text(
            json.dumps(idx, ensure_ascii=False, indent=2), "utf-8")

    def _load_favorites(self) -> set[str]:
        return set(self._load_index()["favorites"])

    def _save_favorites(self, favs: set[str]) -> None:
        idx = self._load_index()
        idx["favorites"] = sorted(favs)
        self._save_index(idx)

    # ---------- 扫描 ----------

    def refresh(self) -> None:
        """重扫资产根：动画目录 + 顶层图片 + uploads/ 用户上传素材。"""
        root = self._scan_root()
        root.mkdir(parents=True, exist_ok=True)
        self._assets = []
        for d in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime,
                        reverse=True):
            if not d.is_dir() or d.name.startswith("."):
                continue
            atlas_files = list(d.glob("*.atlas.json")) + \
                [f for f in d.glob("*.json") if f.name in ("atlas.json", "session.json")]
            atlas = None
            if atlas_files:
                try:
                    atlas = json.loads(atlas_files[0].read_text("utf-8"))
                except (OSError, json.JSONDecodeError):
                    atlas = None
            pngs = sorted(d.glob("*.png"))
            if atlas and pngs:
                self._assets.append(AssetItem(d, "anim", pngs, atlas))
        for f in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime,
                        reverse=True):
            if f.is_file() and f.suffix.lower() in self.IMAGE_EXTS:
                self._assets.append(AssetItem(f, "image"))
        # uploads/：用户上传素材（收编进本地资产）
        up = root / UPLOAD_DIR
        if up.is_dir():
            for f in sorted(up.iterdir(), key=lambda p: p.stat().st_mtime,
                            reverse=True):
                if f.is_file() and f.suffix.lower() in self.IMAGE_EXTS:
                    self._assets.append(AssetItem(f, "image"))
        self._rebuild_folder_combo()
        self._apply_filter()
        self.state_changed.emit(f"资产库：扫描到 {len(self._assets)} 项（{root.name}/）")

    def _rebuild_folder_combo(self) -> None:
        folders = sorted(set(self._load_index()["folders"].values()))
        self.cmb_folder.blockSignals(True)
        self.cmb_folder.clear()
        self.cmb_folder.addItem("全部文件夹")
        self.cmb_folder.addItems(folders)
        if self._folder_filter in folders:
            self.cmb_folder.setCurrentText(self._folder_filter)
        self.cmb_folder.blockSignals(False)

    def _on_folder_filter(self, text: str) -> None:
        self._folder_filter = "" if text == "全部文件夹" else text
        self._apply_filter()

    @staticmethod
    def _fuzzy_hit(text: str, query: str) -> bool:
        """模糊匹配：query 用空白分词，要求每个词都能在 text 中命中。

        命中方式（满足任一即可）：
        - 连续子串包含（忽略大小写）；
        - 顺序子序列：词的字符按序出现在 text 中（如 “红披”→ “红色_披风”）。
        中英文混排均适用。
        """
        if not query:
            return True
        t = text.lower()
        for tok in query.lower().split():
            if tok in t:
                continue
            i = 0
            for ch in t:
                if i < len(tok) and ch == tok[i]:
                    i += 1
                    if i == len(tok):
                        break
            if i < len(tok):
                return False
        return True

    def _apply_filter(self) -> None:
        mode = self.cmb_filter.currentText()
        query = self.ed_search.text().strip()
        idx = self._load_index()
        favs = set(idx["favorites"])
        current = self._selected_rels()
        self.lst.clear()
        hits = 0
        for a in self._assets:
            rel = self._rel(a.path)
            if mode == "动画资产" and a.kind != "anim":
                continue
            if mode == "单图" and a.kind != "image":
                continue
            if mode == "收藏" and rel not in favs:
                continue
            if mode in CAT_NAMES.values():
                cat = idx["categories"].get(rel, "generated")
                if CAT_NAMES.get(cat) != mode:
                    continue
            folder = idx["folders"].get(rel, "")
            if self._folder_filter and folder != self._folder_filter:
                continue
            if query:
                hay = " ".join([
                    a.label(), rel, folder,
                    str(idx["prompts"].get(rel, "") or ""),
                ])
                if not self._fuzzy_hit(hay, query):
                    continue
            tag = folder and f"  📁{folder}"
            it = QListWidgetItem(a.label() + ("  ♥" if rel in favs else "")
                                 + tag)
            it.setData(Qt.UserRole, rel)
            it.setToolTip(str(a.path))
            pm = self._thumb(a)
            if pm:
                it.setIcon(QIcon(pm))
            self.lst.addItem(it)
            hits += 1
        if query:
            self.state_changed.emit(
                f"资产库：搜索「{query}」命中 {hits} 项 / 共 {len(self._assets)} 项")
        if current:
            for i in range(self.lst.count()):
                if self.lst.item(i).data(Qt.UserRole) in current:
                    self.lst.item(i).setSelected(True)

    def _thumb(self, a: AssetItem) -> QPixmap | None:
        try:
            if a.kind == "anim" and a.frame_files:
                arr = cv2.imread(str(a.frame_files[0]), cv2.IMREAD_UNCHANGED)
            elif a.kind == "image" and a.path.suffix.lower() == ".gif":
                from pixelforge.processing.editing import load_gif_frames
                arr = load_gif_frames(a.path)[0]
            else:
                arr = cv2.imread(str(a.path), cv2.IMREAD_UNCHANGED)
            if arr is None:
                return None
            return _to_pixmap(arr).scaled(
                96, 96, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        except Exception:  # noqa: BLE001  缩略图失败不阻塞浏览
            return None

    # ---------- 选择辅助 ----------

    def _selected_rels(self) -> list[str]:
        return [it.data(Qt.UserRole) for it in self.lst.selectedItems()]

    def _selected_items(self) -> list[tuple[AssetItem, str]]:
        rels = set(self._selected_rels())
        return [(a, self._rel(a.path)) for a in self._assets
                if self._rel(a.path) in rels]

    def _first_selected(self) -> tuple[AssetItem, str] | None:
        sel = self._selected_items()
        return sel[0] if sel else None

    # ---------- 导入 ----------

    def _current_category(self) -> str | None:
        """当前筛选对应的分类键（全部/动画/单图/收藏 -> None）。"""
        mode = self.cmb_filter.currentText()
        for key, name in CAT_NAMES.items():
            if name == mode:
                return key
        return None

    def _import_files(self) -> None:
        """导入外部文件到当前分类（可多选图片；目录则整目录拷入）。"""
        from PySide6.QtWidgets import QFileDialog
        files, _ = QFileDialog.getOpenFileNames(
            self, "导入资产到当前分类（可多选）", "",
            "图片 (*.png *.jpg *.jpeg *.webp *.gif);;所有文件 (*)")
        if not files:
            return
        cat = self._current_category()
        # 云资产：导入到云目录；否则导入到扫描根目录
        base = (self._cloud_dir() if cat == "cloud" else None) \
            or self._scan_root()
        base.mkdir(parents=True, exist_ok=True)
        idx = self._load_index()
        n = 0
        for f in files:
            src = Path(f)
            target = base / src.name
            try:
                if target.resolve() == src.resolve():
                    continue        # 已在该目录
                shutil.copy2(src, target)
                rel = self._rel(target)
                if cat:
                    idx["categories"][rel] = cat
                if self._folder_filter:
                    idx["folders"][rel] = self._folder_filter
                n += 1
            except OSError as e:
                self.state_changed.emit(f"导入失败 {src.name}: {e}")
        self._save_index(idx)
        self.refresh()
        self.state_changed.emit(
            f"已导入 {n} 个文件"
            + (f"→ {CAT_NAMES.get(cat, '')}" if cat else "")
            + (f"｜文件夹「{self._folder_filter}」" if self._folder_filter
               else ""))

    # ---------- 收藏 ----------

    def _toggle_favorite(self) -> None:
        sel = self._selected_rels()
        if not sel:
            self.state_changed.emit("资产库：请先选中（可框选多选）")
            return
        favs = self._load_favorites()
        for rel in sel:
            favs.symmetric_difference_update({rel})
        self._save_favorites(favs)
        self._apply_filter()
        self.state_changed.emit(f"已切换收藏 {len(sel)} 项")

    # ---------- 文件夹 ----------

    def _new_folder(self) -> None:
        sel = self._first_selected()
        if sel is None:
            self.state_changed.emit("资产库：请先选中一项")
            return
        name, ok = QInputDialog.getText(self, "新建文件夹",
                                        "文件夹名称（用于资产归类）：")
        if not (ok and name.strip()):
            return
        idx = self._load_index()
        idx["folders"][sel[1]] = name.strip()
        self._save_index(idx)
        self._rebuild_folder_combo()
        self._apply_filter()
        self.state_changed.emit(f"已归入文件夹「{name.strip()}」: {sel[1]}")

    def _move_to_folder(self, folder: str) -> None:
        idx = self._load_index()
        for a, rel in self._selected_items():
            idx["folders"][rel] = folder
        self._save_index(idx)
        self._rebuild_folder_combo()
        self._apply_filter()
        self.state_changed.emit(f"已移动 {len(self._selected_items())} 项到「{folder}」")

    # ---------- 分类 ----------

    def _set_category(self, cat: str) -> None:
        idx = self._load_index()
        for a, rel in self._selected_items():
            idx["categories"][rel] = cat
        self._save_index(idx)
        self._apply_filter()
        self.state_changed.emit(f"已标记为{CAT_NAMES[cat]}: {len(self._selected_items())} 项")

    # ---------- 上云 ----------

    def _cloud_dir(self) -> Path | None:
        from PySide6.QtCore import QSettings
        d = QSettings("PixelForge", "PixelForge").value(
            "cloud_asset_dir", "")
        return Path(d) if d else None

    def _to_cloud(self) -> None:
        cdir = self._cloud_dir()
        if cdir is None:
            QMessageBox.information(
                self, "PixelForge",
                "请先在 设置 → 通用 → 云资产目录 配置网盘同步文件夹")
            return
        cdir.mkdir(parents=True, exist_ok=True)
        idx = self._load_index()
        n = 0
        for a, rel in self._selected_items():
            target = cdir / a.path.name
            try:
                if a.kind == "anim":
                    dst = cdir / a.path.name
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(a.path, dst)
                else:
                    shutil.copy2(a.path, target)
                idx["categories"][rel] = "cloud"
                n += 1
            except OSError as e:
                self.state_changed.emit(f"上云失败 {rel}: {e}")
        self._save_index(idx)
        self._apply_filter()
        self.state_changed.emit(f"已上云 {n} 项 → {cdir}")

    # ---------- 提示词回填 ----------

    def register_prompt(self, rel_or_paths: list[str], info: dict) -> None:
        """生成完成后登记该资产用的提示词（主界面调用）。"""
        idx = self._load_index()
        for p in rel_or_paths:
            rel = self._rel(Path(p))
            idx["prompts"][rel] = info
        self._save_index(idx)

    def import_external(self, paths, category: str = "local",
                        subdir: str = UPLOAD_DIR) -> list[str]:
        """把外部素材收编进资产库（主界面上传素材时调用）。

        - 库外文件：复制到 <扫描根>/<subdir>/（同名自动加序号），并登记分类
        - 内容相同（MD5 一致）的文件不重复收编，直接复用已有副本
        - 已在扫描根内的文件：原路径返回，不重复拷贝、不改分类

        返回与输入等长的收编后绝对路径列表（供生成流程直接引用）。
        """
        import hashlib
        root = self._scan_root().resolve()
        out: list[str] = []
        idx: dict | None = None
        changed = False
        for p in paths:
            p = Path(p)
            if not p.is_file():
                out.append(str(p))
                continue
            try:
                p.resolve().relative_to(root)
                out.append(str(p))            # 已在库内
                continue
            except ValueError:
                pass
            if idx is None:
                idx = self._load_index()
            dest_dir = root / subdir
            dest_dir.mkdir(parents=True, exist_ok=True)
            digest = hashlib.md5(p.read_bytes()).hexdigest()
            dest = None
            for f in dest_dir.iterdir():      # 同内容复用，避免重复堆积
                if not f.is_file() or f.suffix.lower() not in self.IMAGE_EXTS:
                    continue
                try:
                    if hashlib.md5(f.read_bytes()).hexdigest() == digest:
                        dest = f
                        break
                except OSError:
                    continue
            if dest is None:
                dest = dest_dir / p.name
                i = 1
                while dest.exists():
                    dest = dest_dir / f"{p.stem}_{i}{p.suffix}"
                    i += 1
                shutil.copy2(p, dest)
            rel = self._rel(dest)
            if idx["categories"].get(rel) != category:
                idx["categories"][rel] = category
                changed = True
            out.append(str(dest))
        if changed and idx is not None:
            self._save_index(idx)
        return out

    def _import_prompt(self) -> None:
        sel = self._first_selected()
        if sel is None:
            return
        a, rel = sel
        idx = self._load_index()
        info = idx["prompts"].get(rel)
        if info is None:
            # 兜底：同目录 task_*.json（云端任务响应落盘）
            for tj in sorted(a.path.glob("task_*.json")) if a.kind == "anim" else []:
                try:
                    info = {"prompt": None}  # 结构化提取成本高，先跳过
                    break
                except OSError:
                    pass
        if info is None:
            QMessageBox.information(
                self, "PixelForge",
                "该资产没有登记生成提示词\n（生成时自动登记；旧产物无记录）")
            return
        self.prompt_loaded.emit(info)
        self.state_changed.emit(f"已导入提示词: {rel}")

    # ---------- 重载 ----------

    def _reload_selected(self) -> None:
        sel = self._first_selected()
        if sel is None:
            self.state_changed.emit("资产库：请先选中一项")
            return
        a, rel = sel
        if a.kind == "anim":
            self.frames_to_review.emit(list(a.frame_files), a.atlas)
            arrs = []
            for p in a.frame_files:
                arr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if arr is not None:
                    arrs.append(arr)
            self.frames_to_player.emit(arrs)
            self.state_changed.emit(f"已重载 {a.path.name}（{len(a.frame_files)} 帧）"
                                    "到审核区与播放器")
        else:
            self.image_to_lab.emit(a.path)
            self.state_changed.emit(f"已发送 {a.path.name} 到图像工具坊")

    def _edit_selected(self) -> None:
        """打开像素编辑器（单图；动画资产编辑其首帧）。"""
        sel = self._first_selected()
        if sel is None:
            return
        a, rel = sel
        target = a.frame_files[0] if a.kind == "anim" and a.frame_files \
            else a.path if a.kind == "image" else None
        if target is None or target.suffix.lower() == ".gif":
            self.state_changed.emit("像素编辑器：该资产类型暂不支持")
            return
        from pixelforge.ui.pixel_editor import PixelEditorDialog
        dlg = PixelEditorDialog(str(target), self)
        dlg.edit_done.connect(lambda p: self.refresh())
        dlg.show()

    def _context_menu(self, pos) -> None:
        menu = QMenu(self)
        n_sel = len(self._selected_rels())
        menu.addAction(f"重载（已选 {n_sel} 项）").triggered.connect(
            self._reload_selected)
        act_edit = menu.addAction("✏ 编辑（像素编辑器）")
        act_mat = menu.addAction(
            f"🖼 作为生成素材（{n_sel} 张，AI 修图/编辑用）")
        act_mat_extra = menu.addAction(
            f"📎 作为附属生成素材（追加 {n_sel} 张，不清空现有参考图）")
        act_lab = menu.addAction("→ 导入工具坊")
        act_cloud = menu.addAction("☁ 上云（复制到云资产目录）")
        act_prompt = menu.addAction("📋 导入提示词（生成时用的）")
        menu.addSeparator()
        menu.addAction("♥ 收藏 / 取消收藏").triggered.connect(
            self._toggle_favorite)
        sub_folder = menu.addMenu("📁 移动到文件夹")
        sub_folder.addAction("（无）").triggered.connect(
            lambda: self._move_to_folder(""))
        for f in sorted(set(self._load_index()["folders"].values())):
            sub_folder.addAction(f"📁 {f}").triggered.connect(
                lambda _, fx=f: self._move_to_folder(fx))
        sub_cat = menu.addMenu("🏷 标记分类")
        for cat, name in CAT_NAMES.items():
            sub_cat.addAction(name).triggered.connect(
                lambda _, c=cat: self._set_category(c))
        menu.addSeparator()
        menu.addAction("打开所在目录").triggered.connect(self._open_dir)
        chosen = menu.exec(self.lst.mapToGlobal(pos))
        if chosen == act_edit:
            self._edit_selected()
        elif chosen == act_mat:
            self._as_gen_material()
        elif chosen == act_mat_extra:
            self._as_gen_material(extra=True)
        elif chosen == act_lab:
            self._to_lab()
        elif chosen == act_cloud:
            self._to_cloud()
        elif chosen == act_prompt:
            self._import_prompt()

    # ---------- 双击预览 ----------

    def _preview_selected(self) -> None:
        """双击资产 -> 在中央预览看图器打开（动画取首帧）。"""
        sel = self._first_selected()
        if sel is None:
            return
        a, _rel = sel
        target = (a.frame_files[0] if a.kind == "anim" and a.frame_files
                  else a.path)
        if target is None or not target.exists():
            self.state_changed.emit("该资产没有可预览的图片")
            return
        self.preview_requested.emit(str(target))
        self.state_changed.emit(f"已在预览打开 {target.name}")

    def _as_gen_material(self, extra: bool = False) -> None:
        """选中资产作为生成素材发给主界面。
        extra=False 替换现有参考图；True 追加（不清空当前参考图）。"""
        paths: list[Path] = []
        for a, _rel in self._selected_items():
            if a.kind == "anim" and a.frame_files:
                paths.extend(a.frame_files)      # 动画资产发全部帧
            elif a.kind == "image":
                paths.append(a.path)
        if not paths:
            self.state_changed.emit("没有可作为素材的图片")
            return
        (self.gen_material_extra if extra else self.gen_material).emit(paths)
        self.state_changed.emit(
            f"已作为{'附属' if extra else ''}生成素材发送 {len(paths)} 张图"
            + ("（追加到参考图）" if extra else "（AI 修图/编辑用）"))

    def _to_lab(self) -> None:
        for a, rel in self._selected_items():
            if a.kind == "image":
                self.image_to_lab.emit(a.path)
                self.state_changed.emit(f"已发送 {a.path.name} 到图像工具坊")
                return   # 工具坊一次处理一张
        self.state_changed.emit("工具坊仅支持单图（请选中一张图片资产）")

    def _open_dir(self) -> None:
        sel = self._first_selected()
        if not sel:
            return
        p = sel[0].path
        target = p if p.is_dir() else p.parent
        if sys.platform == "win32":
            os.startfile(str(target))  # noqa: S606  资源管理器打开
        else:
            import subprocess
            subprocess.Popen(["xdg-open", str(target)])

    def _rel(self, p: Path) -> str:
        try:
            return p.resolve().relative_to(self._scan_root().resolve()).as_posix()
        except ValueError:
            return p.name
