"""Unity .unitypackage 打包器（标准 tar.gz 结构，拖入 Unity 即自动导入）。

包内每个资产一个 GUID 目录：
    <guid>/pathname     资产在 Unity 项目内的路径（UTF-8 文本）
    <guid>/asset        资产二进制内容
    <guid>/asset.meta   Unity 元数据（PNG 用 TextureImporter sprite 配置）
"""
from __future__ import annotations

import io
import tarfile
import uuid
from pathlib import Path


def _texture_meta(guid: str) -> str:
    """PNG 资产 meta：Sprite 类型 + Point 过滤（与导入器设置一致）。"""
    return (
        "fileFormatVersion: 2\n"
        f"guid: {guid}\n"
        "TextureImporter:\n"
        "  externalObjects: {}\n"
        "  serializedVersion: 13\n"
        "  textureType: 8\n"          # 8 = Sprite
        "  textureShape: 1\n"
        "  spriteImportMode: 1\n"     # 1 = Single
        "  filterMode: 0\n"           # 0 = Point（像素画）
        "  isReadable: 0\n"
        "  alphaIsTransparency: 1\n"
        "  mipmaps:\n"
        "    enabled: 0\n"
    )


def _default_meta(guid: str) -> str:
    return (
        "fileFormatVersion: 2\n"
        f"guid: {guid}\n"
        "DefaultImporter:\n"
        "  externalObjects: {}\n"
    )


def build_unity_package(files: list[tuple[Path, str]], out: Path) -> Path:
    """把 (本地文件, Unity 项目内路径) 列表打包为 .unitypackage。

    out 以 .unitypackage 结尾；同名资产生成各自独立 GUID。
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    used_guids: set[str] = set()
    with tarfile.open(out, "w:gz") as tar:
        for src, unity_path in files:
            src = Path(src)
            if not src.is_file():
                raise FileNotFoundError(f"打包资产缺失: {src}")
            guid = uuid.uuid4().hex
            while guid in used_guids:  # 理论上不会碰撞，防御重复
                guid = uuid.uuid4().hex
            used_guids.add(guid)
            meta = _texture_meta(guid) if src.suffix.lower() in (".png", ".jpg") \
                else _default_meta(guid)
            data = src.read_bytes()

            def _add(name: str, payload: bytes) -> None:
                info = tarfile.TarInfo(name=f"{guid}/{name}")
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))

            _add("pathname", unity_path.replace("\\", "/").encode("utf-8"))
            _add("asset", data)
            _add("asset.meta", meta.encode("utf-8"))
    return out
