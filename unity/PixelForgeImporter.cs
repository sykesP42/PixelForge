// PixelForge Studio -> Unity 自动导入桥接（阶段4 完整实现）
// 用法：将本文件放入 Unity 项目 Assets/Editor/ 目录。
//
// 监听 Assets/Textures/Generated/ 下的 *.atlas.json（由 GUI "导出到 Unity" 生成），
// 自动完成：帧贴图 Sprite 化（Point 过滤 / 无压缩 / pivot 设置）
//        -> AnimationClip（按 fps 循环）
//        -> AnimatorController（默认状态）。
// 也可通过菜单 PixelForge/导入 Generated 目录全部图集 手动触发。
//
// atlas.json Schema（由 pixelforge ReviewWidget.export_to 生成）：
// {
//   "name": "knight_idle", "fps": 8, "frame_count": 4,
//   "frames": [
//     { "file": "knight_idle_000.png", "x": 12, "y": 3, "width": 40, "height": 62,
//       "pivotX": 0.5, "pivotY": 1.0,
//       "hull": [x0, y0, x1, y1, ...] }
//   ]
// }
// 坐标约定：x/y 为内容 bbox 在画布 PNG 中的像素位置，原点=左上角，y 向下；
//           pivot 为内容 bbox 内的归一化轴点（1.0=脚底）；
//           hull 为内容轮廓多边形的扁平坐标（帧画布像素、顶左原点、可省略），
//           导入时经 Sprite.OverridePhysicsShape 设为物理形状，
//           PolygonCollider2D 挂到 SpriteRenderer 上即自动得到角色轮廓碰撞体。
//           Unity pivot 原点=左下角，y 向上，故需翻转换算（见 ComputePivot）。

#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Animations;
using UnityEngine;

namespace PixelForge
{
    // ---------- atlas.json 反序列化模型（字段名与 Python 导出严格一致） ----------

    [Serializable]
    public class FrameMeta
    {
        public string file;
        public int x;
        public int y;
        public int width;
        public int height;
        public float pivotX;
        public float pivotY;
        public List<float> hull; // 扁平 [x0,y0,x1,y1,...]，帧画布像素（可空）
    }

    [Serializable]
    public class AtlasMeta
    {
        public string name;
        public int fps;
        public int frame_count;
        public List<FrameMeta> frames;
    }

    public class PixelForgeImporter : AssetPostprocessor
    {
        internal const string WatchDir = "Assets/Textures/Generated";
        private const string DoneTag = "pixelforge_v1"; // 防止重导入死循环

        // ---------- 自动监听 ----------

        private static void OnPostprocessAllAssets(
            string[] importedAssets, string[] deletedAssets,
            string[] movedAssets, string[] movedFromAssetPaths,
            bool didDomainReload)
        {
            foreach (string path in importedAssets)
            {
                if (path.EndsWith(".atlas.json", StringComparison.OrdinalIgnoreCase)
                    && path.Replace('\\', '/').StartsWith(WatchDir, StringComparison.OrdinalIgnoreCase))
                {
                    string p = path;
                    EditorApplication.delayCall += () => ImportAtlas(p);
                }
            }
        }

        [MenuItem("PixelForge/导入 Generated 目录全部图集")]
        public static void ImportAll()
        {
            if (!Directory.Exists(WatchDir))
            {
                Debug.LogWarning($"[PixelForge] 目录不存在: {WatchDir}");
                return;
            }
            int n = 0;
            foreach (string f in Directory.GetFiles(WatchDir, "*.atlas.json", SearchOption.AllDirectories))
            {
                ImportAtlas(f.Replace('\\', '/'));
                n++;
            }
            AssetDatabase.Refresh();
            Debug.Log($"[PixelForge] 扫描完成，共导入 {n} 个图集");
        }

        // ---------- 单个图集导入 ----------

        public static void ImportAtlas(string jsonPath)
        {
            if (!File.Exists(jsonPath))
                return;

            AssetImporter jsonImp = AssetImporter.GetAtPath(jsonPath);
            if (jsonImp != null && jsonImp.userData == DoneTag)
                return; // 已处理过

            AtlasMeta meta;
            try
            {
                meta = JsonUtility.FromJson<AtlasMeta>(File.ReadAllText(jsonPath));
            }
            catch (Exception e)
            {
                Debug.LogError($"[PixelForge] JSON 解析失败 {jsonPath}: {e.Message}");
                return;
            }
            if (meta == null || meta.frames == null || meta.frames.Count == 0)
            {
                Debug.LogError($"[PixelForge] atlas 中没有帧: {jsonPath}");
                return;
            }

            string dir = Path.GetDirectoryName(jsonPath)?.Replace('\\', '/') ?? "Assets";

            // 1) 逐帧贴图 -> Sprite（Point 过滤 + 无压缩，保证像素画清晰）
            var sprites = new List<Sprite>();
            foreach (FrameMeta m in meta.frames)
            {
                string texPath = $"{dir}/{m.file}";
                TextureImporter texImp = AssetImporter.GetAtPath(texPath) as TextureImporter;
                if (texImp == null)
                {
                    Debug.LogError($"[PixelForge] 找不到帧贴图: {texPath}");
                    continue;
                }

                Texture2D tex = AssetDatabase.LoadAssetAtPath<Texture2D>(texPath);
                texImp.textureType = TextureImporterType.Sprite;
                texImp.spriteImportMode = SpriteImportMode.Single;
                texImp.spritePivot = ComputePivot(tex, m);
                texImp.filterMode = FilterMode.Point;
                texImp.textureCompression = TextureImporterCompression.Uncompressed;
                texImp.alphaIsTransparency = true;
                texImp.mipmapEnabled = false;
                texImp.spritePixelsPerUnit = 16f; // 像素游戏常用 PPU，可按项目调整
                texImp.userData = DoneTag;
                texImp.SaveAndReimport();

                Sprite sp = AssetDatabase.LoadAssetAtPath<Sprite>(texPath);
                if (sp != null)
                {
                    if (m.hull != null && m.hull.Count >= 6)
                        ApplyHull(sp, m);
                    sprites.Add(sp);
                }
                else Debug.LogError($"[PixelForge] Sprite 加载失败: {texPath}");
            }
            if (sprites.Count == 0)
            {
                Debug.LogError($"[PixelForge] 图集 {meta.name} 未加载到任何精灵");
                return;
            }

            // 2) AnimationClip（按 fps 循环）
            AnimationClip clip = BuildClip(meta, sprites, dir);

            // 3) AnimatorController（单状态默认播放）
            BuildController(meta.name, clip, dir);

            if (jsonImp != null)
            {
                jsonImp.userData = DoneTag;
                EditorUtility.SetDirty(jsonImp);
            }
            Debug.Log($"[PixelForge] 已导入图集 '{meta.name}': {sprites.Count} 帧 @ {meta.fps} FPS");
        }

        /// <summary>JSON 顶左原点 -> Unity 底左原点的 pivot 换算（自测脚本复用）。</summary>
        internal static Vector2 ComputePivot(Texture2D tex, FrameMeta m)
        {
            if (tex == null)
                return new Vector2(0.5f, 0f);
            float px = (m.x + m.pivotX * m.width) / tex.width;
            // pivotY 在内容 bbox 内自顶向下度量（1.0=脚底）：
            // pivot 像素行 = y + pivotY*h，换算到底左原点 => py = (texH - y - pivotY*h)/texH
            float py = (tex.height - m.y - m.pivotY * m.height) / tex.height;
            return new Vector2(Mathf.Clamp01(px), Mathf.Clamp01(py));
        }

        /// <summary>把扁平 hull 多边形（帧画布像素、顶左原点）设为 Sprite 物理形状。
        /// 本地坐标以 pivot 像素点为原点：x 右为正，y 上为正（翻转），单位=世界单位。</summary>
        private static void ApplyHull(Sprite sp, FrameMeta m)
        {
            try
            {
                float ppu = sp.pixelsPerUnit > 0f ? sp.pixelsPerUnit : 16f;
                float pivotPx = m.x + m.pivotX * m.width;   // pivot 像素坐标（帧画布内）
                float pivotPy = m.y + m.pivotY * m.height;
                var pts = new List<Vector2>(m.hull.Count / 2);
                for (int i = 0; i + 1 < m.hull.Count; i += 2)
                {
                    float lx = (m.hull[i] - pivotPx) / ppu;
                    float ly = (pivotPy - m.hull[i + 1]) / ppu; // 顶左 -> 底左翻转
                    pts.Add(new Vector2(lx, ly));
                }
                sp.OverridePhysicsShape(new List<Vector2[]> { pts });
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[PixelForge] hull 物理形状设置失败（已跳过）: {e.Message}");
            }
        }

        private static AnimationClip BuildClip(AtlasMeta meta, List<Sprite> sprites, string dir)
        {
            AnimationClip clip = new AnimationClip { frameRate = Mathf.Max(1, meta.fps) };
            float step = 1f / Mathf.Max(1, meta.fps);
            var keys = new ObjectReferenceKeyframe[sprites.Count];
            for (int i = 0; i < sprites.Count; i++)
                keys[i] = new ObjectReferenceKeyframe { time = i * step, value = sprites[i] };

            EditorCurveBinding binding = EditorCurveBinding.PPtrCurve("", typeof(SpriteRenderer), "m_Sprite");
            AnimationUtility.SetObjectReferenceCurve(clip, binding, keys);

            AnimationClipSettings settings = AnimationUtility.GetAnimationClipSettings(clip);
            settings.loopTime = true;
            AnimationUtility.SetAnimationClipSettings(clip, settings);

            string clipPath = $"{dir}/{meta.name}.anim";
            AnimationClip existing = AssetDatabase.LoadAssetAtPath<AnimationClip>(clipPath);
            if (existing != null)
            {
                EditorUtility.CopySerialized(clip, existing); // 覆盖旧动画，保持 GUID
                UnityEngine.Object.DestroyImmediate(clip);
                clip = existing;
            }
            else
            {
                AssetDatabase.CreateAsset(clip, clipPath);
            }
            return clip;
        }

        private static void BuildController(string name, AnimationClip clip, string dir)
        {
            string ctrlPath = $"{dir}/{name}.controller";
            AnimatorController ctrl = AssetDatabase.LoadAssetAtPath<AnimatorController>(ctrlPath);
            if (ctrl == null)
                ctrl = AnimatorController.CreateAnimatorControllerAtPath(ctrlPath);

            AnimatorState state = ctrl.layers[0].stateMachine.AddState(name);
            state.motion = clip;
            ctrl.layers[0].stateMachine.defaultState = state;
            AssetDatabase.SaveAssets();
        }
    }
}
#endif
