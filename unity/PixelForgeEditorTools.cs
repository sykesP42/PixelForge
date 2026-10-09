// PixelForge 编辑器工具：示例场景引导 + 生成链路一键自测。
// 用法：将本文件与 PixelForgeImporter.cs 一起放入 Unity 项目 Assets/Editor/ 目录。
//
// 菜单：
//   PixelForge/创建示例场景   —— 搭建 2D 演示场景（地面 + 玩家 + 碰撞体 + 动画控制器）
//   PixelForge/运行链路自测   —— 断言 Generated 目录全部图集的
//                                Sprite 化 / pivot 换算 / hull 物理形状 / 动画clip / 控制器
//
// 自测是 PixelForge 导出契约（atlas.json -> Unity 资产）的自动化验收，
// 全绿即代表 Python 侧导出的每一帧在 Unity 内按预期落地。

#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using UnityEditor;
using UnityEditor.Animations;
using UnityEngine;

namespace PixelForge
{
    public static class PixelForgeSceneBootstrap
    {
        [MenuItem("PixelForge/创建示例场景")]
        public static void CreateSampleScene()
        {
            var old = GameObject.Find("PF_Player");
            if (old != null)
                UnityEngine.Object.DestroyImmediate(old);
            var oldGround = GameObject.Find("PF_Ground");
            if (oldGround != null)
                UnityEngine.Object.DestroyImmediate(oldGround);

            // 地面：纯色长条 + 盒碰撞体（验证 PolygonCollider2D 站立）
            var ground = new GameObject("PF_Ground");
            var gsr = ground.AddComponent<SpriteRenderer>();
            gsr.sprite = BuildSolidSprite(640, 32, new Color32(96, 128, 88, 255));
            gsr.sortingOrder = -1;
            ground.AddComponent<BoxCollider2D>();
            ground.transform.position = new Vector3(0f, -1.2f, 0f);

            // 玩家：精灵 + 动画 + 物理（hull 轮廓自动成为 PolygonCollider2D 路径）
            var player = new GameObject("PF_Player");
            var sr = player.AddComponent<SpriteRenderer>();
            var anim = player.AddComponent<Animator>();
            var rb = player.AddComponent<Rigidbody2D>();
            rb.gravityScale = 3f;
            player.AddComponent<PolygonCollider2D>(); // 形状 = Sprite.OverridePhysicsShape

            var ctrl = FindFirstController();
            if (ctrl != null)
                anim.runtimeAnimatorController = ctrl;
            else
                Debug.LogWarning("[PixelForge] 未找到 AnimatorController，"
                                 + "请先执行 PixelForge/导入 Generated 目录全部图集");

            var cam = Camera.main;
            if (cam != null)
            {
                cam.orthographic = true;
                cam.orthographicSize = 2.5f;
                cam.transform.position = new Vector3(0f, 0.4f, -10f);
            }

            Selection.activeGameObject = player;
            Debug.Log("[PixelForge] 示例场景已创建：PF_Player 落到 PF_Ground 上"
                      + "（PolygonCollider2D 采用 hull 轮廓），按 Play 观看动画与碰撞。");
        }

        private static RuntimeAnimatorController FindFirstController()
        {
            if (!Directory.Exists(PixelForgeImporter.WatchDir))
                return null;
            foreach (string f in Directory.GetFiles(
                         PixelForgeImporter.WatchDir, "*.controller",
                         SearchOption.AllDirectories))
            {
                var c = AssetDatabase.LoadAssetAtPath<AnimatorController>(f.Replace('\\', '/'));
                if (c != null)
                    return c;
            }
            return null;
        }

        private static Sprite BuildSolidSprite(int w, int h, Color32 color)
        {
            var tex = new Texture2D(w, h, TextureFormat.RGBA32, false);
            var colors = new Color32[w * h];
            for (int i = 0; i < colors.Length; i++)
                colors[i] = color;
            tex.SetPixels32(colors);
            tex.Apply();
            return Sprite.Create(tex, new Rect(0, 0, w, h),
                                 new Vector2(0.5f, 0.5f), 100f);
        }
    }

    public static class PixelForgeSelfTest
    {
        [MenuItem("PixelForge/运行链路自测")]
        public static void Run()
        {
            if (!Directory.Exists(PixelForgeImporter.WatchDir))
            {
                EditorUtility.DisplayDialog("PixelForge 链路自测",
                    $"目录不存在: {PixelForgeImporter.WatchDir}\n"
                    + "请先在 PixelForge 中「导出到 Unity」。", "确定");
                return;
            }

            PixelForgeImporter.ImportAll(); // 幂等（DoneTag 跳过已导入）
            AssetDatabase.Refresh();

            int total = 0, failed = 0;
            var report = new StringBuilder();

            foreach (string jsonPath in Directory.GetFiles(
                         PixelForgeImporter.WatchDir, "*.atlas.json",
                         SearchOption.AllDirectories))
            {
                string p = jsonPath.Replace('\\', '/');
                AtlasMeta meta;
                try
                {
                    meta = JsonUtility.FromJson<AtlasMeta>(File.ReadAllText(p));
                }
                catch (Exception e)
                {
                    report.AppendLine($"✘ JSON 解析失败 {p}: {e.Message}");
                    failed++;
                    continue;
                }
                if (meta == null || meta.frames == null || meta.frames.Count == 0)
                    continue;
                string dir = Path.GetDirectoryName(p)?.Replace('\\', '/') ?? "Assets";

                foreach (FrameMeta m in meta.frames)
                {
                    total++;
                    string problem = CheckFrame(meta, m, dir);
                    if (problem != null)
                    {
                        failed++;
                        report.AppendLine($"✘ [{meta.name}] {m.file}: {problem}");
                    }
                }

                string clipPath = $"{dir}/{meta.name}.anim";
                var clip = AssetDatabase.LoadAssetAtPath<AnimationClip>(clipPath);
                if (clip == null)
                {
                    failed++;
                    report.AppendLine($"✘ [{meta.name}] 缺 AnimationClip {clipPath}");
                }
                else if (Mathf.Abs(clip.frameRate - Mathf.Max(1, meta.fps)) > 0.5f)
                {
                    failed++;
                    report.AppendLine(
                        $"✘ [{meta.name}] clip 帧率 {clip.frameRate} != {meta.fps}");
                }

                string ctrlPath = $"{dir}/{meta.name}.controller";
                if (AssetDatabase.LoadAssetAtPath<AnimatorController>(ctrlPath) == null)
                {
                    failed++;
                    report.AppendLine($"✘ [{meta.name}] 缺 AnimatorController {ctrlPath}");
                }
            }

            report.AppendLine();
            report.AppendLine($"自测完成：{(total - failed)} / {total} 项通过"
                              + (failed == 0 ? " —— 链路验收全绿 ✔" : ""));
            string msg = report.ToString();
            if (failed == 0)
                Debug.Log($"[PixelForge] {msg}");
            else
                Debug.LogWarning($"[PixelForge] {msg}");
            EditorUtility.DisplayDialog("PixelForge 链路自测", msg, "确定");
        }

        /// <summary>单帧检查，返回问题描述；null = 通过。</summary>
        private static string CheckFrame(AtlasMeta meta, FrameMeta m, string dir)
        {
            string texPath = $"{dir}/{m.file}";
            var imp = AssetImporter.GetAtPath(texPath) as TextureImporter;
            if (imp == null)
                return "找不到贴图";
            if (imp.textureType != TextureImporterType.Sprite)
                return "贴图未按 Sprite 导入";
            if (imp.filterMode != FilterMode.Point)
                return "过滤模式不是 Point（像素画会糊）";

            var sp = AssetDatabase.LoadAssetAtPath<Sprite>(texPath);
            if (sp == null)
                return "Sprite 加载失败";

            // pivot 契约：TextureImporter.spritePivot（归一化、rect 相对）
            var tex = AssetDatabase.LoadAssetAtPath<Texture2D>(texPath);
            Vector2 want = PixelForgeImporter.ComputePivot(tex, m);
            if ((imp.spritePivot - want).sqrMagnitude > 0.0004f)
                return $"pivot 不符: 期望({want.x:F3},{want.y:F3}) "
                       + $"实际({imp.spritePivot.x:F3},{imp.spritePivot.y:F3})";

            // hull 契约：有 hull 的帧必须有物理形状（PolygonCollider2D 数据源）
            if (m.hull != null && m.hull.Count >= 6 && sp.GetPhysicsShapeCount() < 1)
                return "hull 存在但 Sprite 无物理形状（需重新导入该图集）";

            return null;
        }
    }
}
#endif
