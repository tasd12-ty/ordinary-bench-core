"""
三阶段管线：plan_motion -> render_dynamic -> encode_video -> organize。
"""

import json
import logging
import shutil
import subprocess
from pathlib import Path

from motion.trajectory import plan_scene, ObjectMotionPlan

logger = logging.getLogger(__name__)

BLENDER_DIR = Path(__file__).resolve().parent / "blender"
RENDER_SCRIPT = BLENDER_DIR / "render_dynamic.py"
ASSETS_DIR = BLENDER_DIR / "assets"


def plan_motion(
    scene_id: str,
    n_objects: int,
    n_frames: int,
    fps: int,
    properties: dict,
    motion_mix: dict,
    bounds: float,
    min_dist: float,
    seed: int,
    plans_dir: Path,
    motion_params: dict = None,
    n_moving: int = None,
    min_reversals: int = None,
    camera_plan: dict = None,
) -> Path:
    """第一阶段：生成运动规划 JSON。"""
    plans = plan_scene(
        n_objects=n_objects,
        n_frames=n_frames,
        properties=properties,
        motion_mix=motion_mix,
        bounds=bounds,
        min_dist=min_dist,
        seed=seed,
        motion_params=motion_params,
        n_moving=n_moving,
        min_reversals=min_reversals,
    )

    plan_data = {
        "scene_id": scene_id,
        "n_objects": n_objects,
        "n_frames": n_frames,
        "fps": fps,
        "objects": [p.to_dict() for p in plans],
    }

    if camera_plan is not None:
        plan_data["camera"] = camera_plan

    plans_dir.mkdir(parents=True, exist_ok=True)
    plan_path = plans_dir / f"{scene_id}_plan.json"
    with open(plan_path, "w") as f:
        json.dump(plan_data, f, indent=2)

    logger.info(f"Motion plan saved: {plan_path}")
    return plan_path


def render_dynamic(
    plan_path: Path,
    render_output: Path,
    cfg: dict,
) -> Path:
    """第二阶段：调用 Blender 子进程渲染帧。"""
    blender = cfg["blender"]["executable"]
    rendering = cfg["rendering"]

    render_output.mkdir(parents=True, exist_ok=True)

    cmd = [
        blender,
        "--background",
        "--python", str(RENDER_SCRIPT),
        "--",
        "--plan_json", str(plan_path),
        "--base_scene_blendfile", str(ASSETS_DIR / "base_scene_v5.blend"),
        "--properties_json", str(ASSETS_DIR / "properties.json"),
        "--shape_dir", str(ASSETS_DIR / "shapes_v5"),
        "--material_dir", str(ASSETS_DIR / "materials_v5"),
        "--output_dir", str(render_output),
        "--width", str(rendering["width"]),
        "--height", str(rendering["height"]),
        "--samples", str(rendering["samples"]),
        "--camera_distance", str(rendering.get("camera_distance", 12.0)),
        "--elevation", str(rendering.get("elevation", 30.0)),
        "--azimuth", str(rendering.get("azimuth", rendering.get("azimuth_start", 45.0))),
    ]

    if cfg["blender"].get("use_gpu", False):
        cmd.extend(["--use_gpu", "1"])

    logger.info(f"Rendering: {plan_path.name} -> {render_output}")

    n_frames = 48  # 默认值
    try:
        with open(plan_path) as f:
            n_frames = json.load(f).get("n_frames", 48)
    except Exception:
        pass

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(3600, n_frames * 60),
        )
        if result.returncode != 0:
            log_dir = Path(cfg["output"]["dir"]) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"{plan_path.stem}_blender.log"
            with open(log_file, "w") as f:
                f.write(f"=== STDOUT ===\n{result.stdout}\n")
                f.write(f"=== STDERR ===\n{result.stderr}\n")
            logger.error(f"Render failed. Log: {log_file}")
            logger.error(f"Stderr tail:\n{result.stderr[-500:]}")
            raise RuntimeError(f"Render failed: {plan_path.name}")
        logger.info(f"Render complete: {plan_path.name}")
    except subprocess.TimeoutExpired:
        logger.error(f"Render timed out: {plan_path.name}")
        raise

    return render_output


def encode_video(
    frames_dir: Path,
    output_path: Path,
    fps: int = 24,
    crf: int = 18,
) -> bool:
    """使用 ffmpeg 将 PNG 帧编码为 MP4。成功时返回 True。"""
    frame_pattern = str(frames_dir / "frame_%04d.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", frame_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", str(crf),
        str(output_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.error(f"ffmpeg failed: {result.stderr[-300:]}")
            return False
        logger.info(f"Video encoded: {output_path}")
        return True
    except FileNotFoundError:
        logger.error("未找到 ffmpeg，请安装 ffmpeg 以启用视频编码。")
        return False
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg timed out")
        return False


def organize_scene(
    scene_id: str,
    render_output: Path,
    output_dir: Path,
    video_cfg: dict = None,
    fps: int = 24,
) -> dict:
    """第三阶段：填充事件、复制帧和场景 JSON 到最终输出目录。"""
    src_json = render_output / "temporal_scene.json"
    dst_json = output_dir / "scenes" / f"{scene_id}.json"
    dst_json.parent.mkdir(parents=True, exist_ok=True)

    if not src_json.exists():
        logger.warning(f"temporal_scene.json not found in {render_output}")
        return {"scene_id": scene_id, "status": "missing_json"}

    # 加载并丰富时序场景数据
    with open(src_json) as f:
        temporal_scene = json.load(f)

    # 通过 QRR + TRR 变化检测填充事件
    from extraction.temporal import detect_qrr_changes, detect_trr_changes
    events = []
    try:
        qrr_events = detect_qrr_changes(temporal_scene, tau=0.10)
        events.extend(qrr_events)
        logger.info(f"  {len(qrr_events)} QRR reversal events detected")
    except Exception as e:
        logger.warning(f"  QRR event detection failed: {e}")
    try:
        trr_events = detect_trr_changes(temporal_scene)
        events.extend(trr_events)
        logger.info(f"  {len(trr_events)} TRR change events detected")
    except Exception as e:
        logger.warning(f"  TRR event detection failed: {e}")
    temporal_scene["events"] = events

    # 写入丰富后的 JSON
    with open(dst_json, "w") as f:
        json.dump(temporal_scene, f, indent=2)

    # 复制帧图像
    src_frames = render_output / "frames"
    dst_frames = output_dir / "images" / scene_id
    if src_frames.exists():
        if dst_frames.exists():
            shutil.rmtree(dst_frames)
        shutil.copytree(src_frames, dst_frames)

    # 验证输出完整性
    n_expected = temporal_scene.get("n_frames", 0)
    actual_frames = list(dst_frames.glob("frame_*.png")) if dst_frames.exists() else []
    if len(actual_frames) < n_expected:
        logger.warning(f"  Missing frames: expected {n_expected}, got {len(actual_frames)}")

    # 视频编码
    video_path = None
    if video_cfg and video_cfg.get("encode", False) and dst_frames.exists():
        videos_dir = output_dir / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        video_path = videos_dir / f"{scene_id}.mp4"
        crf = video_cfg.get("crf", 18)
        encode_ok = encode_video(dst_frames, video_path, fps=fps, crf=crf)
        if encode_ok and not video_cfg.get("keep_frames", True):
            shutil.rmtree(dst_frames)
            logger.info(f"  Frames removed (keep_frames=false)")

    # 清理临时渲染目录
    try:
        shutil.rmtree(render_output)
    except Exception as e:
        logger.warning(f"Cleanup failed for {render_output}: {e}")

    return {
        "scene_id": scene_id,
        "status": "ok",
        "n_events": len(temporal_scene.get("events", [])),
        "n_frames": len(actual_frames),
        "video": str(video_path) if video_path else None,
    }


def build_scene(
    scene_id: str,
    split_name: str,
    split_cfg: dict,
    cfg: dict,
    seed: int,
) -> dict:
    """为单个场景运行完整的 3 阶段管线。"""
    output_dir = Path(cfg["output"]["dir"])
    animation = cfg.get("animation", {})
    motion_cfg = cfg.get("motion", {})
    objects_cfg = cfg.get("objects", {})
    rendering = cfg["rendering"]
    video_cfg = cfg.get("video", {})

    n_frames = animation.get("n_frames", 48)
    fps = animation.get("fps", 24)
    n_objects = split_cfg.get("n_objects", None)
    if n_objects is None:
        import random as _rng
        _rng.seed(seed)
        n_objects = _rng.randint(
            objects_cfg.get("min_count", 3),
            objects_cfg.get("max_count", 5),
        )

    # 加载属性
    props_path = ASSETS_DIR / "properties.json"
    with open(props_path) as f:
        properties = json.load(f)

    # 从配置中所有已知运动类型构建运动混合比例
    motion_mix = {}
    for mtype in ("static", "linear", "circular", "accelerated_linear",
                   "waypoint", "bounce"):
        if mtype in motion_cfg:
            motion_mix[mtype] = motion_cfg[mtype]
    if not motion_mix:
        motion_mix = {"static": 0.2, "linear": 0.6, "circular": 0.2}

    motion_params_keys = {"speed_min", "speed_max", "omega_min", "omega_max",
                          "radius_min", "radius_max", "accel_min", "accel_max"}
    motion_params = {k: motion_cfg[k] for k in motion_params_keys if k in motion_cfg} or None

    n_moving = motion_cfg.get("n_moving", None)
    min_reversals = motion_cfg.get("min_reversals", None)

    # 相机运动规划
    camera_cfg = cfg.get("camera", None)
    camera_plan = None
    if camera_cfg and camera_cfg.get("type", "static") != "static":
        from motion.camera import build_camera_plan
        camera_plan = build_camera_plan(camera_cfg, n_frames, fps)

    plans_dir = output_dir / "plans"
    render_temp = output_dir / "render_temp" / scene_id

    # 第一阶段
    plan_path = plan_motion(
        scene_id=scene_id,
        n_objects=n_objects,
        n_frames=n_frames,
        fps=fps,
        properties=properties,
        motion_mix=motion_mix,
        bounds=objects_cfg.get("bounds", 3.0),
        min_dist=objects_cfg.get("min_dist", 0.25),
        seed=seed,
        plans_dir=plans_dir,
        motion_params=motion_params,
        n_moving=n_moving,
        min_reversals=min_reversals,
        camera_plan=camera_plan,
    )

    # 多相机：若已配置则构建额外相机规划
    multi_camera = cfg.get("multi_camera", None)
    camera_plans_list = [(None, camera_plan)]  # (后缀, 规划) - None 后缀 = 主相机
    if multi_camera:
        from motion.camera import build_camera_plan as _build_cam
        for i, cam_cfg in enumerate(multi_camera):
            suffix = f"_cam{i+1}"
            cp = _build_cam(cam_cfg, n_frames, fps)
            camera_plans_list.append((suffix, cp))

    results = []
    for cam_suffix, cam_plan in camera_plans_list:
        cur_scene_id = scene_id if cam_suffix is None else f"{scene_id}{cam_suffix}"
        cur_render_temp = output_dir / "render_temp" / cur_scene_id

        # 若为副相机，用新相机更新规划 JSON
        if cam_suffix is not None:
            with open(plan_path) as f:
                plan_data = json.load(f)
            plan_data["camera"] = cam_plan
            plan_data["scene_id"] = cur_scene_id
            cam_plan_path = plans_dir / f"{cur_scene_id}_plan.json"
            with open(cam_plan_path, "w") as f:
                json.dump(plan_data, f, indent=2)
            cur_plan_path = cam_plan_path
        else:
            cur_plan_path = plan_path

        # 第二阶段：渲染
        render_output = render_dynamic(cur_plan_path, cur_render_temp, cfg)

        # 第三阶段：整理
        result = organize_scene(cur_scene_id, render_output, output_dir,
                                video_cfg=video_cfg, fps=fps)
        results.append(result)

    # 返回主相机（第一个）的结果
    return results[0]


def build_split(split_name: str, split_cfg: dict, cfg: dict) -> dict:
    """构建一个 split 的所有场景。"""
    n_scenes = split_cfg["n_scenes"]
    start_idx = split_cfg.get("start_idx", 0)
    base_seed = cfg["output"].get("seed", 42)

    output_dir = Path(cfg["output"]["dir"])
    split_entries = []

    for i in range(n_scenes):
        idx = start_idx + i
        scene_id = f"d{split_cfg.get('n_objects', cfg['objects'].get('max_count', 5)):02d}_{idx:06d}"
        seed = base_seed + idx

        logger.info(f"  [{i+1}/{n_scenes}] Building {scene_id}")
        try:
            result = build_scene(scene_id, split_name, split_cfg, cfg, seed)
            split_entries.append(result)
        except Exception as e:
            logger.error(f"  Failed {scene_id}: {e}", exc_info=True)
            split_entries.append({"scene_id": scene_id, "status": "failed", "error": str(e)})

    # 保存 split 索引
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    split_file = splits_dir / f"{split_name}.json"
    with open(split_file, "w") as f:
        json.dump(split_entries, f, indent=2)
    logger.info(f"Split index: {split_file} ({len(split_entries)} scenes)")

    return {
        "n_scenes": len(split_entries),
        "n_ok": sum(1 for e in split_entries if e.get("status") == "ok"),
        "n_failed": sum(1 for e in split_entries if e.get("status") != "ok"),
    }
