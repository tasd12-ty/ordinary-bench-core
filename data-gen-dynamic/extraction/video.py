"""
视频帧提取工具。

从 MP4 视频中提取帧，用于 VLM 评测（如均匀采样）。
"""

import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional


def extract_frames(
    video_path: str,
    frame_indices: List[int],
    output_dir: Optional[str] = None,
    fps: int = 24,
) -> List[str]:
    """
    按索引从视频中提取指定帧。

    Args:
        video_path: MP4 视频路径。
        frame_indices: 待提取帧的 0 起始索引列表。
        output_dir: 提取 PNG 的输出目录，为 None 时使用临时目录。
        fps: 视频帧率（用于时间戳计算）。

    Returns:
        输出 PNG 文件路径列表。
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="frames_")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = []
    for idx in frame_indices:
        timestamp = idx / fps
        out_path = out / f"frame_{idx:04d}.png"
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{timestamp:.4f}",
            "-i", str(video_path),
            "-frames:v", "1",
            "-q:v", "2",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and out_path.exists():
            paths.append(str(out_path))

    return paths


def extract_uniform_frames(
    video_path: str,
    n_frames: int,
    output_dir: Optional[str] = None,
    total_frames: Optional[int] = None,
    fps: int = 24,
) -> List[str]:
    """
    从视频中均匀提取 N 帧。

    Args:
        video_path: MP4 视频路径。
        n_frames: 要提取的帧数。
        output_dir: 提取 PNG 的输出目录，为 None 时使用临时目录。
        total_frames: 视频总帧数，为 None 时自动检测。
        fps: 视频帧率。

    Returns:
        输出 PNG 文件路径列表。
    """
    if total_frames is None:
        total_frames = get_frame_count(video_path)
        if total_frames is None:
            total_frames = fps * 15  # 回退：假设视频为 15 秒

    if n_frames >= total_frames:
        indices = list(range(total_frames))
    elif n_frames == 1:
        indices = [total_frames // 2]
    else:
        step = (total_frames - 1) / (n_frames - 1)
        indices = [round(i * step) for i in range(n_frames)]

    return extract_frames(video_path, indices, output_dir, fps)


def get_frame_count(video_path: str) -> Optional[int]:
    """使用 ffprobe 获取视频总帧数。"""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "csv=p=0",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception:
        pass
    return None


def get_video_info(video_path: str) -> Optional[dict]:
    """通过 ffprobe 获取视频元数据（时长、帧率、分辨率）。"""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video_path),
    ]
    try:
        import json
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            stream = data.get("streams", [{}])[0]
            fmt = data.get("format", {})
            r_fps = stream.get("r_frame_rate", "24/1")
            num, den = r_fps.split("/")
            return {
                "width": int(stream.get("width", 0)),
                "height": int(stream.get("height", 0)),
                "fps": int(num) / int(den),
                "duration": float(fmt.get("duration", stream.get("duration", 0))),
                "n_frames": int(stream.get("nb_frames", 0)),
            }
    except Exception:
        pass
    return None
