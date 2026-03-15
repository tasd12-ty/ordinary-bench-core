"""
Video frame extraction utilities.

Extract frames from MP4 videos for VLM evaluation (e.g., uniform sampling).
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
    Extract specific frames by index from a video.

    Args:
        video_path: Path to MP4 video.
        frame_indices: List of 0-based frame indices to extract.
        output_dir: Directory for extracted PNGs. Uses temp dir if None.
        fps: Video framerate (for timestamp calculation).

    Returns:
        List of output PNG file paths.
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
    Extract N uniformly spaced frames from a video.

    Args:
        video_path: Path to MP4 video.
        n_frames: Number of frames to extract.
        output_dir: Directory for extracted PNGs. Uses temp dir if None.
        total_frames: Total frames in video. Auto-detected if None.
        fps: Video framerate.

    Returns:
        List of output PNG file paths.
    """
    if total_frames is None:
        total_frames = get_frame_count(video_path)
        if total_frames is None:
            total_frames = fps * 15  # fallback: assume 15s

    if n_frames >= total_frames:
        indices = list(range(total_frames))
    elif n_frames == 1:
        indices = [total_frames // 2]
    else:
        step = (total_frames - 1) / (n_frames - 1)
        indices = [round(i * step) for i in range(n_frames)]

    return extract_frames(video_path, indices, output_dir, fps)


def get_frame_count(video_path: str) -> Optional[int]:
    """Get total frame count from a video using ffprobe."""
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
    """Get video metadata (duration, fps, resolution) via ffprobe."""
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
