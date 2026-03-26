"""渐进式人类基准测试的 HTTP 服务器（v2）。

用法：
    python server_v2.py [--host HOST] [--port PORT] [--questions-dir DIR] ...

托管 frontend_v2/ 单页应用，并提供支持分轮评分反馈的逐场景渐进测试 API 端点。
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any
from urllib.parse import parse_qs, urlparse

from session_v2 import ProgressiveSessionManager

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend_v2"
REPO_ROOT = Path(__file__).resolve().parent.parent


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class ProgressiveHandler(SimpleHTTPRequestHandler):
    """渐进式人类基准服务器的请求处理器。"""

    session_mgr: ProgressiveSessionManager
    images_dir: Path
    multi_view_images_dir: Path
    tasks_dir: Path

    def log_message(self, fmt, *args):
        sys.stderr.write(f"[server] {fmt % args}\n")

    # ------------------------------------------------------------------
    # 路由
    # ------------------------------------------------------------------

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        # API 端点。
        if path == "/api/v2/scene/current":
            return self._handle_get_current_round(parsed)
        if path == "/api/v2/progress":
            return self._handle_get_progress(parsed)

        # 图像服务。
        if path.startswith("/data-images/"):
            return self._serve_data_image(path)
        if path.startswith("/tasks/images/"):
            return self._serve_task_image(path)

        # 前端静态文件。
        self._serve_frontend(path)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        body = self._read_body()

        if path == "/api/v2/session/start":
            return self._handle_start_session(body)
        if path == "/api/v2/scene/allocate":
            return self._handle_allocate_scene(body)
        if path == "/api/v2/round/submit":
            return self._handle_submit_round(body)

        self._send_json({"error": "Not found"}, 404)

    # ------------------------------------------------------------------
    # API 处理器
    # ------------------------------------------------------------------

    def _handle_start_session(self, body: dict):
        annotator_id = body.get("annotator_id", "").strip()
        if not annotator_id:
            return self._send_json({"error": "annotator_id is required"}, 400)

        summary = self.session_mgr.get_progress_summary(annotator_id)

        # 检查是否有正在进行的场景。
        current = self.session_mgr.get_current_round(annotator_id)

        self._send_json({
            "progress": summary,
            "has_current_scene": current is not None,
        })

    def _handle_get_current_round(self, parsed):
        qs = parse_qs(parsed.query)
        annotator_id = (qs.get("annotator_id") or [""])[0].strip()
        if not annotator_id:
            return self._send_json({"error": "annotator_id is required"}, 400)

        data = self.session_mgr.get_current_round(annotator_id)
        if data is None:
            # 无正在进行的场景，前端应请求分配新场景。
            return self._send_json({"needs_allocation": True})

        self._send_json(data)

    def _handle_allocate_scene(self, body: dict):
        annotator_id = body.get("annotator_id", "").strip()
        if not annotator_id:
            return self._send_json({"error": "annotator_id is required"}, 400)

        data = self.session_mgr.allocate_scene(annotator_id)
        if data is None:
            return self._send_json({"done": True})

        self._send_json(data)

    def _handle_submit_round(self, body: dict):
        annotator_id = body.get("annotator_id", "").strip()
        if not annotator_id:
            return self._send_json({"error": "annotator_id is required"}, 400)

        try:
            result = self.session_mgr.submit_round(annotator_id, body)
            self._send_json(result)
        except ValueError as e:
            self._send_json({"error": str(e)}, 400)

    def _handle_get_progress(self, parsed):
        qs = parse_qs(parsed.query)
        annotator_id = (qs.get("annotator_id") or [""])[0].strip()
        if not annotator_id:
            return self._send_json({"error": "annotator_id is required"}, 400)

        summary = self.session_mgr.get_progress_summary(annotator_id)
        self._send_json(summary)

    # ------------------------------------------------------------------
    # 图像服务
    # ------------------------------------------------------------------

    def _serve_data_image(self, path: str):
        # /data-images/images/single_view/xxx.png -> images_dir/../single_view/xxx.png
        rel = path[len("/data-images/"):]
        # 图像存储在 data-gen 输出根目录下。
        # 路径形如：images/single_view/xxx.png 或 images/multi_view/xxx/view_0.png
        img_root = self.images_dir.parent  # single_view 目录的父目录
        if rel.startswith("images/"):
            rel = rel[len("images/"):]
        file_path = img_root / rel
        self._serve_file(file_path)

    def _serve_task_image(self, path: str):
        # /tasks/images/xxx/yyy.png -> tasks_dir/images/xxx/yyy.png（任务图像）
        rel = path[len("/tasks/"):]
        file_path = self.tasks_dir / rel
        self._serve_file(file_path)

    def _serve_file(self, file_path: Path):
        if not file_path.is_file():
            self._send_json({"error": "File not found"}, 404)
            return

        content_type = "application/octet-stream"
        suffix = file_path.suffix.lower()
        type_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".svg": "image/svg+xml",
            ".json": "application/json",
        }
        content_type = type_map.get(suffix, content_type)

        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        self.wfile.write(data)

    # ------------------------------------------------------------------
    # 前端静态文件服务
    # ------------------------------------------------------------------

    def _serve_frontend(self, path: str):
        file_map = {
            "": "index.html",
            "/index.html": "index.html",
            "/app.js": "app.js",
            "/styles.css": "styles.css",
        }

        filename = file_map.get(path)
        if filename is None:
            self._send_json({"error": "Not found"}, 404)
            return

        file_path = FRONTEND_DIR / filename
        if not file_path.is_file():
            self._send_json({"error": f"Frontend file not found: {filename}"}, 404)
            return

        content_type_map = {
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
        }
        suffix = file_path.suffix.lower()
        content_type = content_type_map.get(suffix, "application/octet-stream")

        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # ------------------------------------------------------------------
    # 辅助函数
    # ------------------------------------------------------------------

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Progressive human baseline server (v2)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8124)
    p.add_argument(
        "--questions-dir",
        default=str(REPO_ROOT / "VLM-test" / "output" / "questions"),
    )
    p.add_argument(
        "--scenes-dir",
        default=str(REPO_ROOT / "data-gen" / "output" / "scenes"),
    )
    p.add_argument(
        "--images-dir",
        default=str(REPO_ROOT / "data-gen" / "output" / "images" / "single_view"),
    )
    p.add_argument(
        "--multi-view-images-dir",
        default=str(REPO_ROOT / "data-gen" / "output" / "images" / "multi_view"),
    )
    p.add_argument(
        "--tasks-dir",
        default=str(Path(__file__).resolve().parent / "output" / "tasks"),
    )
    p.add_argument(
        "--responses-dir",
        default=str(Path(__file__).resolve().parent / "output" / "responses"),
    )
    p.add_argument("--test-scenes-file", default=None)
    return p


def main():
    args = build_arg_parser().parse_args()

    mgr = ProgressiveSessionManager(
        questions_dir=args.questions_dir,
        scenes_dir=args.scenes_dir,
        images_dir=args.images_dir,
        multi_view_images_dir=args.multi_view_images_dir,
        tasks_dir=args.tasks_dir,
        responses_dir=args.responses_dir,
        test_scenes_file=args.test_scenes_file,
    )

    ProgressiveHandler.session_mgr = mgr
    ProgressiveHandler.images_dir = Path(args.images_dir)
    ProgressiveHandler.multi_view_images_dir = Path(args.multi_view_images_dir)
    ProgressiveHandler.tasks_dir = Path(args.tasks_dir)

    server = ThreadingHTTPServer((args.host, args.port), ProgressiveHandler)
    print(f"Progressive Human Baseline Server v2")
    print(f"  http://{args.host}:{args.port}")
    print(f"  Questions: {args.questions_dir}")
    print(f"  Scenes:    {args.scenes_dir}")
    print(f"  Images:    {args.images_dir}")
    print(f"  Responses: {args.responses_dir}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
