"""Human baseline v2 server with mode-aware session routing."""
from __future__ import annotations

import argparse
import json
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any
from urllib.parse import parse_qs, urlparse

from session_adapters_v2 import AdaptiveSortSessionAdapter, ProgressiveModeAdapter
from session_v2 import ProgressiveSessionManager

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend_v2"
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
EXAMPLE_ADAPTIVE_SORT_TASKS = THIS_DIR / "examples" / "adaptive_sort_tasks"


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class HumanBaselineV2Handler(SimpleHTTPRequestHandler):
    """Request handler for human-baseline v2."""

    adapters: dict[str, object]
    images_dir: Path
    multi_view_images_dir: Path
    tasks_dir: Path

    def log_message(self, fmt, *args):
        sys.stderr.write(f"[server] {fmt % args}\n")

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/api/v2/capabilities":
            return self._handle_capabilities()
        if path == "/api/v2/scene/current":
            return self._handle_get_current_round(parsed)
        if path == "/api/v2/progress":
            return self._handle_get_progress(parsed)

        if path.startswith("/data-images/"):
            return self._serve_data_image(path)
        if path.startswith("/tasks/images/"):
            return self._serve_task_image(path)

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
    # API
    # ------------------------------------------------------------------

    def _handle_capabilities(self):
        modes = {
            mode_id: adapter.describe()
            for mode_id, adapter in self.adapters.items()
        }
        self._send_json({
            "default_mode": "progressive",
            "modes": modes,
        })

    def _handle_start_session(self, body: dict):
        annotator_id = body.get("annotator_id", "").strip()
        test_mode = self._read_test_mode_from_body(body)
        if not annotator_id:
            return self._send_json({"error": "annotator_id is required"}, 400)

        adapter = self._resolve_adapter(test_mode)
        if adapter is None:
            return self._send_json({"error": f"Unknown test mode: {test_mode}"}, 400)
        if not adapter.is_configured():
            return self._send_json({"error": f"Test mode not configured: {test_mode}"}, 400)

        summary = adapter.get_progress_summary(annotator_id)
        current = adapter.get_current_round(annotator_id)
        self._send_json({
            "test_mode": test_mode,
            "progress": summary,
            "has_current_scene": current is not None,
        })

    def _handle_get_current_round(self, parsed):
        qs = parse_qs(parsed.query)
        annotator_id = (qs.get("annotator_id") or [""])[0].strip()
        test_mode = self._read_test_mode_from_query(parsed)
        if not annotator_id:
            return self._send_json({"error": "annotator_id is required"}, 400)

        adapter = self._resolve_adapter(test_mode)
        if adapter is None:
            return self._send_json({"error": f"Unknown test mode: {test_mode}"}, 400)
        if not adapter.is_configured():
            return self._send_json({"error": f"Test mode not configured: {test_mode}"}, 400)

        data = adapter.get_current_round(annotator_id)
        if data is None:
            return self._send_json({"needs_allocation": True, "test_mode": test_mode})

        self._send_json(data)

    def _handle_allocate_scene(self, body: dict):
        annotator_id = body.get("annotator_id", "").strip()
        test_mode = self._read_test_mode_from_body(body)
        if not annotator_id:
            return self._send_json({"error": "annotator_id is required"}, 400)

        adapter = self._resolve_adapter(test_mode)
        if adapter is None:
            return self._send_json({"error": f"Unknown test mode: {test_mode}"}, 400)
        if not adapter.is_configured():
            return self._send_json({"error": f"Test mode not configured: {test_mode}"}, 400)

        data = adapter.allocate_scene(annotator_id)
        if data is None:
            return self._send_json({"done": True, "test_mode": test_mode})

        self._send_json(data)

    def _handle_submit_round(self, body: dict):
        annotator_id = body.get("annotator_id", "").strip()
        test_mode = self._read_test_mode_from_body(body)
        if not annotator_id:
            return self._send_json({"error": "annotator_id is required"}, 400)

        adapter = self._resolve_adapter(test_mode)
        if adapter is None:
            return self._send_json({"error": f"Unknown test mode: {test_mode}"}, 400)
        if not adapter.is_configured():
            return self._send_json({"error": f"Test mode not configured: {test_mode}"}, 400)

        try:
            result = adapter.submit_round(annotator_id, body)
        except ValueError as exc:
            return self._send_json({"error": str(exc)}, 400)

        self._send_json(result)

    def _handle_get_progress(self, parsed):
        qs = parse_qs(parsed.query)
        annotator_id = (qs.get("annotator_id") or [""])[0].strip()
        test_mode = self._read_test_mode_from_query(parsed)
        if not annotator_id:
            return self._send_json({"error": "annotator_id is required"}, 400)

        adapter = self._resolve_adapter(test_mode)
        if adapter is None:
            return self._send_json({"error": f"Unknown test mode: {test_mode}"}, 400)
        if not adapter.is_configured():
            return self._send_json({"error": f"Test mode not configured: {test_mode}"}, 400)

        summary = adapter.get_progress_summary(annotator_id)
        self._send_json(summary)

    # ------------------------------------------------------------------
    # Images / static files
    # ------------------------------------------------------------------

    def _serve_data_image(self, path: str):
        rel = path[len("/data-images/"):]
        img_root = self.images_dir.parent
        if rel.startswith("images/"):
            rel = rel[len("images/"):]
        self._serve_file(img_root / rel)

    def _serve_task_image(self, path: str):
        rel = path[len("/tasks/"):]
        self._serve_file(self.tasks_dir / rel)

    def _serve_file(self, file_path: Path):
        if not file_path.is_file():
            self._send_json({"error": "File not found"}, 404)
            return

        suffix = file_path.suffix.lower()
        content_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".svg": "image/svg+xml",
            ".json": "application/json",
        }.get(suffix, "application/octet-stream")

        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        self.wfile.write(data)

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

        content_type = {
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
        }.get(file_path.suffix.lower(), "application/octet-stream")

        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # ------------------------------------------------------------------
    # Helpers
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

    def _read_test_mode_from_body(self, body: dict) -> str:
        return str(body.get("test_mode") or "progressive").strip() or "progressive"

    def _read_test_mode_from_query(self, parsed) -> str:
        qs = parse_qs(parsed.query)
        return str((qs.get("test_mode") or ["progressive"])[0]).strip() or "progressive"

    def _resolve_adapter(self, test_mode: str):
        return self.adapters.get(test_mode)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Human baseline v2 server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8124)
    parser.add_argument(
        "--questions-dir",
        default=str(REPO_ROOT / "VLM-test" / "output" / "questions"),
    )
    parser.add_argument(
        "--scenes-dir",
        default=str(REPO_ROOT / "data-gen" / "output" / "scenes"),
    )
    parser.add_argument(
        "--images-dir",
        default=str(REPO_ROOT / "data-gen" / "output" / "images" / "single_view"),
    )
    parser.add_argument(
        "--multi-view-images-dir",
        default=str(REPO_ROOT / "data-gen" / "output" / "images" / "multi_view"),
    )
    parser.add_argument(
        "--tasks-dir",
        default=str(THIS_DIR / "output" / "tasks"),
    )
    parser.add_argument(
        "--responses-dir",
        default=str(THIS_DIR / "output" / "responses"),
    )
    parser.add_argument("--test-scenes-file", default=None)
    parser.add_argument(
        "--adaptive-sort-tasks-dir",
        default=str(EXAMPLE_ADAPTIVE_SORT_TASKS),
        help="Directory containing manifest.json and per-scene adaptive-sort tasks.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()

    progressive_mgr = ProgressiveSessionManager(
        questions_dir=args.questions_dir,
        scenes_dir=args.scenes_dir,
        images_dir=args.images_dir,
        multi_view_images_dir=args.multi_view_images_dir,
        tasks_dir=args.tasks_dir,
        responses_dir=args.responses_dir,
        test_scenes_file=args.test_scenes_file,
    )
    adapters = {
        "progressive": ProgressiveModeAdapter(progressive_mgr),
        "adaptive_sort": AdaptiveSortSessionAdapter(
            tasks_dir=args.adaptive_sort_tasks_dir,
            responses_dir=args.responses_dir,
        ),
    }

    HumanBaselineV2Handler.adapters = adapters
    HumanBaselineV2Handler.images_dir = Path(args.images_dir)
    HumanBaselineV2Handler.multi_view_images_dir = Path(args.multi_view_images_dir)
    HumanBaselineV2Handler.tasks_dir = Path(args.tasks_dir)

    server = ThreadingHTTPServer((args.host, args.port), HumanBaselineV2Handler)
    print("Human Baseline Server v2")
    print(f"  http://{args.host}:{args.port}")
    print(f"  Questions: {args.questions_dir}")
    print(f"  Scenes:    {args.scenes_dir}")
    print(f"  Images:    {args.images_dir}")
    print(f"  Responses: {args.responses_dir}")
    print(f"  Adaptive Sort Tasks: {args.adaptive_sort_tasks_dir}")
    print("  Modes:")
    for mode_id, adapter in adapters.items():
        desc = adapter.describe()
        configured = "configured" if desc.get("configured") else "unconfigured"
        print(f"    - {mode_id}: {configured}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
