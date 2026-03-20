#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from common import REPO_ROOT, slugify
from session import SessionManager


THIS_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = THIS_DIR / "frontend"


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


class HumanBaselineHandler(BaseHTTPRequestHandler):
    tasks_dir: Path = Path(".")
    responses_dir: Path = Path(".")
    images_dir: Path = Path(".")
    multi_view_images_dir: Path = Path(".")
    session_manager: SessionManager | None = None

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return super().log_message(format, *args)

    def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self._send_text("Not found", status=HTTPStatus.NOT_FOUND)
            return

        mime_type, _ = mimetypes.guess_type(str(path))
        mime_type = mime_type or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _manifest_path(self) -> Path:
        return self.tasks_dir / "manifest.json"

    def _task_json_path(self, scene_id: str) -> Path:
        return self.tasks_dir / "json" / f"{scene_id}.json"

    def _response_listing(self) -> list[dict]:
        items = []
        for path in sorted(self.responses_dir.rglob("*.json")):
            try:
                payload = load_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            if "scene_id" not in payload:
                continue
            items.append({
                "scene_id": payload.get("scene_id"),
                "annotator_id": payload.get("annotator_id"),
                "submitted_at": payload.get("submitted_at"),
                "path": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
            })
        return items

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == "/api/manifest":
            manifest_path = self._manifest_path()
            if not manifest_path.exists():
                self._send_json({"error": "manifest not found"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(load_json(manifest_path))
            return

        if path.startswith("/api/tasks/"):
            scene_id = path.rsplit("/", 1)[-1]
            task_path = self._task_json_path(scene_id)
            if not task_path.exists():
                self._send_json({"error": f"task not found for {scene_id}"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(load_json(task_path))
            return

        if path == "/api/responses":
            self._send_json({"items": self._response_listing()})
            return

        # ---- Session management endpoints ----

        if path == "/api/session/next-page":
            if self.session_manager is None:
                self._send_json(
                    {"error": "session management not configured"},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return
            qs = parse_qs(urlparse(self.path).query)
            annotator_id = qs.get("annotator_id", [None])[0]
            test_type = qs.get("test_type", ["single_view"])[0]
            if not annotator_id:
                self._send_json(
                    {"error": "annotator_id query parameter is required"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            page = self.session_manager.allocate_page(annotator_id, test_type)
            if page is None:
                self._send_json({"done": True})
            else:
                self._send_json(page)
            return

        if path == "/api/session/progress":
            if self.session_manager is None:
                self._send_json(
                    {"error": "session management not configured"},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return
            qs = parse_qs(urlparse(self.path).query)
            annotator_id = qs.get("annotator_id", [None])[0]
            if not annotator_id:
                self._send_json(
                    {"error": "annotator_id query parameter is required"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            summary = self.session_manager.get_progress_summary(annotator_id)
            self._send_json(summary)
            return

        # ---- Data image serving ----

        if path.startswith("/data-images/"):
            relative = path.removeprefix("/data-images/")
            # Reject path traversal attempts.
            if ".." in relative:
                self._send_text("Forbidden", status=HTTPStatus.FORBIDDEN)
                return
            # Try single_view first, then multi_view.
            candidate = self.images_dir / relative
            if not candidate.exists():
                candidate = self.multi_view_images_dir / relative
            if not candidate.exists():
                # Fall back to a general images root (parent of single_view dir).
                candidate = self.images_dir.parent / relative
            self._send_file(candidate)
            return

        if path.startswith("/tasks/images/"):
            relative = path.removeprefix("/tasks/")
            self._send_file(self.tasks_dir / relative)
            return

        if path == "/" or path == "/index.html":
            self._send_file(FRONTEND_DIR / "index.html")
            return

        if path in {"/app.js", "/styles.css"}:
            self._send_file(FRONTEND_DIR / path.lstrip("/"))
            return

        self._send_text("Not found", status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        # ---- Session management POST endpoints ----

        if path == "/api/session/start":
            if self.session_manager is None:
                self._send_json(
                    {"error": "session management not configured"},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return
            try:
                body = self._read_json_body()
            except json.JSONDecodeError as exc:
                self._send_json(
                    {"error": f"invalid json: {exc}"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            annotator_id = body.get("annotator_id")
            if not annotator_id:
                self._send_json(
                    {"error": "annotator_id is required"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            summary = self.session_manager.get_progress_summary(annotator_id)
            self._send_json(summary)
            return

        if path == "/api/session/submit-page":
            if self.session_manager is None:
                self._send_json(
                    {"error": "session management not configured"},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return
            try:
                body = self._read_json_body()
            except json.JSONDecodeError as exc:
                self._send_json(
                    {"error": f"invalid json: {exc}"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            annotator_id = body.get("annotator_id")
            if not annotator_id:
                self._send_json(
                    {"error": "annotator_id is required"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            if "scene_id" not in body:
                self._send_json(
                    {"error": "scene_id is required"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            try:
                progress = self.session_manager.submit_page(annotator_id, body)
            except (KeyError, ValueError) as exc:
                self._send_json(
                    {"error": f"submission error: {exc}"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            self._send_json({"ok": True, "progress": progress})
            return

        # ---- Existing /api/responses endpoint ----

        if path != "/api/responses":
            self._send_json({"error": "unsupported endpoint"}, status=HTTPStatus.NOT_FOUND)
            return

        try:
            payload = self._read_json_body()
        except json.JSONDecodeError as exc:
            self._send_json({"error": f"invalid json: {exc}"}, status=HTTPStatus.BAD_REQUEST)
            return

        scene_id = payload.get("scene_id")
        if not scene_id:
            self._send_json({"error": "scene_id is required"}, status=HTTPStatus.BAD_REQUEST)
            return

        annotator_id = slugify(str(payload.get("annotator_id", "anonymous")))
        payload["annotator_id"] = annotator_id
        payload["model"] = f"human/{annotator_id}"

        output_dir = self.responses_dir / annotator_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{scene_id}__{annotator_id}.json"
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        response_path = (
            str(output_path.relative_to(REPO_ROOT))
            if output_path.is_relative_to(REPO_ROOT)
            else str(output_path)
        )
        self._send_json({
            "ok": True,
            "saved_to": response_path,
            "scene_id": scene_id,
            "annotator_id": annotator_id,
        })


def main() -> None:
    parser = argparse.ArgumentParser(description="Human baseline local server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--tasks-dir", default="human-baseline/output/tasks")
    parser.add_argument("--responses-dir", default="human-baseline/output/responses")
    parser.add_argument(
        "--questions-dir",
        default="../VLM-test/output/questions",
        help="Directory containing per-type question JSONs (default: ../VLM-test/output/questions)",
    )
    parser.add_argument(
        "--scenes-dir",
        default="../data-gen/output/scenes",
        help="Directory containing scene JSON files (default: ../data-gen/output/scenes)",
    )
    parser.add_argument(
        "--images-dir",
        default="../data-gen/output/images/single_view",
        help="Directory containing single-view images (default: ../data-gen/output/images/single_view)",
    )
    parser.add_argument(
        "--multi-view-images-dir",
        default="../data-gen/output/images/multi_view",
        help="Directory containing multi-view images (default: ../data-gen/output/images/multi_view)",
    )
    parser.add_argument(
        "--test-scenes-file",
        default=None,
        help="Path to test_scenes.json (default: auto-detect near scenes-dir)",
    )
    args = parser.parse_args()

    handler = HumanBaselineHandler
    handler.tasks_dir = Path(args.tasks_dir).resolve()
    handler.responses_dir = Path(args.responses_dir).resolve()
    handler.images_dir = Path(args.images_dir).resolve()
    handler.multi_view_images_dir = Path(args.multi_view_images_dir).resolve()

    # Initialize SessionManager for session-based question allocation.
    handler.session_manager = SessionManager(
        questions_dir=Path(args.questions_dir).resolve(),
        scenes_dir=Path(args.scenes_dir).resolve(),
        images_dir=handler.images_dir,
        multi_view_images_dir=handler.multi_view_images_dir,
        tasks_dir=handler.tasks_dir,
        responses_dir=handler.responses_dir,
        test_scenes_file=(
            Path(args.test_scenes_file).resolve()
            if args.test_scenes_file
            else None
        ),
    )

    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Human baseline server running at http://{args.host}:{args.port}")
    print(f"Tasks dir:       {handler.tasks_dir}")
    print(f"Responses dir:   {handler.responses_dir}")
    print(f"Questions dir:   {handler.session_manager.questions_dir}")
    print(f"Scenes dir:      {handler.session_manager.scenes_dir}")
    print(f"Images dir:      {handler.images_dir}")
    print(f"Multi-view dir:  {handler.multi_view_images_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
