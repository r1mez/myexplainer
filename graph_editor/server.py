from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from graph_editor.runtime import GraphEditorError, GraphEditorRuntime


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"


class GraphEditorRequestHandler(BaseHTTPRequestHandler):
    runtime: GraphEditorRuntime = None  # type: ignore[assignment]

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/datasets":
            self._handle_list_datasets()
            return
        if parsed.path == "/api/graph":
            self._handle_get_graph(parsed.query)
            return
        if parsed.path.startswith("/api/"):
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"error": f"Unknown endpoint '{parsed.path}'."},
            )
            return
        self._serve_static(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/predict":
            self._handle_predict()
            return
        self._send_json(
            HTTPStatus.NOT_FOUND,
            {"error": f"Unknown endpoint '{parsed.path}'."},
        )

    def log_message(self, format: str, *args) -> None:
        super().log_message(format, *args)

    def _handle_list_datasets(self) -> None:
        try:
            payload = self.runtime.list_datasets()
            self._send_json(HTTPStatus.OK, payload)
        except GraphEditorError as exc:
            self._send_api_error(exc)
        except Exception as exc:
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"Unexpected server error: {exc}"},
            )

    def _handle_get_graph(self, query_string: str) -> None:
        query = parse_qs(query_string)
        dataset = query.get("dataset", [None])[0]
        split = query.get("split", [None])[0]
        index = query.get("index", [None])[0]

        try:
            payload = self.runtime.get_graph(dataset, split, int(index))
            self._send_json(HTTPStatus.OK, payload)
        except (TypeError, ValueError):
            self._send_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "Query parameter 'index' must be an integer."},
            )
        except GraphEditorError as exc:
            self._send_api_error(exc)
        except Exception as exc:
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"Unexpected server error: {exc}"},
            )

    def _handle_predict(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0

        raw_body = self.rfile.read(content_length) if content_length > 0 else b""
        try:
            payload = json.loads(raw_body.decode("utf-8") or "{}")
        except json.JSONDecodeError as exc:
            self._send_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": f"Invalid JSON payload: {exc}"},
            )
            return

        try:
            result = self.runtime.predict_from_payload(payload)
            self._send_json(HTTPStatus.OK, result)
        except GraphEditorError as exc:
            self._send_api_error(exc)
        except Exception as exc:
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"Unexpected server error: {exc}"},
            )

    def _serve_static(self, raw_path: str) -> None:
        relative_path = raw_path.lstrip("/") or "index.html"
        if relative_path == "":
            relative_path = "index.html"

        target = (STATIC_DIR / relative_path).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())):
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Static path is outside the web root."})
            return

        if not target.exists() or not target.is_file():
            if raw_path in {"/", "", "/index.html"}:
                target = STATIC_DIR / "index.html"
            else:
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"error": f"Static asset '{raw_path}' was not found."},
                )
                return

        if not target.exists() or not target.is_file():
            target = STATIC_DIR / "index.html"

        content_type, _ = mimetypes.guess_type(str(target))
        if target.suffix == ".js":
            content_type = "application/javascript; charset=utf-8"
        elif target.suffix == ".css":
            content_type = "text/css; charset=utf-8"
        elif target.suffix == ".html":
            content_type = "text/html; charset=utf-8"

        with target.open("rb") as handle:
            body = handle.read()

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_api_error(self, exc: GraphEditorError) -> None:
        payload = {"error": exc.message}
        if exc.details:
            payload["details"] = exc.details
        self._send_json(exc.status_code, payload)

    def _send_json(self, status_code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the lightweight graph editor web UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind.")
    parser.add_argument("--device", default="cuda:0", help="Torch device for prediction.")
    parser.add_argument(
        "--data-root",
        default=str(PROJECT_ROOT / "data"),
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--param-root",
        default=str(PROJECT_ROOT / "param"),
        help="Parameter/checkpoint root directory.",
    )
    parser.add_argument(
        "--default-dataset",
        default="mutag",
        help="Dataset selected by default in the UI.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = GraphEditorRuntime(
        project_root=PROJECT_ROOT,
        data_root=Path(args.data_root),
        param_root=Path(args.param_root),
        device=args.device,
        default_dataset=args.default_dataset,
    )

    handler_class = GraphEditorRequestHandler
    handler_class.runtime = runtime

    server = ThreadingHTTPServer((args.host, args.port), handler_class)
    print(
        "Graph editor is serving on http://{host}:{port} using device {device}".format(
            host=args.host,
            port=args.port,
            device=runtime.device,
        )
    )
    print(f"Static assets: {STATIC_DIR}")
    print(f"Data root: {runtime.data_root}")
    print(f"Param root: {runtime.param_root}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down graph editor.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
