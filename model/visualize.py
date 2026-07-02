"""Flask web viewer for recorded EEG sessions."""

from __future__ import annotations

import argparse
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from _viewer_core import SessionStore, discover_sessions, parse_positive_ms

APP_DIR = Path(__file__).resolve().parent


def create_app(recordings_path: Path) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(APP_DIR / "templates"),
        static_folder=str(APP_DIR / "static"),
    )
    store = SessionStore(discover_sessions(recordings_path))

    @app.get("/")
    def index() -> str:
        return render_template("viewer.html")

    @app.get("/api/sessions")
    def api_sessions():
        return jsonify({"sessions": store.list_sessions()})

    @app.get("/api/sessions/<int:session_index>/timeline-range")
    def api_timeline_range(session_index: int):
        try:
            window_ms = parse_positive_ms(request.args.get("window_ms", "200"), "Window")
            pre_ms = parse_positive_ms(request.args.get("pre_ms", "100"), "Pre")
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(store.timeline_range(session_index, window_ms=window_ms, pre_ms=pre_ms))

    @app.get("/api/sessions/<int:session_index>/events")
    def api_events(session_index: int):
        return jsonify(store.list_events(session_index))

    @app.get("/api/sessions/<int:session_index>/timeline")
    def api_timeline(session_index: int):
        try:
            window_ms = parse_positive_ms(request.args.get("window_ms", "200"), "Window")
            pre_ms = parse_positive_ms(request.args.get("pre_ms", "100"), "Pre")
            post_ms = parse_positive_ms(request.args.get("post_ms", "100"), "Post")
            target_row = int(request.args.get("target_row", "0"))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(
            store.timeline_window(
                session_index,
                target_row=target_row,
                window_ms=window_ms,
                pre_ms=pre_ms,
                post_ms=post_ms,
            )
        )

    @app.get("/api/sessions/<int:session_index>/events/<int:event_index>")
    def api_event_window(session_index: int, event_index: int):
        try:
            pre_ms = parse_positive_ms(request.args.get("pre_ms", "100"), "Pre")
            post_ms = parse_positive_ms(request.args.get("post_ms", "100"), "Post")
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        try:
            payload = store.event_window(
                session_index,
                event_index,
                pre_ms=pre_ms,
                post_ms=post_ms,
            )
        except IndexError as exc:
            return jsonify({"error": str(exc)}), 404
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(payload)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browse recorded EEG sessions (web UI).")
    parser.add_argument(
        "--path",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "recordings",
        help="Session directory or parent folder containing session_* dirs",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5080)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(args.path)
    print(f"Open http://{args.host}:{args.port}/")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
