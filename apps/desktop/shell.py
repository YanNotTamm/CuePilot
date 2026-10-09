"""CuePilot desktop shell entry point (pywebview).

Spawns the FastAPI sidecar on 127.0.0.1:<random_port>, opens a native WebView2
window (installed on Windows), and cleans up the server when the window closes.
"""

from __future__ import annotations

import os
import secrets
import sys
import threading
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from apps.desktop.server import find_free_port, run_server  # noqa: E402


def _center(webview_module, width: int, height: int) -> dict[str, int]:
    """Compute top-left coords so the window is centered on the primary screen."""
    try:
        screens = webview_module.screens
        if screens:
            s = screens[0]
            sw, sh = s.width, s.height
            return {"x": max(0, (sw - width) // 2), "y": max(0, (sh - height) // 2)}
    except Exception:
        pass
    # Fallback: common 1920x1080
    return {"x": (1920 - width) // 2, "y": (1080 - height) // 2}


def main() -> int:
    import webview

    token = secrets.token_hex(16)
    port = find_free_port()
    os.environ["CUEPILOT_SESSION_TOKEN"] = token

    run_server(port, token)

    # Small poll so the server is up before the window loads the URL.
    ready = False
    import socket

    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                ready = True
                break
        except OSError:
            time.sleep(0.1)
    if not ready:
        print("sidecar did not start; aborting", file=sys.stderr)
        return 1

    url = f"http://127.0.0.1:{port}"

    class Api:
        def pick_folder(self):
            import webview as _wv

            result = _wv.windows[0].create_file_dialog(_wv.FOLDER_DIALOG)
            if isinstance(result, (tuple, list)) and result:
                return result[0]
            if isinstance(result, str) and result:
                return result
            return None

    window = webview.create_window(
        "CuePilot", url, width=1280, height=800, js_api=Api(), **_center(webview, 1280, 800)
    )
    webview.start()

    # uvicorn thread is daemon; process exit terminates it. No zombie process.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
