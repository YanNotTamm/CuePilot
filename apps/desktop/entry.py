"""PyInstaller entry point for CuePilot desktop (built under apps/desktop).

Uses static imports so PyInstaller's module analysis can trace dependencies.
"""

import os
import secrets
import sys
import time
from pathlib import Path

if getattr(sys, "frozen", False):
    _REPO_ROOT = Path(sys._MEIPASS)
else:
    _REPO_ROOT = Path(__file__).resolve().parents[2]

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import webview  # noqa: E402

from apps.desktop.server import find_free_port, run_server  # noqa: E402


def _center(width: int, height: int) -> dict[str, int]:
    """Compute top-left coords so the window is centered on the primary screen."""
    try:
        screens = webview.screens
        if screens:
            s = screens[0]
            sw, sh = s.width, s.height
            return {"x": max(0, (sw - width) // 2), "y": max(0, (sh - height) // 2)}
    except Exception:
        pass
    return {"x": (1920 - width) // 2, "y": (1080 - height) // 2}


def main() -> int:
    token = secrets.token_hex(16)
    port = find_free_port()
    os.environ["CUEPILOT_SESSION_TOKEN"] = token
    run_server(port, token)

    import socket

    ready = False
    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                ready = True
                break
        except OSError:
            time.sleep(0.1)
    if not ready:
        return 1

    class Api:
        def pick_folder(self):
            result = webview.windows[0].create_file_dialog(webview.FOLDER_DIALOG)
            if isinstance(result, (tuple, list)) and result:
                return result[0]
            if isinstance(result, str) and result:
                return result
            return None

    webview.create_window(
        "CuePilot", f"http://127.0.0.1:{port}", width=1280, height=800, js_api=Api(), **_center(1280, 800)
    )
    webview.start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
