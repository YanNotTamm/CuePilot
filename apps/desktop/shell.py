"""CuePilot desktop shell entry point (pywebview).

Thin alias for the canonical entry point (fix R-5): previously this file
duplicated `apps.desktop.entry.main` almost line-for-line (window setup,
sidecar start, readiness poll, folder-picker API), and the two copies
could drift. All behaviour now lives in `apps.desktop.entry.main`;
this module only forwards to it — lazily, so importing this module does
not itself require `webview` to be installed, exactly as before.

Canonical: `apps.desktop.entry` (it is also the PyInstaller spec entry
point and owns the frozen-path handling). `python -m apps.desktop`
still works via `apps.desktop.__main__`, which imports `main` from here.
"""

from __future__ import annotations


def main() -> int:
    """Start the desktop shell via the canonical entry point."""
    from apps.desktop.entry import main as _entry_main

    return _entry_main()


if __name__ == "__main__":
    raise SystemExit(main())
