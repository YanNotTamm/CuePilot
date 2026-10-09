# -*- mode: python ; coding: utf-8 -*-

import os

_dist = os.path.normpath(os.path.join(os.path.abspath(SPECPATH), "web", "dist"))
_repo_root = os.path.abspath(os.path.join(os.path.abspath(SPECPATH), "..", ".."))

a = Analysis(
    ["entry.py"],
    pathex=[_repo_root],
    binaries=[],
    datas=[
        (_dist, "apps/desktop/web/dist"),
    ],
    hiddenimports=[
        "apps.desktop.server",
        "core.stems",
        "core.learn",
        "core.edits",
        "core.metadata",
        "core.settings",
        "uvicorn.logging",
        "uvicorn.loops.auto",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.http.h11_impl",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan.on",
        "webview.platforms.edgechromium",
        "librosa",
        "resampy",
        "soundfile",
        "mutagen",
        "jsonschema",
        "numpy",
        "scipy",
    ],
    excludes=["tkinter"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="CuePilot",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
