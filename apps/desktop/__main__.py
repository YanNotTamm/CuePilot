"""CuePilot desktop launcher (dev).

Jalankan `python -m apps.desktop` untuk membuka aplikasi desktop native.
"""

from apps.desktop.shell import main

if __name__ == "__main__":
    raise SystemExit(main())
