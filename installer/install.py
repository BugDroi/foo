#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-platform installer for Buzzwole / BLACEPHALON stack.
Works on Linux, macOS, and Windows.

Usage:
  python install.py
  python install.py --cpu          # force CPU-only PyTorch
  python install.py --gpu          # force CUDA PyTorch
  python install.py --req FILE     # custom requirements file
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_REQ = ROOT / ".requirements.txt"
FALLBACK_REQ = ROOT / "requirements.txt"
VENV_DIR = ROOT / "venv"

BANNER = """
=============================
 LIB INSTALLER
 (cross-platform)
=============================
"""


def run(cmd: list[str], *, env: dict | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, env=env)


def is_admin() -> bool:
    """True if running with elevated privileges (sudo / Administrator)."""
    if platform.system() == "Windows":
        try:
            import ctypes
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            return False
    return hasattr(os, "geteuid") and os.geteuid() == 0


def find_python() -> str:
    """Prefer python3, then python (Windows often only has 'python')."""
    for name in ("python3", "python"):
        path = shutil.which(name)
        if path:
            # Skip Windows Store stub that just opens the Store
            try:
                out = subprocess.check_output([path, "-c", "import sys; print(sys.version)"], text=True, stderr=subprocess.DEVNULL)
                if out.strip():
                    return path
            except Exception:
                continue
    print("Error: Python 3 not found. Install Python 3.10+ and retry.")
    sys.exit(1)


def venv_python() -> Path:
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def venv_pip() -> list[str]:
    return [str(venv_python()), "-m", "pip"]


def detect_nvidia() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        r = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:
        return False


def create_venv(base_python: str) -> None:
    if VENV_DIR.exists():
        print(f"[INFO] Existing venv found: {VENV_DIR}")
        print("       Reusing it. Delete the folder to recreate from scratch.")
        return
    print("Creating virtual environment (venv)...")
    run([base_python, "-m", "venv", str(VENV_DIR)])


def upgrade_pip() -> None:
    print("Upgrading pip / setuptools / wheel...")
    run(venv_pip() + ["install", "--upgrade", "pip", "setuptools", "wheel"])


def install_torch(force_cpu: bool, force_gpu: bool) -> None:
    print()
    print("--- Detecting GPU ---")
    use_gpu = False
    if force_cpu:
        print("[FORCE] CPU-only PyTorch requested.")
    elif force_gpu:
        print("[FORCE] CUDA PyTorch requested.")
        use_gpu = True
    elif detect_nvidia():
        print("[OK] NVIDIA GPU detected.")
        use_gpu = True
    else:
        print("[SKIP] No NVIDIA GPU detected.")

    pkgs = ["torch", "torchvision", "torchaudio"]
    if use_gpu:
        print("Installing GPU-enabled PyTorch... (may take several minutes)")
        # Default PyPI index serves CUDA builds on Linux/Windows when available
        run(venv_pip() + ["install"] + pkgs)
    else:
        print("Installing CPU-only PyTorch...")
        run(
            venv_pip()
            + [
                "install",
                *pkgs,
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ]
        )


def install_requirements(req_file: Path) -> None:
    print()
    print("--- Installing remaining libraries ---")
    if not req_file.is_file():
        print(f"Error: requirements file not found: {req_file}")
        sys.exit(1)
    # Filter out torch lines if someone put them in the req file
    lines = []
    for raw in req_file.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            lines.append(raw)
            continue
        name = s.split("==")[0].split(">=")[0].split("<")[0].split("[")[0].strip().lower()
        if name in {"torch", "torchvision", "torchaudio"}:
            print(f"  [skip] {s}  (handled by GPU/CPU step)")
            continue
        lines.append(raw)

    filtered = ROOT / ".requirements.filtered.tmp"
    filtered.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        run(venv_pip() + ["install", "-r", str(filtered)])
    finally:
        filtered.unlink(missing_ok=True)


def print_activate_hint() -> None:
    print()
    print("=============================")
    print(" Installation complete!")
    print("=============================")
    if platform.system() == "Windows":
        print("Activate the virtual environment with:")
        print(r"  venv\Scripts\activate")
        print("Then run:")
        print("  python app.py")
    else:
        print("Activate the virtual environment with:")
        print("  source venv/bin/activate")
        print("Then run:")
        print("  python app.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-platform venv installer")
    parser.add_argument("--cpu", action="store_true", help="Force CPU-only PyTorch")
    parser.add_argument("--gpu", action="store_true", help="Force CUDA PyTorch")
    parser.add_argument(
        "--req",
        type=Path,
        default=None,
        help="Path to requirements file (default: .requirements.txt)",
    )
    parser.add_argument(
        "--allow-root",
        action="store_true",
        help="Allow running as root/Administrator (not recommended)",
    )
    args = parser.parse_args()

    if args.cpu and args.gpu:
        print("Error: --cpu and --gpu are mutually exclusive.")
        sys.exit(1)

    print(BANNER)
    print(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"CWD: {ROOT}")

    if is_admin() and not args.allow_root:
        print("Error: Do not run this script with sudo / as Administrator.")
        print("Example:  python install.py")
        print("(If you really need it: python install.py --allow-root)")
        sys.exit(1)

    req = args.req
    if req is None:
        if DEFAULT_REQ.is_file():
            req = DEFAULT_REQ
        elif FALLBACK_REQ.is_file():
            req = FALLBACK_REQ
        else:
            print("Error: neither .requirements.txt nor requirements.txt found.")
            sys.exit(1)
    req = req.resolve()

    base_py = find_python()
    print(f"Base Python: {base_py}")
    ver = subprocess.check_output([base_py, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"], text=True).strip()
    print(f"Python version: {ver}")

    create_venv(base_py)
    py = venv_python()
    if not py.is_file():
        print(f"Error: venv Python not found at {py}")
        sys.exit(1)

    upgrade_pip()
    install_torch(force_cpu=args.cpu, force_gpu=args.gpu)
    install_requirements(req)
    print_activate_hint()


if __name__ == "__main__":
    main()
