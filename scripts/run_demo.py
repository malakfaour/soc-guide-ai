"""Start the SOC Intelligence demo stack with one command."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_ROOT = PROJECT_ROOT / "soc-frontend"
BACKEND_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:5173"
BACKEND_HEALTH_URL = f"{BACKEND_URL}/health"


def command_exists(command: str) -> bool:
    from shutil import which

    return which(command) is not None


def start_process(command: list[str], cwd: Path) -> subprocess.Popen:
    return subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )


def url_is_ready(url: str, timeout_seconds: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            return 200 <= response.status < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def wait_for_backend(process: subprocess.Popen | None, timeout_seconds: int = 90) -> bool:
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        if url_is_ready(BACKEND_HEALTH_URL):
            print(f"[backend] Ready: {BACKEND_HEALTH_URL}")
            return True

        if process is not None and process.poll() is not None:
            output = process.stdout.read() if process.stdout else ""
            if output:
                print(output)
            print("[backend] FastAPI exited before becoming healthy.")
            return False

        time.sleep(1)

    print(f"[backend] Timed out waiting for {BACKEND_HEALTH_URL}")
    return False


def wait_for_frontend(process: subprocess.Popen, timeout_seconds: int = 20) -> str:
    deadline = time.time() + timeout_seconds
    output_lines: list[str] = []

    while time.time() < deadline and process.poll() is None:
        line = process.stdout.readline() if process.stdout else ""
        if line:
            output_lines.append(line)
            print(f"[frontend] {line.rstrip()}")
        if url_is_ready(FRONTEND_URL):
            return FRONTEND_URL

    if process.poll() is not None and output_lines:
        print("".join(output_lines))

    return FRONTEND_URL


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return

    if os.name == "nt":
        process.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        process.terminate()

    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()


def main() -> int:
    npm_command = "npm.cmd" if os.name == "nt" else "npm"
    if not command_exists(npm_command):
        print("npm is required to run the frontend demo.")
        return 1

    backend_command = [
        sys.executable,
        "-m",
        "uvicorn",
        "app:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]
    frontend_command = [npm_command, "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]

    backend: subprocess.Popen | None = None
    if url_is_ready(BACKEND_HEALTH_URL):
        print(f"[demo] Reusing running FastAPI backend at {BACKEND_URL}")
    else:
        print("[demo] Starting FastAPI backend on http://localhost:8000")
        backend = start_process(backend_command, PROJECT_ROOT)

    if not wait_for_backend(backend):
        if backend is not None:
            stop_process(backend)
        return 1

    print("[demo] Starting React frontend")
    frontend = start_process(frontend_command, FRONTEND_ROOT)
    frontend_url = wait_for_frontend(frontend)

    if frontend.poll() is not None:
        output = frontend.stdout.read() if frontend.stdout else ""
        print(output)
        if backend is not None:
            stop_process(backend)
        print("[demo] Frontend exited before startup completed.")
        return frontend.returncode or 1

    print(f"[demo] Opening {frontend_url}")
    webbrowser.open(frontend_url)
    print("[demo] Press Ctrl+C to stop backend and frontend.")

    try:
        while (backend is None or backend.poll() is None) and frontend.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[demo] Stopping demo stack...")
    finally:
        stop_process(frontend)
        if backend is not None:
            stop_process(backend)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
