"""Install / manage the quant_signal stream stack as macOS launchd agents.

The stream modules (predictor, execution, simulation, producer, materializer),
the Flink watchdog, and the dashboard API are all long-running processes that
historically ran via `nohup` from a terminal — which died when the terminal
closed or the Mac rebooted. This script turns each into a launchd LaunchAgent:

  - RunAtLoad=true  -> starts at user login (survives reboot)
  - KeepAlive=true  -> launchd restarts it if it crashes or exits
  - ThrottleInterval=10 -> min seconds between crash restarts (avoids hot loops)
  - runs scripts/service.sh, which sources .env then execs the venv python,
    so API keys never live in the plist files

launchd is chosen over a docker-compose of these apps because they already run
natively against host Redpanda/Redis (ports 9092/6380); wrapping them in
containers would require host.docker.internal wiring, env passing, and rebuilds
on every code change. The infra containers (redpanda/redis/flink) keep their
own compose file with restart: unless-stopped.

Usage:
  uv run python -m scripts.install_services install     # write plists, take over
  uv run python -m scripts.install_services status      # launchctl + log tail
  uv run python -m scripts.install_services logs api    # tail one service log
  uv run python -m scripts.install_services uninstall   # stop and remove agents
"""

from __future__ import annotations

import argparse
import os
import plistlib
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
LAUNCH_AGENTS_DIR = Path.home() / "Library" / "LaunchAgents"
LABEL_PREFIX = "com.quantsignal"

# name -> (args passed to `python -m ...`, pkill pattern for legacy nohup procs)
# Patterns must NOT start with a dash (macOS pkill treats a leading "-" as an
# option) and must anchor on the venv python path so we never kill unrelated
# processes (e.g. the installer itself).
SERVICES: dict[str, tuple[list[str], str]] = {
    "producer": (
        ["stream.producer"],
        r"\.venv/bin/python[^ ]* -m stream\.producer\b",
    ),
    "signal": (
        ["stream.xs_signal"],
        r"\.venv/bin/python[^ ]* -m stream\.xs_signal\b",
    ),
    "execution": (
        ["stream.execution"],
        r"\.venv/bin/python[^ ]* -m stream\.execution\b",
    ),
    "simulation": (
        ["stream.simulation"],
        r"\.venv/bin/python[^ ]* -m stream\.simulation\b",
    ),
    "materializer": (
        ["stream.materializer"],
        r"\.venv/bin/python[^ ]* -m stream\.materializer\b",
    ),
    "watchdog": (
        ["scripts.stream_watchdog", "--interval", "60", "--fix"],
        r"\.venv/bin/python[^ ]* -m scripts\.stream_watchdog\b",
    ),
    "api": (
        ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"],
        r"\.venv/bin/(python[^ ]* )?uvicorn api\.main:app\b",
    ),
}

LOG_PATH = {name: Path(f"/tmp/stream_{name}.log") for name in SERVICES}


def label(name: str) -> str:
    return f"{LABEL_PREFIX}.{name}"


def plist_path(name: str) -> Path:
    return LAUNCH_AGENTS_DIR / f"{label(name)}.plist"


def write_plist(name: str) -> Path:
    service_script = PROJECT_DIR / "scripts" / "service.sh"
    args, _ = SERVICES[name]
    plist: dict = {
        "Label": label(name),
        "ProgramArguments": [str(service_script), *args],
        "WorkingDirectory": str(PROJECT_DIR),
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": 10,
        "EnvironmentVariables": {"PYTHONUNBUFFERED": "1"},
        "StandardOutPath": str(LOG_PATH[name]),
        "StandardErrorPath": str(LOG_PATH[name]),
    }
    LAUNCH_AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    path = plist_path(name)
    with path.open("wb") as fh:
        plistlib.dump(plist, fh)
    return path


def run(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def stop_legacy(name: str) -> None:
    """Kill old nohup-style processes so launchd is the single owner."""
    _, pattern = SERVICES[name]
    result = run(["pkill", "-f", pattern])
    if result.returncode == 0:
        print(f"  stopped legacy {name} process(es)")
    elif result.returncode == 1:
        print(f"  no legacy {name} process running")
    else:
        print(f"  pkill for {name} returned {result.returncode}: {result.stderr.strip()}")


def install(name: str) -> None:
    path = write_plist(name)
    stop_legacy(name)
    result = run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)])
    if result.returncode == 0:
        print(f"  installed {label(name)}")
    else:
        # Already bootstrapped is fine; any other error is not.
        if "already bootstrapped" in result.stderr or "Bootstrap failed" in result.stderr:
            print(f"  {label(name)} already loaded")
        else:
            print(f"  bootstrap {name} failed: {result.stderr.strip()}")


def uninstall(name: str) -> None:
    result = run(["launchctl", "bootout", f"gui/{os.getuid()}/{label(name)}"])
    if result.returncode == 0 or "Could not find service" in result.stderr:
        print(f"  stopped {label(name)}")
    else:
        print(f"  bootout {name} failed: {result.stderr.strip()}")
    path = plist_path(name)
    if path.exists():
        path.unlink()
        print(f"  removed {path}")


def status(name: str) -> None:
    path = plist_path(name)
    print(f"== {name} ({label(name)}) ==")
    if not path.exists():
        print("  not installed")
        return
    result = run(["launchctl", "print", f"gui/{os.getuid()}/{label(name)}"])
    if result.returncode != 0:
        print("  not loaded (no launchd record)")
    else:
        for line in result.stdout.splitlines():
            if line.strip().startswith(("state =", "pid =", "last exit code")):
                print(f"  {line.strip()}")
    log = LOG_PATH[name]
    if log.exists():
        size = log.stat().st_size
        tail = ""
        if size:
            tail = run(["tail", "-n", "2", str(log)]).stdout.strip()
        print(f"  log {log} ({size} bytes)")
        if tail:
            print(f"  {tail}")
    print()


def cmd_install(_: argparse.Namespace) -> None:
    print("writing launchd agents under", LAUNCH_AGENTS_DIR)
    for name in SERVICES:
        install(name)


def cmd_uninstall(_: argparse.Namespace) -> None:
    for name in SERVICES:
        uninstall(name)


def cmd_status(args: argparse.Namespace) -> None:
    names = [args.service] if args.service else list(SERVICES)
    for name in names:
        status(name)


def cmd_logs(args: argparse.Namespace) -> None:
    if args.service not in SERVICES:
        print(f"unknown service {args.service}; choices: {', '.join(SERVICES)}")
        sys.exit(1)
    log = LOG_PATH[args.service]
    if not log.exists():
        print(f"no log at {log}")
        sys.exit(1)
    print(f"--- {log} ---")
    subprocess.run(["tail", "-n", str(args.lines), str(log)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("install", help="write plists and take over from nohup").set_defaults(
        fn=cmd_install
    )
    sub.add_parser("uninstall", help="stop and remove all agents").set_defaults(fn=cmd_uninstall)
    st = sub.add_parser("status", help="show launchd state for each service")
    st.add_argument("service", nargs="?", default=None, help="single service name")
    st.set_defaults(fn=cmd_status)
    lg = sub.add_parser("logs", help="tail a service log")
    lg.add_argument("service", choices=list(SERVICES))
    lg.add_argument("-n", "--lines", type=int, default=30)
    lg.set_defaults(fn=cmd_logs)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
