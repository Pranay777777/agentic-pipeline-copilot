"""Sandbox CLI.

    python -m copilot.sandbox build [--image copilot-sandbox:0.1]   # the pinned image
    python -m copilot.sandbox check [--image copilot-sandbox:0.1]   # prove the isolation

`check` starts the real container with the real flags and a probe instead of
a notebook, and fails unless every escape it tries is blocked - the evidence
step 69's exit criterion asks for ("sandbox verified isolated").
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from copilot.sandbox import harness
from copilot.sandbox.runner import DEFAULT_IMAGE, DockerSandbox, Limits, Sandbox, build_command

CONTEXT = Path("sandbox")
EXPECTED = {
    "network": "blocked",
    "dns": "blocked",
    "write_job": "blocked",
    "write_root": "blocked",
    "write_out": "allowed",
    "write_tmp": "allowed",
}


def isolation(sandbox: Sandbox) -> tuple[bool, dict[str, Any]]:
    """Run the probe in the sandbox; True only if every escape was blocked."""
    with tempfile.TemporaryDirectory(prefix="copilot-probe-") as tmp:
        job, out = Path(tmp) / "job", Path(tmp) / "out"
        job.mkdir()
        out.mkdir()
        (job / "job.json").write_text(json.dumps({"mode": "probe"}), encoding="utf-8")
        (job / "harness.py").write_text(Path(harness.__file__).read_text("utf-8"), "utf-8")
        execution = sandbox.run(job, out)
        result_file = out / "result.json"
        if not result_file.exists():
            return False, {"error": (execution.stderr or execution.stdout)[-800:]}
        found: dict[str, Any] = json.loads(result_file.read_text("utf-8"))["probe"]
    ok = all(str(found.get(k, "")).startswith(v) for k, v in EXPECTED.items())
    ok = ok and found.get("uid") != 0 and not found.get("env_secrets")
    return ok, found


def main(argv: list[str] | None = None, sandbox: Sandbox | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="copilot.sandbox",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=["build", "check"])
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="must match SANDBOX_IMAGE")
    args = parser.parse_args(argv)
    if args.command == "build":
        command = build_command(args.image, CONTEXT)
        print(" ".join(command))
        return subprocess.run(command, check=False).returncode  # noqa: S603 - fixed argv
    sandbox = sandbox or DockerSandbox(args.image, Limits(timeout_s=60))
    ok, found = isolation(sandbox)
    for key, value in found.items():
        print(f"{key:<12} {value}")
    print("isolated" if ok else "NOT ISOLATED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
