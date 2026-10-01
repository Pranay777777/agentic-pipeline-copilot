"""Start the sandbox container with every isolation flag, and enforce the time limit.

The container gets:

- **no network** (`--network none`) - generated code cannot exfiltrate data
  or fetch anything; the hostname is `localhost` so Spark can resolve itself;
- a **read-only root filesystem** and a read-only job mount; the only
  writable places are a size-capped `/tmp` (nosuid, nodev) and the `/out`
  directory for the result;
- **no capabilities**, `no-new-privileges`, a non-root user;
- **CPU, memory (no swap) and process caps**;
- a **hard timeout**, after which the container is killed, not asked.

No host environment variable is passed in, so no key can leak into it.
"""

from __future__ import annotations

import subprocess
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

DEFAULT_IMAGE = "copilot-sandbox:0.2"
"""0.2 adds ruff, mypy and pytest for the static gate and generated tests (ADR-006)."""


@dataclass(frozen=True)
class Limits:
    cpus: float = 2.0
    memory: str = "2g"
    pids: int = 512
    tmp_size: str = "1g"
    timeout_s: float = 240.0


@dataclass(frozen=True)
class Execution:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    seconds: float


class Sandbox(Protocol):
    def run(self, job: Path, out: Path) -> Execution: ...


Runner = Callable[..., Any]


def docker_command(
    image: str, job: Path, out: Path, limits: Limits, name: str, docker: str = "docker"
) -> list[str]:
    """The exact `docker run` invocation - kept pure so the flags are unit-tested."""
    return [
        docker,
        "run",
        "--rm",
        "--name",
        name,
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--user",
        "1000:1000",
        "--cpus",
        str(limits.cpus),
        "--memory",
        limits.memory,
        "--memory-swap",
        limits.memory,
        "--pids-limit",
        str(limits.pids),
        "--tmpfs",
        # exec: Spark's native codecs (snappy) unpack a .so into /tmp and load it; the
        # generated Python can already run anything in here, so noexec adds nothing.
        f"/tmp:rw,exec,nosuid,nodev,size={limits.tmp_size}",  # noqa: S108 - in the container
        "--env",
        "HOME=/tmp",
        # With no network the container's random hostname resolves to nothing, and
        # Spark's JVM exits looking itself up; localhost always resolves (/etc/hosts).
        "--hostname",
        "localhost",
        "--env",
        "SPARK_LOCAL_IP=127.0.0.1",
        "--env",
        "SPARK_LOCAL_HOSTNAME=localhost",
        "--volume",
        f"{job.resolve()}:/job:ro",
        "--volume",
        f"{out.resolve()}:/out:rw",
        image,
        "python",
        "/job/harness.py",
    ]


class DockerSandbox:
    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        limits: Limits | None = None,
        docker: str = "docker",
        runner: Runner = subprocess.run,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.image, self.limits, self.docker = image, limits or Limits(), docker
        self._run, self._clock = runner, clock

    def run(self, job: Path, out: Path) -> Execution:
        out.chmod(0o777)  # the container's non-root user writes the result here
        name = f"copilot-sandbox-{uuid.uuid4().hex[:12]}"
        command = docker_command(self.image, job, out, self.limits, name, self.docker)
        started = self._clock()
        try:
            done = self._run(
                command,
                capture_output=True,
                text=True,
                timeout=self.limits.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            self._kill(name)
            return Execution(
                -1, _text(exc.stdout), _text(exc.stderr), True, self._clock() - started
            )
        except FileNotFoundError:
            message = f"'{self.docker}' was not found - install Docker and start it"
            return Execution(-1, "", message, False, 0.0)
        return Execution(done.returncode, done.stdout, done.stderr, False, self._clock() - started)

    def _kill(self, name: str) -> None:
        self._run(
            [self.docker, "kill", name], capture_output=True, text=True, timeout=30, check=False
        )

    def available(self) -> bool:
        """True when the daemon answers and the image exists."""
        try:
            done = self._run(
                [self.docker, "image", "inspect", self.image],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
        return bool(done.returncode == 0)


def _text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value or ""


def build_command(image: str, context: Path, docker: str = "docker") -> Sequence[str]:
    return [docker, "build", "--tag", image, str(context)]
