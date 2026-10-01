"""Demo GIF: a transcript drawn as an animated terminal, frame by frame."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from copilot.demo import COLOURS, colour, main, screens

LINES = [
    "replaying tests/cassettes/orders_silver.jsonl - no model, no network",
    "  plan #1: sent back - target 'orders' would overwrite a catalog table",
    "  plan #2: ok - gemini-3.8-flash, 3780 tokens",
    "  review: passed · static: clean · sandbox: 6 cell(s) in 41s · tests: 4/4 passed",
    "ready: runs/demo/orders_silver.py + runs/demo/test_orders_silver.py " + "x" * 80,
]


def test_lines_are_coloured_by_what_they_say() -> None:
    assert colour("$ python -m copilot.agents run") == COLOURS["command"]
    assert colour(LINES[1]) == COLOURS["bad"]
    assert colour(LINES[2]) == COLOURS["good"] and colour(LINES[4]) == COLOURS["good"]
    assert colour(LINES[0]) == COLOURS["dim"] and colour(LINES[3]) == COLOURS["text"]


def test_the_command_types_then_the_transcript_appears() -> None:
    frames = screens("run it", LINES)
    assert frames[0] == (["$ r"], 40, 1)
    assert frames[-1][1] == 5000 and frames[-1][0][0] == "$ run it"
    assert len(frames[-1][0]) == 1 + 6  # the long ready line wraps onto two rows
    long = screens("x" * 150, ["done"])[-1]
    assert long[2] == 2 and long[0][2] == "done"  # a long command wraps and stays "command"


def test_cli_writes_an_animated_gif(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    transcript = tmp_path / "demo.txt"
    transcript.write_text("\n".join(LINES) + "\n\n", encoding="utf-8")
    out = tmp_path / "images" / "demo.gif"
    pr = "pull request: https://github.com/owner/copilot-playground/pull/1"
    argv = [str(transcript), "--command", "python -m copilot.agents run", "--append", pr]
    assert main([*argv, "--out", str(out)]) == 0
    with Image.open(out) as gif:
        assert gif.format == "GIF" and getattr(gif, "n_frames", 1) > 10 and gif.info["loop"] == 0
    assert capsys.readouterr().out.startswith(f"{out} (")
