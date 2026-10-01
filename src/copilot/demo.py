"""Demo GIF (step 82): draw a real CLI transcript as an animated terminal.

    python -m copilot.agents run "<spec>" --replay tests/cassettes/orders_silver.jsonl \\
        --out runs/demo/orders.py 2>&1 | tee runs/demo.txt
    python -m copilot.demo runs/demo.txt --command 'python -m copilot.agents run "<spec>"' \\
        --append "pull request: https://github.com/<owner>/copilot-playground/pull/1"

The transcript is a replayed run - the recorded model answers, executed for real
in the sandbox - so the GIF is reproducible and costs no quota. Nothing is
typed by hand: every line after the command comes from the transcript.
Pillow is a dev dependency; it is imported only here.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Any

WIDTH = 96  # columns
BACKGROUND = (24, 26, 33)
COLOURS = {
    "command": (125, 207, 255),
    "good": (126, 211, 33),
    "bad": (255, 107, 107),
    "dim": (150, 156, 170),
    "text": (230, 232, 236),
}
FONTS = ("DejaVuSansMono.ttf", "consola.ttf", "Menlo.ttc", "LiberationMono-Regular.ttf")


def colour(line: str) -> tuple[int, int, int]:
    lowered = line.lower()
    if line.startswith("$ "):
        return COLOURS["command"]
    if any(w in lowered for w in ("sent back", "fail", "error", "rejected", "escalated")):
        return COLOURS["bad"]
    if lowered.startswith(("ready", "reviewed", "pull request")) or ": ok" in lowered:
        return COLOURS["good"]
    if lowered.startswith(("replaying", "running with")):
        return COLOURS["dim"]
    return COLOURS["text"]


def screens(command: str, lines: list[str]) -> list[tuple[list[str], int, int]]:
    """(visible lines, milliseconds, command rows) per frame: the command types, then lines."""
    wrapped = [w for line in lines for w in (textwrap.wrap(line, WIDTH) or [""])]
    shown: list[tuple[list[str], int, int]] = []
    prompt = f"$ {command}"
    for end in range(3, len(prompt) + 1, 3):
        part = textwrap.wrap(prompt[:end], WIDTH)
        shown.append((part, 40, len(part)))
    typed = textwrap.wrap(prompt, WIDTH)
    shown.append((typed, 600, len(typed)))
    for n in range(1, len(wrapped) + 1):
        shown.append(([*typed, *wrapped[:n]], 700, len(typed)))
    last, _, rows = shown[-1]
    shown[-1] = (last, 5000, rows)
    return shown


def _font(size: int) -> Any:
    from PIL import ImageFont

    for name in FONTS:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)  # pragma: no cover - every CI image has DejaVu


def render(command: str, lines: list[str], out: Path, size: int = 16) -> int:
    """Write the GIF; returns the number of frames."""
    from PIL import Image, ImageDraw

    font = _font(size)
    cell_w = float(font.getlength("M")) or size / 2
    cell_h = size + 6
    frames = screens(command, lines)
    rows = max(len(visible) for visible, _, _ in frames)
    width, height = int(cell_w * (WIDTH + 4)) + 1, cell_h * (rows + 2)
    images = []
    for visible, _, command_rows in frames:
        image = Image.new("RGB", (width, height), BACKGROUND)
        draw = ImageDraw.Draw(image)
        for row, text in enumerate(visible):
            fill = COLOURS["command"] if row < command_rows else colour(text)
            draw.text((int(cell_w * 2), cell_h * (row + 1)), text, font=font, fill=fill)
        images.append(image)
    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        out,
        save_all=True,
        append_images=images[1:],
        duration=[ms for _, ms, _ in frames],
        loop=0,
        optimize=True,
    )
    return len(images)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="copilot.demo", description=__doc__)
    parser.add_argument("transcript", type=Path)
    parser.add_argument("--command", required=True, help="the command line shown being typed")
    parser.add_argument("--append", action="append", default=[], help="a closing line to add")
    parser.add_argument("--out", type=Path, default=Path("docs/images/demo.gif"))
    args = parser.parse_args(argv)
    lines = [x.rstrip() for x in args.transcript.read_text(encoding="utf-8").splitlines()]
    count = render(args.command, [x for x in lines if x] + args.append, args.out)
    print(f"{args.out} ({count} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
