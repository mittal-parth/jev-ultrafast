"""Side-by-side comparison video with independent elapsed timers."""

import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path, help="compare_decision_models output directory")
parser.add_argument(
    "-o",
    "--output",
    type=Path,
    default=ROOT / "artifacts" / "jev-vs-sarvam-prism.mp4",
)
args = parser.parse_args()
source = args.source.resolve()
jev = json.loads((source / "jev" / "state.json").read_text())
sarvam = json.loads((source / "sarvam" / "state.json").read_text())
task = "flights"
if (source / "task.json").exists():
    task = json.loads((source / "task.json").read_text()).get("task", task)
hold_ms = 2500 if task == "flights" else 800


def load_frames(run_dir):
    sc = run_dir / "screencast"
    frames = [(0, Image.open(run_dir / "frames" / "000000.jpg").convert("RGB"))]
    if sc.exists():
        frames += sorted((int(p.stem), Image.open(p).convert("RGB")) for p in sc.glob("*.jpg"))
    return frames


jev_frames = load_frames(source / "jev")
sarvam_frames = load_frames(source / "sarvam")
end = max(jev["elapsed_ms"], sarvam["elapsed_ms"], 1) + hold_ms
folder = source / "video-frames"
folder.mkdir(parents=True, exist_ok=True)

def _load_font(size, bold=False):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def font(n, bold=False):
    return _load_font(n, bold)


def mono(n):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, n)
        except OSError:
            continue
    return ImageFont.load_default()


def pick(frames, t):
    return next(im for ts, im in reversed(frames) if ts <= t)


ink, muted, jev_green, sarvam_orange = "#172a20", "#6a766c", "#2a743f", "#c45c26"
W, H = 1600, 900
fps = 30


def panel(draw, x, y, w, h, title, subtitle, color):
    draw.rounded_rectangle((x, y, x + w, y + h), radius=12, fill="#202124")
    for j, c in enumerate(["#de8278", "#d6bd6e", "#8dbd8a"]):
        draw.ellipse((x + 14 + j * 16, y + 14, x + 23 + j * 16, y + 23), fill=c)
    draw.text((x + 58, y + 10), subtitle, font=mono(11), fill="#d4d6d5")
    draw.text((x + w - 180, y + 8), title, font=font(14, True), fill=color)


for i in range(round(end * fps / 1000)):
    t = min(end, round(i * 1000 / fps))
    canvas = Image.new("RGB", (W, H), "#f3f4ec")
    d = ImageDraw.Draw(canvas)
    headline = (
        "Zürich → London · Google Flights · 1×"
        if task == "flights"
        else "Decision models · same task · same fixture"
    )
    d.text((36, 22), headline, font=font(28, True), fill=ink)
    subtitle = (
        "Jev vs Sarvam Prism · same goal · text via each provider's API key"
        if task == "flights"
        else "Jev (TypeSafe) vs Sarvam Prism — timers run independently"
    )
    d.text((38, 62), subtitle, font=font(16), fill=muted)

    pw, ph = 740, 520
    py = 110
    for col, (label, color, frames, state, model_name) in enumerate(
        [
            ("JEV", jev_green, jev_frames, jev, jev["decisions"][0]["model"] if jev["decisions"] else "jev"),
            (
                "SARVAM PRISM",
                sarvam_orange,
                sarvam_frames,
                sarvam,
                sarvam["decisions"][0]["model"] if sarvam["decisions"] else "sarvam-prism",
            ),
        ]
    ):
        px = 36 + col * (pw + 48)
        chrome_label = "google.com/travel/flights" if task == "flights" else "127.0.0.1 · research fixture"
        panel(d, px, py, pw, ph, label, chrome_label, color)
        elapsed = state["elapsed_ms"] if t >= state["elapsed_ms"] else t
        shot = pick(frames, elapsed)
        sw, sh = shot.size
        scale = min((pw - 4) / sw, (ph - 44) / sh)
        nw, nh = int(sw * scale), int(sh * scale)
        shot = shot.resize((nw, nh), Image.Resampling.LANCZOS)
        ox, oy = px + 2 + (pw - 4 - nw) // 2, py + 36 + (ph - 44 - nh) // 2
        canvas.paste(shot, (ox, oy))
        done = t >= state["elapsed_ms"] and state["status"] == "done"
        timer_color = color if done else ink
        d.text((px + 16, py + ph + 14), f"{elapsed / 1000:05.2f}s", font=mono(36), fill=timer_color)
        d.text((px + 16, py + ph + 58), model_name, font=font(14), fill=muted)
        decisions = [x for x in state["decisions"] if x["elapsed_ms"] <= elapsed]
        if decisions:
            med = sorted(x["latency_ms"] for x in decisions)[len(decisions) // 2]
            d.text((px + 200, py + ph + 24), f"median decision {med:.0f} ms", font=font(14), fill=muted)
        status = state["status"].upper() if t >= state["elapsed_ms"] else "RUNNING"
        d.text((px + pw - 140, py + ph + 24), status, font=font(16, True), fill=timer_color)

    winner = None
    if t >= jev["elapsed_ms"] and t >= sarvam["elapsed_ms"]:
        if jev["elapsed_ms"] < sarvam["elapsed_ms"]:
            winner = f"Jev finished first ({jev['elapsed_ms']} ms vs {sarvam['elapsed_ms']} ms)"
        elif sarvam["elapsed_ms"] < jev["elapsed_ms"]:
            winner = f"Sarvam Prism finished first ({sarvam['elapsed_ms']} ms vs {jev['elapsed_ms']} ms)"
        else:
            winner = f"Tie at {jev['elapsed_ms']} ms"
    d.line((36, H - 48, W - 36, H - 48), fill="#d3d9cc", width=2)
    d.line((36, H - 48, 36 + (W - 72) * t / end, H - 48), fill=jev_green, width=3)
    verified = jev.get("verification", {}).get("passed") and sarvam.get("verification", {}).get("passed")
    footer = winner or (
        (
            "Live Google Flights · Jev System One text vs Sarvam 105B text · "
            + ("verified" if verified else "headless run (calendar did not finish on either arm)")
        )
        if task == "flights"
        else "Reading-room demo · click-only task · no text helper"
    )
    d.text((36, H - 32), footer, font=font(13), fill=muted)
    canvas.save(folder / f"{i:04d}.png")

args.output.parent.mkdir(parents=True, exist_ok=True)
subprocess.run(
    [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        str(folder / "%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-movflags",
        "+faststart",
        str(args.output),
    ],
    check=True,
)
print("Wrote", args.output, "duration", round(end / 1000, 2), "s")
