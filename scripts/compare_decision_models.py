"""Run Jev and Sarvam Prism on the same browser task; record screencasts for comparison."""

import argparse
import base64
import json
import os
import shutil
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from browser_harness.admin import ensure_daemon  # noqa: E402

from examples.flights import GOALS, URL, verify  # noqa: E402
from jev_ultrafast import Agent  # noqa: E402
from jev_ultrafast.demo import load_environment  # noqa: E402
from jev_ultrafast.model import use_decision_provider  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "jev_ultrafast" / "static"
PORT = int(os.environ.get("COMPARE_DEMO_PORT", "8767"))
ORIGIN = f"http://127.0.0.1:{PORT}"

RESEARCH_GOAL = (
    "Open the article titled 'A browser is a choice, not a conversation' from the reading room. "
    "Stop when the full article text is visible."
)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.headers.get("Host") != f"127.0.0.1:{PORT}":
            self.send_response(403)
            self.end_headers()
            return
        path = urlparse(self.path).path
        files = {"/fixture.html": ("fixture.html", "text/html"), "/": ("fixture.html", "text/html")}
        if path not in files:
            self.send_response(404)
            self.end_headers()
            return
        name, mime = files[path]
        payload = (STATIC / name).read_text().encode()
        self.send_response(200)
        self.send_header("Content-Type", mime + "; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args):
        pass


def serve():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def research_verify(page):
    text = page.get("text", "")
    passed = "A browser is a choice, not a conversation" in text and "Freshness is part of correctness" in text
    return {"passed": passed, "url": page.get("url", "")}


def configure_text(provider):
    """TYPE_TEXT via Sarvam chat (same subscription key as the Sarvam arm). Jev arm uses System One."""
    os.environ.pop("TEXT_HELPER", None)
    os.environ.pop("TEXT_MODEL_API_KEY", None)
    os.environ.pop("TEXT_MODEL_BASE_URL", None)
    os.environ["TEXT_MODEL_API_KEY"] = os.environ["SARVAM_API_KEY"]
    os.environ["TEXT_MODEL_BASE_URL"] = "https://api.sarvam.ai/v1"
    os.environ["TEXT_MODEL"] = os.environ.get("TEXT_MODEL", "sarvam-105b-conversations")
    os.environ["TEXT_MODEL_REASONING"] = "none"
    if provider == "jev":
        os.environ["TEXT_HELPER"] = "systemone"
        os.environ.pop("TEXT_MODEL_API_KEY", None)
        os.environ.pop("TEXT_MODEL_BASE_URL", None)


def capture_screencast(agent, folder, epoch_holder, errors, *, width=1120, height=780):
    from browser_harness.helpers import drain_events

    frames = folder / "screencast"
    frames.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()

    def worker():
        try:
            while not stop.is_set():
                for event in drain_events():
                    if event["method"] != "Page.screencastFrame":
                        continue
                    if event.get("session_id") != agent.browser.session:
                        continue
                    p = event["params"]
                    anchor = epoch_holder[0] if epoch_holder[0] is not None else p["metadata"]["timestamp"]
                    timestamp = max(0, round((p["metadata"]["timestamp"] - anchor) * 1000))
                    (frames / f"{timestamp:06d}.jpg").write_bytes(base64.b64decode(p["data"]))
                    agent.browser.call("Page.screencastFrameAck", sessionId=p["sessionId"])
                stop.wait(0.015)
        except Exception as exc:
            errors.append(str(exc))

    agent.browser.call(
        "Page.startScreencast",
        format="jpeg",
        quality=80,
        maxWidth=width,
        maxHeight=height,
        everyNthFrame=2,
    )
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return stop, thread, frames


def run_arm(name, provider, url, out_dir, results, *, verify_fn, task_label):
    use_decision_provider(provider)
    configure_text(provider)

    folder = out_dir / name
    folder.mkdir(parents=True, exist_ok=True)
    errors = []
    goal = GOALS if task_label == "flights" else RESEARCH_GOAL
    agent = Agent(url, goal, record_dir=folder / "step_frames", screenshots=True)
    (folder / "frames").mkdir(exist_ok=True)
    (folder / "frames" / "000000.jpg").write_bytes(
        base64.b64decode(agent.browser.call("Page.captureScreenshot", format="jpeg", quality=85)["data"])
    )
    epoch_holder = [time.time()]
    stop, worker, screencast = capture_screencast(agent, folder, epoch_holder, errors)
    state = None
    try:
        for state in agent.run():
            action = state["history"][-1]["action"] if state["history"] else ""
            print(f"[{name}]", state["elapsed_ms"], state["status"], action, flush=True)
    finally:
        time.sleep(0.15)
        stop.set()
        worker.join(timeout=3)
        try:
            agent.browser.call("Page.stopScreencast")
        except Exception:
            pass
        state = agent.snapshot()
        state["final_page"] = agent.browser.observe(screenshot=False)
        state["verification"] = verify_fn(state["final_page"])
        state["provider"] = provider
        state["task"] = task_label
        state["recording_errors"] = errors
        state["screencast_dir"] = str(screencast)
        (folder / "state.json").write_text(json.dumps(state, indent=2))
        agent.close()
    results[name] = state
    print(f"[{name}] verification:", state["verification"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=ROOT / "artifacts" / "compare")
    parser.add_argument(
        "--task",
        choices=("research", "flights"),
        default="flights",
        help="research fixture (clicks only) or live Google Flights",
    )
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Still write summary/video artifacts when verification fails",
    )
    args = parser.parse_args()
    load_environment()
    os.environ.setdefault("BU_CDP_URL", "http://127.0.0.1:9222")
    if args.task == "flights":
        os.environ["TRIM_LARGE_ACTION_SPACES"] = "1"
    ensure_daemon()
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    server = None
    if args.task == "flights":
        url = URL
        verify_fn = verify
        goal = GOALS
    else:
        server = serve()
        url = f"{ORIGIN}/fixture.html?scenario=research"
        verify_fn = research_verify
        goal = RESEARCH_GOAL
    (out_dir / "goal.txt").write_text(goal)
    (out_dir / "task.json").write_text(json.dumps({"task": args.task, "url": url}, indent=2))
    results = {}
    attempts = 3 if args.task == "flights" else 1
    for name, provider in (("jev", "jev"), ("sarvam", "sarvam")):
        best_dir = None
        best_ms = -1
        for attempt in range(1, attempts + 1):
            if attempt > 1:
                shutil.rmtree(out_dir / name, ignore_errors=True)
            try:
                run_arm(name, provider, url, out_dir, results, verify_fn=verify_fn, task_label=args.task)
                elapsed = results[name]["elapsed_ms"]
                if elapsed > best_ms:
                    best_ms = elapsed
                    if best_dir:
                        shutil.rmtree(best_dir, ignore_errors=True)
                    best_dir = out_dir / f".{name}-best"
                    shutil.copytree(out_dir / name, best_dir)
                if results[name]["verification"]["passed"]:
                    break
                if attempt < attempts:
                    print(f"[{name}] attempt {attempt} did not verify; retrying", flush=True)
            except Exception as error:
                print(f"[{name}] attempt {attempt} failed: {error}", flush=True)
                if attempt == attempts and name not in results:
                    raise
        if best_dir and not results.get(name, {}).get("verification", {}).get("passed"):
            shutil.rmtree(out_dir / name, ignore_errors=True)
            shutil.move(str(best_dir), str(out_dir / name))
            results[name] = json.loads((out_dir / name / "state.json").read_text())
            print(f"[{name}] kept longest run at {results[name]['elapsed_ms']} ms for comparison", flush=True)
    if server:
        server.shutdown()
    summary = {
        name: {
            "elapsed_ms": results[name]["elapsed_ms"],
            "status": results[name]["status"],
            "decisions": len(results[name]["decisions"]),
            "actions": len(results[name]["history"]),
            "text_calls": len(results[name].get("text_calls", [])),
            "median_decision_ms": sorted(d["latency_ms"] for d in results[name]["decisions"])[
                len(results[name]["decisions"]) // 2
            ]
            if results[name]["decisions"]
            else None,
            "verified": results[name]["verification"]["passed"],
        }
        for name in results
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not all(v["verified"] for v in summary.values()) and not args.allow_unverified:
        raise SystemExit("One or both runs failed verification")


if __name__ == "__main__":
    main()
