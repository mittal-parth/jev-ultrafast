"""Run the local research fixture with Jev and Sarvam Prism in parallel; record both."""

import argparse
import base64
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from browser_harness.admin import ensure_daemon  # noqa: E402

from jev_ultrafast import Agent  # noqa: E402
from jev_ultrafast.demo import load_environment  # noqa: E402
from jev_ultrafast.model import use_decision_provider  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "jev_ultrafast" / "static"
PORT = int(os.environ.get("COMPARE_DEMO_PORT", "8767"))
ORIGIN = f"http://127.0.0.1:{PORT}"

GOAL = (
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
        files = {
            "/fixture.html": ("fixture.html", "text/html"),
            "/": ("fixture.html", "text/html"),
        }
        if path not in files:
            self.send_response(404)
            self.end_headers()
            return
        name, mime = files[path]
        body = (STATIC / name).read_text()
        if "scenario=research" not in self.path and path.endswith("fixture.html"):
            body = body  # research is selected via query below
        payload = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", mime + "; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args):
        pass


def serve():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def verify(page):
    text = page.get("text", "")
    passed = "A browser is a choice, not a conversation" in text and "Freshness is part of correctness" in text
    return {"passed": passed, "url": page.get("url", "")}


def capture_screencast(agent, folder, epoch_holder, errors):
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
                    timestamp = max(0, round((p["metadata"]["timestamp"] - epoch_holder[0]) * 1000))
                    (frames / f"{timestamp:06d}.jpg").write_bytes(base64.b64decode(p["data"]))
                    agent.browser.call("Page.screencastFrameAck", sessionId=p["sessionId"])
                stop.wait(0.015)
        except Exception as exc:
            errors.append(str(exc))

    agent.browser.call(
        "Page.startScreencast", format="jpeg", quality=80, maxWidth=960, maxHeight=640, everyNthFrame=2
    )
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return stop, thread, frames


def run_arm(name, provider, url, out_dir, results):
    use_decision_provider(provider)

    folder = out_dir / name
    folder.mkdir(parents=True, exist_ok=True)
    errors = []
    agent = Agent(url, GOAL, record_dir=folder / "step_frames", screenshots=True)
    (folder / "frames").mkdir(exist_ok=True)
    (folder / "frames" / "000000.jpg").write_bytes(
        base64.b64decode(
            agent.browser.call("Page.captureScreenshot", format="jpeg", quality=85)["data"]
        )
    )
    epoch_holder = [time.time()]
    stop, worker, screencast = capture_screencast(agent, folder, epoch_holder, errors)
    state = None
    try:
        for state in agent.run():
            action = state["history"][-1]["action"] if state["history"] else ""
            print(f"[{name}]", state["elapsed_ms"], state["status"], action, flush=True)
    finally:
        time.sleep(0.1)
        stop.set()
        worker.join(timeout=3)
        try:
            agent.browser.call("Page.stopScreencast")
        except Exception:
            pass
        state = agent.snapshot()
        state["final_page"] = agent.browser.observe(screenshot=False)
        state["verification"] = verify(state["final_page"])
        state["provider"] = provider
        state["recording_errors"] = errors
        state["screencast_dir"] = str(screencast)
        (folder / "state.json").write_text(json.dumps(state, indent=2))
        agent.close()
    results[name] = state
    print(f"[{name}] verification:", state["verification"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=ROOT / "artifacts" / "compare")
    args = parser.parse_args()
    load_environment()
    os.environ.setdefault("BU_CDP_URL", "http://127.0.0.1:9222")
    ensure_daemon()
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    server = serve()
    url = f"{ORIGIN}/fixture.html?scenario=research"
    (out_dir / "goal.txt").write_text(GOAL)
    results = {}
    for name, provider in (("jev", "jev"), ("sarvam", "sarvam")):
        run_arm(name, provider, url, out_dir, results)
    server.shutdown()
    summary = {
        name: {
            "elapsed_ms": results[name]["elapsed_ms"],
            "status": results[name]["status"],
            "decisions": len(results[name]["decisions"]),
            "actions": len(results[name]["history"]),
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
    if not all(v["verified"] for v in summary.values()):
        raise SystemExit("One or both runs failed verification")


if __name__ == "__main__":
    main()
