from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from demo_video_recorder import (
    DEFAULTS,
    FAST_SMOKE_TEST_DEFAULTS,
    EdgeTTSBackend,
    SubtitleStyle,
    WebUIRecorder,
)


ROOT = Path(__file__).resolve().parent
STATE_FILE = ROOT / "temp" / "webui_state.json"
OUT_DIR = ROOT / "out"
FULL_FFMPEG = Path("/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg")
FULL_FFPROBE = Path("/opt/homebrew/opt/ffmpeg-full/bin/ffprobe")
TAB_INDEX = {
    "Array": 0,
    "Solve": 1,
    "Regression": 2,
    "Pearson": 3,
    "Gradient": 4,
    "Decision Tree": 5,
    "K-means": 6,
    "Cache": 7,
}


CUES = {
    "intro_1": "Hey everyone, quick tour of my EE2211 solver.",
    "intro_2": "The exams are open book, but the clock still hurts.",
    "intro_3": "I made this so I could skip the boring coding and think harder.",
    "intro_4": "It worked well for me, so let me show the actual flow.",
    "array_tab": "I usually start with Array when I just want a quick matrix check.",
    "array_type": "The field starts blank, so I type the matrix from the question.",
    "array_parsed": "First, I check that the parsed matrix is what I meant.",
    "array_metrics": "Then I glance at shape, rank, determinant, and inverse status.",
    "array_inverse": "If an inverse shortcut exists, it appears right below.",
    "solve_tab": "For linear systems, I jump to the Solve tab.",
    "solve_type": "Here I type X and y for a least squares style question.",
    "solve_system": "The system summary tells me if it is exact or least squares.",
    "solve_solution": "This is the fitted weight vector I would copy into working.",
    "solve_fit": "The fitted target is a quick check against the original y.",
    "solve_error": "And here are fitted values, residuals, and the error.",
    "reg_tab": "Now the big one: third order regression, exam style.",
    "reg_select": "I pick polynomial regression, numeric targets, order three, and test prediction.",
    "transpose_x": "When the data is long, I type it sideways first.",
    "transpose_x_click": "Transpose X turns the row into the column the formula expects.",
    "transpose_y": "Same trick for y, which saves a surprising amount of typing.",
    "transpose_test": "And I can do the same thing for test inputs.",
    "reg_fit": "First I fit it without regularization, like the basic closed form.",
    "reg_model": "The model box confirms the setup I just chose.",
    "reg_matrix": "The polynomial matrix is built for me, including the cubic terms.",
    "reg_weights": "These weights are usually the main answer I need.",
    "reg_pred": "The app also gives predictions and training error, so I can check myself.",
    "reg_test": "For test points, it gives predicted values and test error too.",
    "ridge_select": "Now I turn on ridge to compare with regularization.",
    "ridge_fit": "Same data, new lambda, no need to recode the whole thing.",
    "ridge_model": "The form now says ridge, so I know this is the regularized run.",
    "ridge_weights": "I compare these weights against the previous ones.",
    "pearson_tab": "Pearson is for quick correlation questions.",
    "pearson_type": "I type one x column and two y columns to check both at once.",
    "pearson_result": "The result matrix gives the correlations directly.",
    "pearson_intermediate": "If I need to show steps, the means and covariance are expanded too.",
    "gradient_tab": "Gradient descent is nice when the question gives a cost function.",
    "gradient_type": "I type the expression, starting point, learning rate, and iterations.",
    "gradient_function": "It differentiates the function symbolically, which is handy for checking.",
    "gradient_history": "The history table shows every update step.",
    "gradient_final": "The final parameters are ready to copy.",
    "tree_class_tab": "Decision Tree starts with impurity calculations.",
    "tree_class_type": "I enter class counts and leave all measures selected.",
    "tree_counts": "First I check the class counts are lined up correctly.",
    "tree_gini": "Then I can read Gini, entropy, and classification error.",
    "tree_entropy": "Entropy is right below, using the same class count table.",
    "tree_error": "Classification error is there too, in case that is the measure asked.",
    "tree_reg_select": "For regression trees, I switch modes in the same tab.",
    "tree_reg_type": "I type x, y, and a threshold to test.",
    "tree_threshold": "The threshold summary breaks down left and right errors.",
    "tree_candidates": "The candidate table is useful when the best split is asked.",
    "tree_best": "This gives the best one-level split directly.",
    "kmeans_tab": "K-means is for clustering questions.",
    "kmeans_type": "I type the points, initial centroids, and an iteration limit.",
    "kmeans_summary": "The result summary tells me points, dimensions, k, and iterations.",
    "kmeans_initial": "I scroll through the first cluster assignment.",
    "kmeans_iteration": "Then the later iteration shows the centroids settling.",
    "cache_tab": "The cache is the part I use more than expected.",
    "cache_show": "Every important input and result gets saved automatically.",
    "cache_copy": "I open plain text and copy a cached matrix.",
    "cache_reuse": "Then I paste it back into Array and reuse it immediately.",
    "finish_1": "That is basically my exam-time workflow.",
    "finish_2": "Hope it saves you time too. Good luck for EE2211.",
}


class DemoDriver:
    def __init__(self, recorder: WebUIRecorder) -> None:
        self.r = recorder

    @property
    def page(self):
        return self.r.current_page

    def wait(self, seconds: float = 0.35) -> None:
        self.page.wait_for_timeout(int(seconds * 1000))

    def tab(self, name: str):
        self.r.find(role="tab", name=name).click()
        self.wait(0.7)
        panel = self.panel(name)
        panel.wait_for(state="visible", timeout=10_000)
        return panel

    def panel(self, name: str):
        return self.page.locator('[data-baseweb="tab-panel"]').nth(TAB_INDEX[name])

    def choose(self, panel, label: str, option: str) -> None:
        combo = panel.get_by_role("combobox", name=re.compile(re.escape(label)))
        self.r.element(combo).click()
        self.wait(0.35)
        self.r.element(self.page.get_by_role("option", name=option, exact=True)).click()
        self.wait(1.0)

    def click_text(self, panel, text: str) -> None:
        self.r.element(panel.get_by_text(text, exact=True)).click()
        self.wait(0.8)

    def fill_textbox(self, panel, label: str, value: str) -> None:
        locator = panel.get_by_role("textbox", name=label, exact=True)
        self.r.element(locator).fill(value)
        self.wait(0.55)

    def fill_labeled_input(self, panel, label: str, value: str) -> None:
        locator = panel.get_by_label(label, exact=True)
        self.r.element(locator).fill(value)
        self.wait(0.55)

    def click_button(self, panel, name: str, wait: float = 1.2) -> None:
        self.r.element(panel.get_by_role("button", name=name, exact=True)).click()
        self.wait(wait)

    def highlight_text(self, panel, text: str, *, exact: bool = False) -> None:
        locator = panel.get_by_text(text, exact=exact).first
        self.r.element(locator).highlight(duration_ms=1250)

    def paste_into_textbox(self, panel, label: str, value: str) -> None:
        locator = panel.get_by_role("textbox", name=label, exact=True)
        self.r.element(locator).select_clear_paste(0.55, value)
        self.wait(0.7)


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(exist_ok=True)
    reset_webui_state()

    ffmpeg = str(FULL_FFMPEG if FULL_FFMPEG.exists() else "ffmpeg")
    ffprobe = str(FULL_FFPROBE if FULL_FFPROBE.exists() else "ffprobe")
    verify_subtitle_support(ffmpeg)

    port = args.port or free_port()
    app = start_streamlit(port)
    output_path = OUT_DIR / (
        "ee2211-solver-demo-smoke.mp4" if args.smoke else "ee2211-solver-demo.mp4"
    )
    tts = None
    if not args.no_tts and not args.smoke:
        tts = EdgeTTSBackend(
            save_dir=OUT_DIR / "ee2211-solver-demo.tts",
            speaker="en-US-AvaMultilingualNeural",
            speed="+40%",
            volume="+0%",
            ffprobe=ffprobe,
            cache=True,
        )

    defaults = FAST_SMOKE_TEST_DEFAULTS if args.smoke else DEFAULTS
    recorder = WebUIRecorder(
        output_path,
        headless=True,
        viewport=(1440, 900),
        video_backend="playwright",
        scroll_duration_ms=180 if args.smoke else 950,
        action_pause_seconds=0.03 if args.smoke else 0.35,
        typed_character_delay=0 if args.smoke else 0.04,
        capture_framerate=15,
        video_scale_width=1280,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        subtitle_style=SubtitleStyle(
            font_size=10,
            alignment="bottom_center",
            outline=1,
            shadow=0,
            margin_vertical=18,
        ),
        tts=tts,
        words_per_minute=defaults.words_per_minute,
        min_pause_seconds=defaults.min_pause_seconds,
    )

    try:
        cues = recorder.prepare_cues(CUES, async_tts=tts is not None)
        url = f"http://127.0.0.1:{port}"
        recorder.open_web(url, start_recording=False, wait_until="domcontentloaded")
        recorder.current_page.set_default_timeout(12_000)
        recorder.current_page.get_by_role("heading", name="EE2211 Solver").wait_for()
        recorder.current_page.wait_for_timeout(800)
        recorder.start_recording()
        run_demo(DemoDriver(recorder), cues)
        recorder.stop_recording()
    finally:
        recorder.close()
        stop_streamlit(app)


def run_demo(demo: DemoDriver, cues: dict[str, object]) -> None:
    r = demo.r

    def show(key: str, panel, text: str, *, exact: bool = False) -> None:
        r.explain_during(
            cues[key],
            lambda: demo.highlight_text(panel, text, exact=exact),
            tail_seconds=0.45,
        )

    r.explain(cues["intro_1"])
    r.explain(cues["intro_2"])
    r.explain(cues["intro_3"])
    r.explain(cues["intro_4"])

    panel = demo.tab("Array")
    r.explain(cues["array_tab"])
    r.explain_during(
        cues["array_type"],
        lambda: demo.fill_textbox(panel, "Array", "1 2 3\n4 5 6"),
    )
    demo.click_button(panel, "Inspect", wait=1.2)
    panel.get_by_text("Parsed Matrix").first.wait_for(state="visible", timeout=12_000)
    show("array_parsed", panel, "Parsed Matrix")
    show("array_metrics", panel, "shape", exact=True)
    show("array_inverse", panel, "Right Inverse")

    panel = demo.tab("Solve")
    r.explain(cues["solve_tab"])
    r.explain_during(
        cues["solve_type"],
        lambda: (
            demo.choose(panel, "Equation", "Xw = y"),
            demo.fill_textbox(panel, "X", "1 0\n1 1\n1 2\n1 3"),
            demo.fill_textbox(panel, "y", "1\n2\n5\n10"),
        ),
    )
    demo.click_button(panel, "Solve", wait=1.2)
    panel.get_by_text("Solution").first.wait_for(state="visible", timeout=12_000)
    show("solve_system", panel, "System")
    show("solve_solution", panel, "Solution")
    show("solve_fit", panel, "Fitted Target")
    show("solve_error", panel, "Error")

    panel = demo.tab("Regression")
    r.explain(cues["reg_tab"])
    r.explain_during(
        cues["reg_select"],
        lambda: (
            demo.choose(panel, "Model", "Polynomial"),
            demo.choose(panel, "Y interpretation", "Numeric regression"),
            demo.choose(panel, "Regularization", "None"),
            demo.click_text(panel, "Predict X_test"),
            demo.fill_labeled_input(panel, "Polynomial order", "3"),
        ),
    )
    r.explain_during(
        cues["transpose_x"],
        lambda: demo.fill_textbox(panel, "X", "0 1 2 3 4 5"),
    )
    r.explain_during(
        cues["transpose_x_click"],
        lambda: demo.click_button(panel, "Transpose X", wait=1.0),
    )
    r.explain_during(
        cues["transpose_y"],
        lambda: (
            demo.fill_textbox(panel, "Y", "1 1.75 4.2 10.45 22.6 42.75"),
            demo.click_button(panel, "Transpose Y", wait=1.0),
        ),
    )
    r.explain_during(
        cues["transpose_test"],
        lambda: (
            demo.fill_textbox(panel, "X_test", "6 7"),
            demo.click_button(panel, "Transpose X_test", wait=1.0),
            demo.fill_textbox(panel, "Y_test (optional)", "73\n115.45"),
        ),
    )
    r.explain_during(cues["reg_fit"], lambda: demo.click_button(panel, "Fit", wait=1.4))
    panel.get_by_text("Weights").first.wait_for(state="visible", timeout=12_000)
    show("reg_model", panel, "Model")
    show("reg_matrix", panel, "Model Matrix")
    show("reg_weights", panel, "Weights")
    show("reg_pred", panel, "Training Prediction")
    show("reg_test", panel, "Test Prediction")
    r.explain_during(
        cues["ridge_select"],
        lambda: (
            demo.choose(panel, "Regularization", "Ridge"),
            demo.fill_labeled_input(panel, "Lambda", "0.5"),
            demo.choose(panel, "Ridge form", "primal form"),
        ),
    )
    r.explain_during(cues["ridge_fit"], lambda: demo.click_button(panel, "Fit", wait=1.4))
    show("ridge_model", panel, "Model")
    show("ridge_weights", panel, "Weights")

    panel = demo.tab("Pearson")
    r.explain(cues["pearson_tab"])
    r.explain_during(
        cues["pearson_type"],
        lambda: (
            demo.fill_textbox(panel, "X", "1\n2\n3\n4\n5"),
            demo.fill_textbox(panel, "Y", "2 5\n4 4\n5 3\n8 2\n10 1"),
        ),
    )
    demo.click_button(panel, "Calculate", wait=1.2)
    panel.get_by_text("Intermediate values").first.wait_for(state="visible", timeout=12_000)
    show("pearson_result", panel, "Result")
    show("pearson_intermediate", panel, "Intermediate values")

    panel = demo.tab("Gradient")
    r.explain(cues["gradient_tab"])
    r.explain_during(
        cues["gradient_type"],
        lambda: (
            demo.fill_textbox(panel, "Cost function C(...)", "x**2 + 2*y**2 + x*y"),
            demo.fill_textbox(panel, "Initial values", "3, -2"),
            demo.fill_labeled_input(panel, "Learning rate", "0.1"),
            demo.fill_labeled_input(panel, "Iterations", "6"),
        ),
    )
    demo.click_button(panel, "Run", wait=1.3)
    panel.get_by_text("Final Parameters").first.wait_for(state="visible", timeout=12_000)
    show("gradient_function", panel, "Function")
    show("gradient_history", panel, "History")
    show("gradient_final", panel, "Final Parameters")

    panel = demo.tab("Decision Tree")
    r.explain(cues["tree_class_tab"])
    r.explain_during(
        cues["tree_class_type"],
        lambda: (
            demo.choose(panel, "Mode", "Classification impurity"),
            demo.fill_textbox(panel, "Class counts by node", "8 2\n3 7"),
            demo.choose(panel, "Impurity measure", "all"),
        ),
    )
    demo.click_button(panel, "Calculate", wait=1.2)
    panel.get_by_text("Class Counts").first.wait_for(state="visible", timeout=12_000)
    show("tree_counts", panel, "Class Counts")
    show("tree_gini", panel, "Gini")
    show("tree_entropy", panel, "Entropy")
    r.explain_during(
        cues["tree_reg_select"],
        lambda: demo.choose(panel, "Mode", "Regression MSE / split"),
    )
    r.explain_during(
        cues["tree_reg_type"],
        lambda: (
            demo.fill_textbox(panel, "x values", "0.5\n1.1\n1.8\n2.6\n3.4\n4.0"),
            demo.fill_textbox(panel, "y values", "1.0\n1.4\n2.2\n5.8\n6.3\n7.1"),
            demo.fill_textbox(panel, "Decision threshold", "2.5"),
        ),
    )
    demo.click_button(panel, "Calculate", wait=1.3)
    panel.get_by_text("Best Split").first.wait_for(state="visible", timeout=12_000)
    show("tree_threshold", panel, "Threshold Summary")
    show("tree_candidates", panel, "Candidate Splits")
    show("tree_best", panel, "Best Split")

    panel = demo.tab("K-means")
    r.explain(cues["kmeans_tab"])
    r.explain_during(
        cues["kmeans_type"],
        lambda: (
            demo.fill_textbox(
                panel,
                "Points X",
                "1 1\n1.5 2\n3 4\n5 7\n3.5 5\n4.5 5\n3.5 4.5",
            ),
            demo.fill_textbox(panel, "Initial centroids", "1 1\n5 7"),
            demo.fill_textbox(panel, "Max iterations", "5"),
        ),
    )
    demo.click_button(panel, "Run", wait=1.3)
    panel.get_by_text("Iteration").first.wait_for(state="visible", timeout=12_000)
    show("kmeans_summary", panel, "Result")
    show("kmeans_initial", panel, "Initial clusters")
    show("kmeans_iteration", panel, "Iteration 1")

    panel = demo.tab("Cache")
    r.explain(cues["cache_tab"])
    r.explain_during(
        cues["cache_show"],
        lambda: demo.highlight_text(panel, "Max allowed cache"),
    )
    r.explain(cues["cache_copy"])
    demo.r.element(panel.get_by_text("Plain text", exact=True).first).click()
    demo.wait(0.4)
    copied = demo.r.element(panel.locator("code").first).copy_text()
    panel = demo.tab("Array")
    r.explain_during(
        cues["cache_reuse"],
        lambda: demo.paste_into_textbox(panel, "Array", copied),
    )
    demo.click_button(panel, "Inspect", wait=1.2)
    panel.get_by_text("Parsed Matrix").first.wait_for(state="visible", timeout=12_000)
    demo.highlight_text(panel, "Parsed Matrix")

    r.explain(cues["finish_1"])
    r.explain(cues["finish_2"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record the EE2211 Solver WebUI demo.")
    parser.add_argument("--smoke", action="store_true", help="Run a fast, no-TTS smoke recording.")
    parser.add_argument("--no-tts", action="store_true", help="Record subtitles only, without narration audio.")
    parser.add_argument("--port", type=int, default=None, help="Port for the temporary Streamlit server.")
    return parser.parse_args()


def reset_webui_state() -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    empty_widgets = {
        key: ""
        for key in (
            "array_inspect_raw",
            "solve_x_raw",
            "solve_y_raw",
            "reg_x_raw",
            "reg_y_raw",
            "reg_x_test_raw",
            "reg_y_test_raw",
            "pearson_x_raw",
            "pearson_y_raw",
            "gradient_cost_expression",
            "gradient_initial_values",
            "gradient_tolerance",
            "tree_counts_raw",
            "tree_x_raw",
            "tree_y_raw",
            "tree_threshold",
            "kmeans_x_raw",
            "kmeans_centroids_raw",
            "kmeans_max_iterations",
        )
    }
    payload = {
        "version": 1,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "widgets": {
            "max_allowed_cache": 50,
            **empty_widgets,
        },
        "matrix_cache": [],
    }
    STATE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def verify_subtitle_support(ffmpeg: str) -> None:
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-filters"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"Could not inspect ffmpeg filters:\n{result.stderr}")
    if " subtitles " not in result.stdout and "\n .. subtitles" not in result.stdout:
        raise RuntimeError("The selected ffmpeg build does not expose the subtitles filter.")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def start_streamlit(port: int) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env.update(
        {
            "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
            "STREAMLIT_SERVER_HEADLESS": "true",
        }
    )
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "webui.py",
        "--server.headless",
        "true",
        "--server.port",
        str(port),
        "--server.fileWatcherType",
        "none",
        "--browser.gatherUsageStats",
        "false",
    ]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    wait_for_server(process, port)
    return process


def wait_for_server(process: subprocess.Popen[str], port: int) -> None:
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 45
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = ""
            if process.stdout is not None:
                output = process.stdout.read()
            raise RuntimeError(f"Streamlit exited early:\n{output}")
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if response.status < 500:
                    return
        except Exception as exc:
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"Streamlit did not start on {url}: {last_error}")


def stop_streamlit(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


if __name__ == "__main__":
    main()
