"""A real B200 for the design sessions' GPU tools, on Modal (pay per second, no idle machine).

The image follows SOL-ExecBench's docker/Dockerfile (CUDA 13.1.1 + cuDNN, CUTLASS v4.4.1, the uv-locked Python
environment, fbtriton 3.7.1, cutlass-dsl-libs-cu13 4.4.2), so candidates run on the portal's software stack.
Each call handles one compile/test request with loop/toolserver.py --once. Containers have no network access and
stay warm for 30 seconds between calls. Clocks are not locked (the portal locks SM 1500 MHz / DRAM 3996 MHz).

Auth: the profile saved by `modal setup` (~/.modal.toml), or MODAL_TOKEN_ID / MODAL_TOKEN_SECRET from the
environment or the repo's .env (never printed).

  .venv/bin/python loop/b200_modal.py info                 # GPU name, clocks, whether clocks can be locked
  .venv/bin/python loop/calibrate_b200.py                  # time portal-measured kernels, compare with the portal
"""

import json
import os
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
SOLX = ROOT / "SOL-ExecBench"
DATA = SOLX / "data" / "benchmark"           # every problem (7.7 MB), mounted at /problems/<level>/<name>
GPU = "B200"                         # exactly B200 (sm_100); B300/GB300 are a different chip

RUNTIME_ENV = {
    # Host C++ compiles (e.g. a solution's binding.cpp) go through ccache, kept on a Modal volume, so an unchanged
    # PyTorch binding rebuilds in about a second. nvcc keeps the real g++ as its host compiler (ccache cannot
    # preprocess torch-extension nvcc commands).
    "PATH": "/usr/lib/ccache:/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "NVCC_PREPEND_FLAGS": "-ccbin=/usr/bin/g++",
    "CCACHE_DIR": "/cache/ccache",
    "CCACHE_MAXSIZE": "5G",
    "CCACHE_BASEDIR": "/tmp",
    "CCACHE_NOHASHDIR": "1",
    "CCACHE_SLOPPINESS": "include_file_mtime,include_file_ctime,time_macros",
    "CUDA_HOME": "/usr/local/cuda",
    "CUDACXX": "/usr/local/cuda/bin/nvcc",
    "CUTLASS_DIR": "/usr/local/cutlass",
    "CPLUS_INCLUDE_PATH": "/usr/local/cutlass/include:/venv/lib/python3.12/site-packages/include",
    "LD_LIBRARY_PATH": "/venv/lib/python3.12/site-packages/nvidia/cu13/lib",
    "PYTHONPATH": "/sol-execbench/src",
    "TRITON_CACHE_DIR": "/cache/triton",
}
UV_ENV = {"UV_LINK_MODE": "copy", "UV_COMPILE_BYTECODE": "1", "UV_PYTHON_DOWNLOADS": "never",
          "UV_PROJECT_ENVIRONMENT": "/venv", "UV_HTTP_TIMEOUT": "600", "UV_PYTHON": "/usr/bin/python3.12"}


def load_token():
    """Put MODAL_TOKEN_ID / MODAL_TOKEN_SECRET into the environment from .env if they are not set already."""
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                name, value = line.split("=", 1)
                name = name.strip().removeprefix("export ").strip().upper()
                if name in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET") and not os.environ.get(name):
                    os.environ[name] = value.strip().strip('"').strip("'")
    if not (os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET")) and \
            not (Path.home() / ".modal.toml").exists():
        raise SystemExit("No Modal token: run `.venv/bin/python -m modal setup`, or put MODAL_TOKEN_ID / "
                         "MODAL_TOKEN_SECRET in .env")


def write_trace(path, res):
    """A Modal test result -> a harness-style JSONL trace (read by archive.read_traces), plus a .log on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    import problem
    var_axes = problem.current().var_axes
    for w in res.get("workloads") or []:
        axes = w.get("axes") or dict(zip(var_axes, (int(x) for x in w["workload"].split(","))))
        lines.append(json.dumps({"workload": {"axes": axes}, "evaluation": {
            "status": w["status"], "log": w.get("log", ""),
            "performance": {"latency_ms": w["latency_us"] / 1e3 if w.get("latency_us") else None}}}))
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
    wl = res.get("workloads") or []
    if not wl or any(w["status"] != "PASSED" for w in wl):
        bad = "\n".join(f"{w['workload']}: {w['status']}\n{w.get('log', '')}" for w in wl if w["status"] != "PASSED")
        path.with_suffix(".log").write_text(f"{bad}\n{res.get('error') or ''}\n{res.get('console_tail') or ''}")


image = (
    modal.Image.from_registry("nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04", add_python="3.12")
    .apt_install("curl", "git", "wget", "python3", "python3-dev", "build-essential", "libblas-dev", "liblapack-dev",
                 "ccache", "cmake", "ninja-build")
    .run_commands("git clone --depth 1 -b v4.4.1 https://github.com/NVIDIA/cutlass.git /usr/local/cutlass")
    .pip_install("uv")
    .add_local_file(SOLX / "pyproject.toml", "/sol-execbench/pyproject.toml", copy=True)
    .add_local_file(SOLX / "uv.lock", "/sol-execbench/uv.lock", copy=True)
    .add_local_file(SOLX / "README.md", "/sol-execbench/README.md", copy=True)
    .run_commands("cd /sol-execbench && uv sync --frozen --no-install-project --all-groups", env=UV_ENV)
    .add_local_dir(SOLX / "src", "/sol-execbench/src", copy=True)
    .run_commands(
        "cd /sol-execbench && uv sync --frozen --no-editable --all-groups"
        " && uv pip uninstall --python /venv triton"
        " && uv pip install --python /venv --no-deps fbtriton==3.7.1"
        " && uv pip install --python /venv --no-deps --force-reinstall nvidia-cutlass-dsl-libs-cu13==4.4.2",
        env=UV_ENV)
    # fbtriton 3.7.1 ships triton/runtime/launch.h but looks for it under backends/nvidia (same fix as on CSF3).
    .run_commands("ln -sf ../../runtime/launch.h /venv/lib/python3.12/site-packages/triton/backends/nvidia/launch.h")
    # Skip timing samples whose CUPTI window is empty (timestamp skew on Modal); see loop/patch_harness_timing.py.
    .add_local_file(ROOT / "loop" / "patch_harness_timing.py", "/solx/patch_harness_timing.py", copy=True)
    .run_commands("python3 /solx/patch_harness_timing.py /sol-execbench/src/sol_execbench/core/bench/timing.py "
                  "/venv/lib/python3.12/site-packages/sol_execbench/core/bench/timing.py")
    .env({"SOLX_IMAGE": "sol-execbench-v1.1"})
    .add_local_dir(DATA, "/problems")
    .add_local_file(ROOT / "loop" / "toolserver.py", "/solx/loop/toolserver.py")
    .add_local_file(ROOT / "loop" / "static_any.py", "/solx/loop/static_any.py")
    .add_local_file(ROOT / "loop" / "b200probe.py", "/solx/loop/b200probe.py")
)

app = modal.App("solx-b200", image=image)
DEFAULT_PROBLEM = "L1/038_flux_multi_head_rmsnorm_qk"


def problem_ref():
    """'<level>/<name>' of the current problem, for requests (local side only)."""
    import problem
    p = problem.current()
    return f"{p.level}/{p.name}"
cache = modal.Volume.from_name("solx-b200-cache", create_if_missing=True)
work = modal.Volume.from_name("solx-b200-work", create_if_missing=True)   # per-session probe workspaces


# scaledown_window: keep a finished container warm this long for the next call. LLM turns usually take longer than
# this, so a short window bills less idle time than it loses to cold starts.
@app.function(gpu=GPU, cpu=8.0, memory=32768, timeout=1800, scaledown_window=30, max_containers=1, block_network=True,
              volumes={"/cache": cache, "/work": work})
def handle(req: dict) -> dict:
    """One request on the B200: compile / test (as loop/toolserver.py), or probe (run a short Python script)."""
    import subprocess
    import tempfile
    import time
    d = Path(tempfile.mkdtemp())
    if req.get("kind") == "probe":
        ws = str(req.get("workspace") or "").strip("/")
        if ws and ".." not in ws:                   # the session's files persist between probes (and containers)
            work.reload()
            d = Path("/work") / ws
            d.mkdir(parents=True, exist_ok=True)
        (d / "probe.py").write_text(req["script"])
        env = {**os.environ, **RUNTIME_ENV, "PYTHONPATH": "/sol-execbench/src:/solx/loop"}
        t0 = time.time()
        try:
            p = subprocess.run(["/venv/bin/python", "probe.py"], cwd=d, env=env, capture_output=True, text=True,
                               timeout=int(req.get("timeout", 120)))
            out, err, rc = p.stdout, p.stderr, p.returncode
        except subprocess.TimeoutExpired as e:
            out, err, rc = (e.stdout or b"").decode(errors="replace"), "timed out", -1
        if ws and ".." not in ws:
            work.commit()
        return dict(id=req.get("id"), kind="probe", gpu="NVIDIA B200", returncode=rc, seconds=round(time.time() - t0, 1),
                    stdout=out[-6000:], stderr=err[-3000:])
    (d / "req.json").write_text(json.dumps(req))
    p = subprocess.run(["/venv/bin/python", "/solx/loop/toolserver.py", "--once", str(d / "req.json"),
                        "--out", str(d / "res.json"), "--problem", f"/problems/{req.get('problem', DEFAULT_PROBLEM)}"],
                       env={**os.environ, **RUNTIME_ENV}, capture_output=True, text=True, timeout=1700)
    cache.commit()                                  # keep new Triton cache entries for the next container
    if (d / "res.json").exists():
        return json.loads((d / "res.json").read_text())
    return dict(id=req.get("id"), kind=req.get("kind"), error=(p.stdout[-2000:] + "\n" + p.stderr[-3000:]).strip())


@app.function(gpu=GPU, timeout=600, max_containers=1, block_network=True)
def info() -> dict:
    """GPU identity and clocks, and whether this container may lock clocks like the portal does."""
    import subprocess
    q = lambda *a: subprocess.run(["nvidia-smi", *a], capture_output=True, text=True)
    out = dict(query=q("--query-gpu=name,compute_cap,driver_version,clocks.sm,clocks.mem,clocks.max.sm,"
                       "clocks.max.mem,power.limit", "--format=csv").stdout)
    lock = q("-lgc", "1500,1500")
    out["lock_sm"] = (lock.stdout + lock.stderr).strip()[-300:]
    if lock.returncode == 0:
        q("-rgc")
    env = {**os.environ, **RUNTIME_ENV}
    out["torch"] = subprocess.run(["/venv/bin/python", "-c", "import torch, triton; print(torch.__version__, "
                                   "triton.__version__, torch.cuda.get_device_name(0), "
                                   "torch.cuda.get_device_capability(0))"],
                                  env=env, capture_output=True, text=True).stdout.strip()
    return out


if __name__ == "__main__":
    load_token()
    if sys.argv[1:] == ["info"]:
        with modal.enable_output(), app.run():
            print(json.dumps(info.remote(), indent=1))
