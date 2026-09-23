"""Run the official ForwardGNN core node-classification experiments on CPU.

Uses upstream training, model selection, seeds and intermediate SF layer outputs.
The compatibility shim only makes unused legacy cached operators optional.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import runpy
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "third_party/forwardgnn"
SOURCE = UPSTREAM / "src"
RESULTS = ROOT / "results/paper_reproduction"


def worker():
    # Multiple independent configurations may run concurrently. Serialize the
    # same configuration so a resumed driver cannot overwrite an active run.
    import fcntl
    def flag(name):
        return sys.argv[sys.argv.index(name) + 1]
    lock_root = RESULTS / "locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    key = "_".join(flag(name) for name in ("--exp-setting", "--dataset", "--model", "--num-layers"))
    with (lock_root / f"{key}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run_worker()


def run_worker():
    kind = sys.argv[2]
    sys.path.insert(0, str(SOURCE))
    import torch
    torch.set_num_threads(1)
    import settings
    settings.RESULTS_ROOT = RESULTS / "official"
    script = SOURCE / ("train_forward.py" if kind == "sf" else "train_backprop.py")
    sys.argv = [str(script)] + sys.argv[3:]
    os.chdir(SOURCE)
    runpy.run_path(str(script), run_name="__main__")


def prepare_splits(datasets):
    import torch
    hashes = {}
    split_root = ROOT / "third_party/forwardgnn-datasplits"
    for dataset in datasets:
        # Locate author-published tensors, not a new random split.
        found = sorted(split_root.rglob(f"{dataset}/node-5splits/*.pt"))
        if len(found) != 15:
            raise RuntimeError(f"Expected 15 official node split tensors for {dataset}, got {len(found)}")
        for path in found:
            indices = torch.load(path, weights_only=True)
            if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
                raise ValueError(f"Unexpected split tensor: {path}")
            target = UPSTREAM / "datasplits" / dataset / "node-5splits" / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            hashes[str(target.relative_to(UPSTREAM))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["CitationFull-CiteSeer", "CitationFull-Cora_ML"])
    parser.add_argument("--backbones", nargs="+", choices=("GCN", "SAGE", "GAT"), default=["GCN"])
    parser.add_argument("--max-layers", type=int, default=4)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--setting", default="core_nodeclass")
    args = parser.parse_args()
    if not 1 <= args.runs <= 5 or args.max_layers < 1:
        parser.error("runs must be 1..5 and max-layers positive")
    RESULTS.mkdir(parents=True, exist_ok=True)
    log_root = RESULTS / "logs" / args.setting
    log_root.mkdir(parents=True, exist_ok=True)
    split_hashes = prepare_splits(args.datasets)
    import torch
    import torch_geometric
    manifest = {
        "configuration": vars(args), "platform": platform.platform(),
        "python": sys.version, "torch": torch.__version__, "pyg": torch_geometric.__version__,
        "device": "cpu", "threads": 1,
        "upstream_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True).strip(),
        "split_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT / "third_party/forwardgnn-datasplits", text=True).strip(),
        "split_sha256": split_hashes,
        "compatibility": "Optional import shim for UNUSED legacy cached operators; standard PyG operators unchanged.",
        "limitations": "Newer PyTorch/PyG and CPU; H100 GPU memory/time not reproduced.",
        "compatibility_file_sha256": {
            str(path.relative_to(UPSTREAM)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (SOURCE / "gnn/__init__.py", SOURCE / "gnn/gnn_conv.py", SOURCE / "gnn/optional_cached.py")
        },
    }
    manifest_path = RESULTS / f"manifest_{args.setting}.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["configuration"] != vars(args):
            raise ValueError("This setting already has a different configuration; choose a new --setting")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    jobs = []
    for dataset in args.datasets:
        for backbone in args.backbones:
            # Greedy SF emits results for every completed layer in one max-depth fit.
            jobs.append((dataset, backbone, "sf", args.max_layers))
            jobs.extend((dataset, backbone, "bp", depth) for depth in range(1, args.max_layers + 1))
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    for number, (dataset, backbone, method, depth) in enumerate(jobs, 1):
        model = f"GNN_SingleForward-{backbone}" if method == "sf" else f"GNN-{backbone}"
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", method,
                   "--task", "node-class", "--model", model, "--dataset", dataset,
                   "--num-layers", str(depth), "--num-hidden", "128", "--num-runs", str(args.runs),
                   "--seed", "100", "--epochs", str(args.epochs), "--lr", "0.001",
                   "--val-every", "2", "--patience", "100", "--gpu", "-1", "--exp-setting", args.setting]
        if method == "sf":
            command += ["--append-label", "none", "--temperature", "1", "--aug-edge-direction", "bidirection"]
        log = log_root / f"{dataset}_{method}_{backbone}_L{depth}.log"
        print(f"[{number}/{len(jobs)}] {dataset} {method} {backbone} L={depth}; log={log}", flush=True)
        started = time.perf_counter()
        with log.open("a") as handle:
            handle.write("\nCOMMAND " + json.dumps(command) + "\n")
            result = subprocess.run(command, cwd=SOURCE, env=env, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Experiment failed: inspect {log}")
        print(f"completed in {time.perf_counter()-started:.1f}s", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker()
    else:
        main()
