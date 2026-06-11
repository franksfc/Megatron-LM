#!/usr/bin/env python3
"""Monitor Orin/SSM training logs without touching the live run.

The older Orin trainer wrote JSONL records. The MindSpeed/Megatron path writes
Megatron text logs and logs W&B metrics inside the trainer process. This tool
supports both formats so the same health check can be used during migration.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


TRAIN_RE = re.compile(
    r"iteration\s+(?P<step>\d+)/\s*(?P<total>\d+).*?"
    r"consumed samples:\s*(?P<samples>\d+).*?"
    r"elapsed time per iteration \(ms\):\s*(?P<ms>[-+0-9.Ee]+).*?"
    r"learning rate:\s*(?P<lr>[-+0-9.Ee]+).*?"
    r"global batch size:\s*(?P<gbs>\d+).*?"
    r"lm loss:\s*(?P<loss>[-+0-9.Ee]+).*?"
    r"grad norm:\s*(?P<grad>[-+0-9.Ee]+).*?"
    r"number of nan iterations:\s*(?P<nan>\d+)"
)
EVAL_LOSS_RE = re.compile(
    r"validation loss at iteration\s+(?P<step>\d+)(?:\s+on validation set)?.*?"
    r"lm loss value:\s*(?P<loss>[-+0-9.Ee]+)"
)
@dataclass
class TrainRecord:
    source: str
    step: int
    total: int
    consumed_samples: int
    elapsed_ms: float
    learning_rate: float
    global_batch_size: int
    loss: float
    grad_norm: float
    nan_iterations: int


@dataclass
class EvalRecord:
    source: str
    step: int
    loss: float


def load_last_records(path: Path, max_lines: int = 200) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()[-max_lines:]
    records = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def check_jsonl_records(
    records: list[dict[str, Any]],
    target_eval_loss: float | None,
    tolerance: float,
    log_file: Path,
    max_stale_seconds: float | None,
) -> tuple[bool, dict[str, Any]]:
    if not records:
        return False, {"reason": "no records yet"}
    if max_stale_seconds is not None and log_file.is_file():
        age_seconds = time.time() - log_file.stat().st_mtime
        if age_seconds > max_stale_seconds:
            return False, {"reason": "stale_jsonl", "age_seconds": age_seconds}
    latest = records[-1]
    if target_eval_loss is not None:
        eval_records = [record for record in records if "eval_loss" in record]
        if eval_records:
            diff = abs(float(eval_records[-1]["eval_loss"]) - target_eval_loss)
            if diff > tolerance:
                return False, {
                    "reason": "eval_loss_out_of_tolerance",
                    "diff": diff,
                    "tolerance": tolerance,
                }
    for key in ("loss", "eval_loss"):
        if key in latest and math.isnan(float(latest[key])):
            return False, {"reason": f"{key}_is_nan", "latest": latest}
    return True, {"latest": latest, "records": len(records)}


def parse_rank_log(path: Path) -> tuple[list[TrainRecord], list[EvalRecord]]:
    train_records: list[TrainRecord] = []
    eval_records: list[EvalRecord] = []
    if not path.is_file():
        return train_records, eval_records
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            train_match = TRAIN_RE.search(line)
            if train_match:
                train_records.append(
                    TrainRecord(
                        source=path.name,
                        step=int(train_match.group("step")),
                        total=int(train_match.group("total")),
                        consumed_samples=int(train_match.group("samples")),
                        elapsed_ms=float(train_match.group("ms")),
                        learning_rate=float(train_match.group("lr")),
                        global_batch_size=int(train_match.group("gbs")),
                        loss=float(train_match.group("loss")),
                        grad_norm=float(train_match.group("grad")),
                        nan_iterations=int(train_match.group("nan")),
                    )
                )
                continue
            eval_match = EVAL_LOSS_RE.search(line)
            if eval_match:
                eval_records.append(
                    EvalRecord(
                        source=path.name,
                        step=int(eval_match.group("step")),
                        loss=float(eval_match.group("loss")),
                    )
                )
    return train_records, eval_records


def find_rank_logs(log_dir: Path) -> list[Path]:
    return sorted(path for path in log_dir.glob("rank*.log") if ".launcher." not in path.name)


def summarize_window(records: list[TrainRecord], start: int, end: int) -> dict[str, Any] | None:
    values = [record.elapsed_ms for record in records if start <= record.step <= end]
    if not values:
        return None
    return {
        "start": start,
        "end": end,
        "n": len(values),
        "avg_ms": sum(values) / len(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def summarize_last(records: list[TrainRecord], count: int) -> dict[str, Any] | None:
    values = [record.elapsed_ms for record in records[-count:]]
    if not values:
        return None
    return {
        "last_n": len(values),
        "avg_ms": sum(values) / len(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def list_checkpoints(path: Path | None) -> list[str]:
    if path is None or not path.is_dir():
        return []
    names = []
    for child in sorted(path.iterdir()):
        if child.is_dir() and (child.name.startswith("iter_") or child.name.startswith("global_step")):
            names.append(child.name)
    return names


def query_wandb_summary(entity: str, project: str, run_id: str, base_url: str | None) -> dict[str, Any]:
    try:
        import wandb
    except Exception as exc:  # pragma: no cover - depends on runtime env
        return {"ok": False, "error": f"import wandb failed: {exc}"}
    try:
        api = wandb.Api(overrides={"base_url": base_url} if base_url else None)
        run = api.run(f"{entity}/{project}/{run_id}")
        keys = [
            "train/global_step",
            "train/loss",
            "train/learning_rate",
            "train/grad_norm",
            "train/epoch",
            "eval/loss",
        ]
        return {
            "ok": True,
            "state": run.state,
            "url": run.url,
            "summary": {key: run.summary.get(key) for key in keys if key in run.summary},
        }
    except Exception as exc:  # pragma: no cover - depends on network/API
        return {"ok": False, "error": str(exc)}


def process_counts(hosts: list[str], pattern: str) -> dict[str, Any]:
    if not hosts or not pattern:
        return {}
    counts: dict[str, Any] = {}
    for host in hosts:
        host = host.strip()
        if not host:
            continue
        if host in ("localhost", "127.0.0.1"):
            cmd = f"pgrep -af {pattern!r} | wc -l"
        else:
            cmd = f"ssh -o BatchMode=yes -o ConnectTimeout=5 {host!r} \"pgrep -af {pattern!r} | wc -l\""
        try:
            output = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
            counts[host] = int(output.strip().splitlines()[-1])
        except Exception as exc:
            counts[host] = {"error": str(exc)}
    return counts


def monitor_text_logs(args: argparse.Namespace) -> tuple[bool, dict[str, Any]]:
    rank_logs = find_rank_logs(args.log_dir)
    train_records: list[TrainRecord] = []
    eval_records: list[EvalRecord] = []
    latest_mtime = 0.0
    for path in rank_logs:
        latest_mtime = max(latest_mtime, path.stat().st_mtime)
        train_part, eval_part = parse_rank_log(path)
        train_records.extend(train_part)
        eval_records.extend(eval_part)
    train_records.sort(key=lambda record: (record.step, record.source))
    eval_records.sort(key=lambda record: (record.step, record.source))

    status: dict[str, Any] = {
        "mode": "megatron_text",
        "log_dir": str(args.log_dir),
        "rank_logs": [path.name for path in rank_logs],
        "train_records": len(train_records),
        "eval_records": len(eval_records),
        "checkpoints": list_checkpoints(args.checkpoint_dir),
    }
    if latest_mtime:
        status["log_age_seconds"] = time.time() - latest_mtime
    if args.wandb_run:
        status["wandb"] = query_wandb_summary(
            args.wandb_entity,
            args.wandb_project,
            args.wandb_run,
            args.wandb_base_url,
        )
    if args.hosts:
        status["process_counts"] = process_counts(args.hosts.split(","), args.process_pattern or args.wandb_run)

    ok = True
    reasons: list[str] = []
    if not train_records:
        ok = False
        reasons.append("no train records yet")
    if args.max_stale_seconds is not None and latest_mtime:
        age_seconds = time.time() - latest_mtime
        if age_seconds > args.max_stale_seconds:
            ok = False
            reasons.append(f"log stale for {age_seconds:.1f}s")
    if train_records:
        latest = train_records[-1]
        status["latest_train"] = asdict(latest)
        status["last_speed"] = summarize_last(train_records, args.last_n)
        windows = []
        for item in args.speed_windows.split(","):
            item = item.strip()
            if not item:
                continue
            start_text, end_text = item.split("-", 1)
            summary = summarize_window(train_records, int(start_text), int(end_text))
            if summary:
                windows.append(summary)
        status["speed_windows"] = windows
        if math.isnan(latest.loss):
            ok = False
            reasons.append("latest loss is NaN")
        if latest.nan_iterations:
            ok = False
            reasons.append(f"nan iterations reported: {latest.nan_iterations}")
    if eval_records:
        status["latest_eval"] = asdict(eval_records[-1])
        if math.isnan(eval_records[-1].loss):
            ok = False
            reasons.append("latest eval loss is NaN")
    if args.expect_step is not None and train_records and train_records[-1].step < args.expect_step:
        ok = False
        reasons.append(f"latest step {train_records[-1].step} < expected {args.expect_step}")
    status["ok"] = ok
    status["reasons"] = reasons
    return ok, status


def monitor_jsonl(args: argparse.Namespace) -> tuple[bool, dict[str, Any]]:
    target_eval_loss = (
        None if str(args.target_eval_loss).lower() == "none" else float(args.target_eval_loss)
    )
    records = load_last_records(args.log_file)
    ok, details = check_jsonl_records(
        records,
        target_eval_loss,
        args.loss_tolerance,
        args.log_file,
        args.max_stale_seconds,
    )
    return ok, {"mode": "jsonl", "log_file": str(args.log_file), "ok": ok, **details}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument(
        "--target-eval-loss",
        default="2.213824510574341",
        help="Expected eval loss for JSONL mode, or 'none' to only check health.",
    )
    parser.add_argument("--loss-tolerance", type=float, default=0.02)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--max-stale-seconds", type=float, default=1800.0)
    parser.add_argument("--last-n", type=int, default=50)
    parser.add_argument("--speed-windows", default="10-30,50-100,100-200,500-1000")
    parser.add_argument("--expect-step", type=int, default=None)
    parser.add_argument("--wandb-run", default="")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "Ponder2-adaptive"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", "franksfc-lumia-group"))
    parser.add_argument("--wandb-base-url", default=os.environ.get("WANDB_BASE_URL", "https://api.bandw.top"))
    parser.add_argument("--hosts", default="")
    parser.add_argument("--process-pattern", default="")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if args.log_dir is None and args.log_file is None:
        parser.error("provide --log-dir for Megatron logs or --log-file for JSONL logs")

    while True:
        if args.log_dir is not None:
            ok, status = monitor_text_logs(args)
        else:
            ok, status = monitor_jsonl(args)
        print(json.dumps(status, sort_keys=True), flush=True)
        if args.once:
            return 0 if ok else 1
        if not ok and status.get("train_records", 0):
            return 1
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
