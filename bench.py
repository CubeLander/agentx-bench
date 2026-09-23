"""Thin, fail-closed launcher. Scheduling and metrics belong to official AIPerf."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.parse import urlsplit
import uuid

from prepare import CACHE, PROTOCOL, ROOT, hf_environment, verify_dataset


def read_target(path: Path) -> dict:
    target = json.loads(path.read_text())
    required = {"url", "model", "model_revision", "tokenizer", "tokenizer_revision", "engine", "hardware",
                "max_model_len", "host_kv_budget_gib", "speculative_decoding"}
    missing = required - target.keys()
    if missing:
        raise ValueError(f"Missing target fields: {sorted(missing)}")
    url = urlsplit(target["url"])
    if (url.scheme not in {"http", "https"} or not url.hostname or url.username
            or url.password or url.query or url.fragment or url.path not in {"", "/"}):
        raise ValueError("url must be a credential-free http(s) origin, without /v1")
    if target["max_model_len"] < 256000:
        raise ValueError("Official 256k corpus requires >=256000 context; no filtering")
    if target["host_kv_budget_gib"] < 0:
        raise ValueError("host_kv_budget_gib cannot be negative")
    spec = target["speculative_decoding"]
    if spec.get("enabled") is not False:
        if spec.get("enabled") is not True or not all(
            key in spec for key in ("forced_acceptance_length", "speed_bench_evidence",
                                    "server_configuration")
        ):
            raise ValueError("Speculation requires forced acceptance and SPEED-Bench evidence")
    tokenizer = Path(target["tokenizer"]).expanduser()
    if not tokenizer.is_absolute():
        tokenizer = path.resolve().parent / tokenizer
    target["tokenizer"] = str(tokenizer.resolve())
    # Only an explicit target origin is used. This launcher never starts a model,
    # changes server memory limits, inspects other devices, or scans endpoints.
    return target


def command(target: dict, profile: str, concurrency: int, output: Path) -> list[str]:
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    return [
        str(Path(sys.executable).with_name("aiperf")), "profile",
        "--scenario", PROTOCOL["scenario"],
        "--public-dataset", PROTOCOL["dataset_alias"],
        "--url", target["url"], "--model", target["model"],
        "--tokenizer", target["tokenizer"],
        "--endpoint-type", "chat", "--streaming", "--use-server-token-count",
        "--extra-inputs", "ignore_eos:true",
        "--cache-bust", "first_turn_prefix",
        "--system-idle-gap-cap-seconds", "10",
        "--trajectory-start-min-ratio", str(PROTOCOL["trajectory_start_min_ratio"]),
        "--trajectory-start-max-ratio", str(PROTOCOL["trajectory_start_max_ratio"]),
        "--warmup-requests-per-lane", str(PROTOCOL["warmup_requests_per_lane"]),
        "--benchmark-duration", str(PROTOCOL["profiles"][profile]["measurement_seconds"]),
        "--benchmark-grace-period", "30",
        "--random-seed", str(PROTOCOL["seed"]),
        "--concurrency", str(concurrency), "--ui", "simple",
        "--artifact-dir", str(output),
    ]


def verify_harness() -> None:
    dist = importlib.metadata.distribution("aiperf")
    origin = json.loads(dist.read_text("direct_url.json") or "{}")
    if origin.get("vcs_info", {}).get("commit_id") != PROTOCOL["harness_commit"]:
        raise RuntimeError("Installed aiperf is not the pinned AgentX harness; use uv sync --locked")


def summarize(output: Path, returncode: int) -> dict:
    files = sorted(output.rglob("profile_export_aiperf.json"))
    reports = []
    for path in files:
        data = json.loads(path.read_text())
        metadata = data.get("metadata", {})
        reports.append({
            "path": str(path.relative_to(output)),
            "submission_valid": metadata.get("submission_valid"),
            "submission_invalid_reasons": metadata.get("submission_invalid_reasons", []),
        })
    valid = returncode == 0 and len(reports) == 1 and reports[0]["submission_valid"] is True
    return {"status": "completed" if valid else "failed_or_invalid",
            "returncode": returncode, "upstream_reports": reports,
            "note": "Upstream validity is not independent hardware/speculation compliance certification."}


def run(target: dict, profile: str, concurrency: int) -> int:
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if "REPLACE_WITH_" in json.dumps(target):
        raise ValueError("Replace example target values before running")
    verify_harness()
    if not Path(target["tokenizer"]).is_dir():
        raise ValueError("tokenizer must be a prepared local directory (offline replay)")
    os.environ.update(hf_environment(offline=True))
    receipt = verify_dataset()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "artifacts" / f"{stamp}-{profile}-c{concurrency}-{uuid.uuid4().hex[:8]}"
    output.mkdir(parents=True)
    argv = command(target, profile, concurrency, output / "aiperf")
    env = dict(os.environ)
    env.setdefault("AIPERF_DATASET_CONFIGURATION_TIMEOUT", "1800")
    env.setdefault("AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT", "1800")
    # Keep reconstruction caches within this corpus revision too.
    env.setdefault("XDG_CACHE_HOME", str(CACHE / "xdg"))
    manifest = {
        "protocol": PROTOCOL, "profile": profile, "target": target,
        "concurrent_agent_clients": concurrency, "dataset_receipt": receipt,
        "command": argv, "started_utc": stamp, "status": "running",
        "one_hour_measurement": profile == "formal",
        "timing": "preparation + warmup + measurement + up to 30s drain + export",
        "wrapper_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "wrapper_tracked_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip()),
    }
    manifest_path = output / "run.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Artifacts: {output}\nFull progress: {output / 'harness.log'}", flush=True)
    start = time.monotonic()
    try:
        with (output / "harness.log").open("w") as log:
            returncode = execute_owned(argv, env, log)
        summary = summarize(output / "aiperf", returncode)
    except KeyboardInterrupt:
        summary = {"status": "cancelled", "returncode": 130}
    except Exception as exc:
        summary = {"status": "failed", "returncode": 1, "error_type": type(exc).__name__}
    summary["total_harness_wall_seconds"] = time.monotonic() - start
    manifest.update(summary)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "completed" else (summary["returncode"] or 1)


def execute_owned(argv, env, log) -> int:
    # Own one process group. Cancellation must not leave AIPerf workers issuing
    # requests, nor send signals to the caller's group or any model process.
    with subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT,
                          start_new_session=True) as process:
        try:
            return process.wait()
        except KeyboardInterrupt:
            for sig, grace in ((signal.SIGINT, 30), (signal.SIGTERM, 10), (signal.SIGKILL, 5)):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    process.wait(timeout=grace)
                    break
                except subprocess.TimeoutExpired:
                    continue
            raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "run"])
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--profile", choices=PROTOCOL["profiles"], default="smoke")
    parser.add_argument("--concurrency", type=int, required=True,
                        help="Concurrent agent session trees, not HTTP requests")
    args = parser.parse_args()
    target = read_target(args.target)
    if args.action == "plan":
        argv = command(target, args.profile, args.concurrency, ROOT / "artifacts" / "PLAN")
        print(json.dumps({"protocol": PROTOCOL, "target": target, "command": argv}, indent=2))
        return 0
    return run(target, args.profile, args.concurrency)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"agentx-bench: {exc}", file=sys.stderr)
        sys.exit(2)
