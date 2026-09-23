"""Stage the immutable official corpus in an isolated, offline replay cache."""

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PROTOCOL = json.loads((ROOT / "protocol.json").read_text())
CACHE = ROOT / ".cache" / PROTOCOL["dataset_revision"]


def hf_environment(*, offline: bool) -> dict[str, str]:
    # Explicit subpaths prevent ambient shared-cache settings from defeating isolation.
    return {
        "HF_HOME": str(CACHE / "hf"),
        "HF_HUB_CACHE": str(CACHE / "hf" / "hub"),
        "HF_DATASETS_CACHE": str(CACHE / "hf" / "datasets"),
        # Dataset-only offline mode: the pinned harness's global-offline
        # tokenizer path mistakes local directories for Hub IDs. Local tokenizer
        # directories load normally without global-offline; no remote ID is used.
        "HF_HUB_OFFLINE": "",
        "TRANSFORMERS_OFFLINE": "",
        "HF_DATASETS_OFFLINE": "1" if offline else "0",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    }


def prepare_dataset() -> None:
    os.environ.update(hf_environment(offline=False))
    from datasets import load_dataset

    ds = load_dataset(
        PROTOCOL["dataset"], revision=PROTOCOL["dataset_revision"], split="train"
    )
    if len(ds) != PROTOCOL["dataset_sessions"]:
        raise RuntimeError(f"Expected 393 sessions, received {len(ds)}")
    cache_files = [Path(item["filename"]).resolve() for item in ds.cache_files]
    if not cache_files or not all(p.is_relative_to(CACHE) for p in cache_files):
        raise RuntimeError("Dataset escaped its isolated revision cache")
    receipt = {
        "dataset": PROTOCOL["dataset"],
        "revision": PROTOCOL["dataset_revision"],
        "sessions": len(ds),
        "fingerprint": ds._fingerprint,
        "cache_files": [str(p.relative_to(CACHE)) for p in cache_files],
    }
    # Prove the exact public-loader call resolves to the pinned Arrow data offline.
    # Upstream currently has no CLI dataset-revision flag. No loader patch or
    # local-directory/unsafe-override route is used here.
    code = """
import json, os
from datasets import load_dataset
d = load_dataset(os.environ['AGENTX_DATASET'], split='train',
                 trust_remote_code=False, streaming=False)
print(json.dumps({'fingerprint': d._fingerprint, 'sessions': len(d),
                  'files': [x['filename'] for x in d.cache_files]}))
"""
    env = {**os.environ, **hf_environment(offline=True),
           "AGENTX_DATASET": PROTOCOL["dataset"]}
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, check=True,
        capture_output=True, text=True, timeout=180,
    )
    observed = json.loads(result.stdout.strip().splitlines()[-1])
    if (observed["fingerprint"] != receipt["fingerprint"]
            or observed["sessions"] != receipt["sessions"]
            or [Path(p).resolve() for p in observed["files"]] != cache_files):
        raise RuntimeError("Upstream offline loader did not select the frozen dataset")
    (CACHE / "prepared.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


def verify_dataset() -> dict:
    receipt = json.loads((CACHE / "prepared.json").read_text())
    for key, value in (("dataset", PROTOCOL["dataset"]),
                       ("revision", PROTOCOL["dataset_revision"]),
                       ("sessions", PROTOCOL["dataset_sessions"])):
        if receipt[key] != value:
            raise RuntimeError(f"Prepared dataset mismatch: {key}")
    # Check in the actual run interpreter, not merely the preparation receipt.
    from datasets import load_dataset

    ds = load_dataset(PROTOCOL["dataset"], split="train",
                      trust_remote_code=False, streaming=False)
    expected = [(CACHE / p).resolve() for p in receipt["cache_files"]]
    observed = [Path(p["filename"]).resolve() for p in ds.cache_files]
    if (ds._fingerprint != receipt["fingerprint"] or len(ds) != receipt["sessions"]
            or expected != observed):
        raise RuntimeError("Offline dataset identity changed; refuse replay")
    return receipt


if __name__ == "__main__":
    prepare_dataset()
