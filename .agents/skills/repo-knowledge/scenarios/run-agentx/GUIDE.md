# Run or update AgentX

Use before changing replay settings, dependency/data pins, or diagnosing a run.
The root README owns ordinary commands. This guide preserves paid traps, not a
second copy of the CLI manual.

## Website protocol is not the harness defaults

Fletcher selected the official 256k corpus and official replay, with a 15-minute
smoke and one-hour formal window **after** warmup. At harness `56a0cf70`, the
scenario minimum is 900s but its defaults are 1800s and start ratios 0/1, and the
extra warmup request count defaults to unset. The website instead says 3600s,
start ratios .25/.75, primers + ten extra requests/lane. `protocol.json` and the
launcher set these explicitly. Smoke changes only measured duration, never the
corpus, DAG, output budget, warmup or delays. `submission_valid` alone cannot prove
website conformance; the runtime validator even accepts changed start ratios.

Before upgrading, inspect `common/scenario/inferencex_agentx_mvp.py`, the CLI
converter/resolver, `config/phases.py`, and the published methodology. Run the
real parser/resolver tests; don't merely compare command strings. Never silently
change the frozen protocol to make an upstream validator happy.

## Revision pinning without patching the official loader

The pinned official HF Weka loader has no CLI dataset-revision setting. Preparation
loads the exact commit into a dedicated revision-specific HF dataset cache, then
proves that the native public-loader call selects the same fingerprint and Arrow
files offline. `verify_dataset` repeats that small check before real runs.
Do not replace the public alias with a local JSON directory: upstream treats that
as an unverified corpus and needs unsafe-override. Do not set a corpus subset or
`--max-context-length` to fix overflows; the latter can filter whole conversations.

Global `HF_HUB_OFFLINE=1` is **not** interchangeable with dataset-only offline:
this pinned harness's `Tokenizer._resolve_local_snapshot` sends a local tokenizer
directory to `snapshot_download`, causing HFValidationError. Our CPU wire test
observed this. `HF_DATASETS_OFFLINE=1` keeps datasets offline while clearing
`HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` lets explicit local tokenizer paths
load correctly. This is not a network sandbox. Models needing custom tokenizer
code require separate review; do not silently enable trust-remote-code.

## Bounded validation on 2026-09-23

The actual locked 256k corpus was downloaded and all rows parsed with upstream
`WekaTrace`: 393 sessions, 68,266 model requests, 1,697 subagent groups, max recorded
input+output 255,999. Offline cache selection matched the pinned revision.

`tests/test_transport.py` executes the unmodified harness against a task-owned
loopback streaming mock with a generated tiny tokenizer and tiny Weka fixture.
It deliberately uses a two-second, local-corpus unsafe override. Its result must
remain invalid. This verifies client wiring and export, **not** model performance,
256k tokenization fidelity, a 15-minute endurance run, or NPU serving.

The client never starts or stops model servers. Server/hardware admission and
speculation/DRAM compliance are deployment responsibilities. Obtain an explicit
target before sending sustained traffic. Preserve upstream invalidity reasons and
partial artifacts; no answer-quality claims are possible from synthetic payloads.
