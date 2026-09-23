# AgentX bench

Official AgentX **256k** workload, official replay implementation, two measurement
durations. This repository only pins, launches and records the benchmark. It does
not implement another scheduler, mock agent, prompt generator, or scoring suite.

| Profile | Measured window | Warmup | Intended use |
|---|---:|---|---|
| `formal` | 60 minutes | Official primers + 10 requests/lane | Website-duration reproduction |
| `smoke` | 15 minutes | Identical | Shortened development regression |

Both use all 393 official source sessions as the sampling pool, their subagent
DAGs, original replay delays, full requested outputs and prefix relationships.
Loading the full pool does **not** mean every source request completes in a run.
Concurrency means live agent **session trees**, not a fixed HTTP request count.

**15 minutes is measurement time, not end-to-end wall time.** Dataset reconstruction,
deep-prefix warmup, up to 30 seconds of drain, and export add overhead. First-time
preparation can be substantial. We preserve upstream drain accounting and metrics.

## Install and prepare (CPU only)

```bash
git clone git@github.com:CubeLander/agentx-bench.git
cd agentx-bench
uv sync --locked
uv run --frozen python prepare.py
uv run --frozen pytest -q
```

`uv.lock` pins dependencies and the official harness Git commit. `protocol.json`
pins the dataset revision, scenario and explicit website-methodology settings.
Preparation downloads the public corpus without credentials into a repository-local
cache, then checks that the **unmodified official public loader** selects exactly
that dataset offline. Runtime refuses a different fingerprint/file identity.
No raw data, virtual environment, tokenizer or result is committed.

### Prepare the target

Copy `examples/target.json` to a local, ignored file (e.g. `.cache/target.json`).
Replace its example values with your deployed model/engine revisions, server
origin, accelerator allocation, context limit, host-memory budget, and a **local
frozen tokenizer directory**. Use the tokenizer matching the served model; record
its revision in your target, especially if different from the weights revision.
Relative tokenizer paths resolve relative to the target JSON, not the shell cwd.

Download any tokenizer files beforehand, for example using `hf download` with an
immutable model revision and tokenizer/config-only include patterns suitable for
that model, or use the tokenizer already present with your deployed weights.
The corpus loader is offline; the tokenizer is an explicit local directory.
Global `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` are cleared because this pinned
harness mistakes local tokenizer directories for Hub IDs in that mode. This is
not a network sandbox; no remote tokenizer ID or model-weight fetch is requested.

The endpoint must be a trusted, explicitly selected OpenAI-compatible chat server
with streaming usage counts and `ignore_eos` support. The initial wrapper accepts
credential-free origins only; never put API keys in target files or command lines.
Do not use a paid provider endpoint merely to test the launcher.

The server must support the official corpus's 256,000-token input+output cap **plus
its actual chat-template/marker overhead**, normally with a >=262,144 native
window. We deliberately do not pass `--max-context-length`: upstream can use that
flag to discard entire conversations, which would change this selected corpus.
Unsupported models are not made to fit by truncation or dropped requests.

## Inspect and run

```bash
# No server requests; print the exact frozen recipe.
uv run --frozen python bench.py plan --target .cache/target.json \
  --profile smoke --concurrency 8

# Requires an already running, authorized server. Does not launch/modify it.
uv run --frozen python bench.py run --target .cache/target.json \
  --profile smoke --concurrency 8

uv run --frozen python bench.py run --target .cache/target.json \
  --profile formal --concurrency 8
```

8 is an **example**, not a calibrated operating point. Choose and freeze the
client count before comparing candidates; run separate points for a capacity
curve. No automatic concurrency sweep, arrival-rate overlay, client-side memory
emulation, random subset, length scaling, or output truncation is added.

Follow progress with `tail -f artifacts/<run>/harness.log`. Output includes:

- `run.json`: protocol, target declarations, exact command, preparation receipt,
  overall status and total harness wall time;
- `harness.log`: upstream progress, including configuration/warmup/profiling;
- `aiperf/`: unmodified upstream metrics, request records, logs and configuration.

Ctrl-C cancels only the task-owned harness process group, not the serving engine.
The wrapper exits unsuccessfully for nonzero harness exit, absent/ambiguous result
files, or upstream `submission_valid != true`; partial outputs remain available.
Smoke is always labelled 900 seconds even if the harness validity stamp is true.
This stamp is **not** independent certification of server configuration or an
official SemiAnalysis submission/endorsement.

## Protocol choices and comparison boundaries

- Follow the [website methodology](https://inferencex.semianalysis.com/agentx/methodology):
  trajectory start 25–75%, primer warmup plus 10 requests per lane, seeded content,
  per-play prefix cache busting, closed-loop DAG replay. All are explicit.
- The pinned harness's MVP defaults differ: start 0–100%, measurement 1800s, no
  extra per-lane warmup by default. Its validator accepts 900s. We do **not** rely
  on those defaults or equate passing its validator with matching the website.
- Retain throughput, TTFT, output interactivity/latency distributions and errors
  together. Do not replace official metrics with a home-grown aggregate score.
  Closed-loop faster candidates may reach a different request mix in equal time.
- Synthetic content does **not** measure model answer quality, code correctness,
  real tool execution, or agent task completion. No semantic quality claim.
- Server settings are operator declarations, not client-enforced controls. Record
  precision, parallelism, KV capacity/offload, cache routing and software versions.
  Multi-replica deployments need conversation-aware routing; this initial wrapper
  targets one server and does not silently select a router-specific policy.
- Follow the official host-DRAM allocation rule: nonstandard systems capped at
  3 TB, proportional allocation for the accelerators used; listed standard systems
  follow installed capacity. Do not borrow the full host budget for a small slice.
- Speculative decoding needs model/speculator/draft-length/thinking-mode matched
  [SPEED-Bench acceptance evidence](https://inferencex.semianalysis.com/agentx/methodology)
  and the server's forced-acceptance setting. Otherwise disable it and disclose
  that configuration. The wrapper never guesses acceptance from synthetic tokens.
- Cost/energy are not invented by the client: retain allocation and timing, and
  attach declared prices or measured energy separately when needed.
- Hardware admission, accelerator leases and server startup/cleanup belong to the
  deployment owner. This repository neither initializes devices nor resumes jobs.

## Sources

- [Official 256k data and Apache-2.0 data card](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k)
- [Official AgentX harness](https://github.com/SemiAnalysisAI/agentx-harness)
- [Pinned replay guide](https://github.com/SemiAnalysisAI/agentx-harness/blob/56a0cf70f4c0359454ee4bd15a17770b541a3e3e/docs/tutorials/agentx-mvp.md)

Upstream code/data retain their licenses and attribution; they are dependencies,
not vendored or relicensed by this repository.
