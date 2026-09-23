import json
from pathlib import Path

import pytest

import bench
from prepare import PROTOCOL, hf_environment


@pytest.fixture
def target():
    return bench.read_target(bench.ROOT / "examples/target.json")


def resolve(target, profile):
    # Exercise the real CLI parser, converter and plan builder, not a mock.
    from aiperf.cli import app
    from aiperf.config.flags.resolver import resolve_config
    from aiperf.config.loader import build_benchmark_plan
    from aiperf.config.resolution.plan import BenchmarkRun
    from aiperf.config.resolution.resolvers import build_default_resolver_chain

    argv = bench.command(target, profile, 4, Path("/tmp/agentx-bench-config-test"))
    _, bound, _ = app.parse_args(argv[1:])
    cfg = resolve_config(bound.arguments["cli_config"], None)
    plan = build_benchmark_plan(cfg)
    run = BenchmarkRun(benchmark_id="protocol-test", cfg=plan.configs[0],
                       artifact_dir=Path("/tmp/agentx-bench-config-test"))
    build_default_resolver_chain().resolve_all(run)
    assert run.resolved.scenario_outcome.submission_valid is True
    assert run.cfg.get_profiling_phases()[0].timing_mode == "agentic_replay"
    return cfg.model_dump(mode="json")


def test_profiles_only_change_measurement(target):
    smoke = resolve(target, "smoke")
    formal = resolve(target, "formal")
    s = smoke["benchmark"]["phases"][0]
    f = formal["benchmark"]["phases"][0]
    assert s["duration"] == 900
    assert f["duration"] == 3600
    f["duration"] = 900
    assert smoke == formal
    assert s["warmup_requests_per_lane"] == 10
    assert s["trajectory_start_min_ratio"] == .25
    assert s["trajectory_start_max_ratio"] == .75
    assert s["concurrency"] == 4
    ds = smoke["benchmark"]["datasets"][0]
    assert ds["dataset"] == PROTOCOL["dataset_alias"]
    assert ds["entries"] is None
    assert ds["max_context_length"] is None
    assert ds["synthesis"] is None
    assert ds["ignore_trace_delays"] is False
    assert ds["cache_bust"]["target"] == "first_turn_prefix"
    assert smoke["benchmark"]["endpoint"]["extra"]["ignore_eos"] is True
    assert smoke["benchmark"]["endpoint"]["use_server_token_count"] is True


def test_harness_identity():
    bench.verify_harness()


@pytest.mark.parametrize("url", ["http://u:p@localhost:8000", "http://localhost?key=x",
                                 "http://localhost/v1", "file:///tmp/server"])
def test_unsafe_origins_rejected(target, tmp_path, url):
    target["url"] = url
    p = tmp_path / "target.json"
    p.write_text(json.dumps(target))
    with pytest.raises(ValueError, match="credential-free"):
        bench.read_target(p)


@pytest.mark.parametrize("change", [{"max_model_len": 16384},
                                   {"speculative_decoding": {"enabled": True}},
                                   {"host_kv_budget_gib": -1}])
def test_noncompliant_target_rejected(target, tmp_path, change):
    p = tmp_path / "target.json"
    p.write_text(json.dumps({**target, **change}))
    with pytest.raises(ValueError):
        bench.read_target(p)


def test_no_arbitrary_flag_passthrough(target):
    argv = bench.command(target, "smoke", 4, Path("out"))
    assert "--unsafe-override" not in argv
    assert "--max-context-length" not in argv
    assert "--num-dataset-entries" not in argv
    with pytest.raises(ValueError):
        bench.command(target, "smoke", 0, Path("out"))


@pytest.mark.parametrize("returncode,valid,expected", [(0, True, "completed"),
    (1, True, "failed_or_invalid"), (0, False, "failed_or_invalid"),
    (0, None, "failed_or_invalid")])
def test_report_does_not_hide_failures(tmp_path, returncode, valid, expected):
    data = {"metadata": {"submission_valid": valid,
                         "submission_invalid_reasons": ["sentinel"]}}
    p = tmp_path / "profile_export_aiperf.json"
    p.write_text(json.dumps(data))
    result = bench.summarize(tmp_path, returncode)
    assert result["status"] == expected
    assert result["upstream_reports"][0]["submission_invalid_reasons"] == ["sentinel"]


def test_missing_or_multiple_reports_fail(tmp_path):
    assert bench.summarize(tmp_path, 0)["status"] == "failed_or_invalid"
    for name in ("a", "b"):
        d = tmp_path / name
        d.mkdir()
        (d / "profile_export_aiperf.json").write_text(
            '{"metadata":{"submission_valid":true}}')
    assert bench.summarize(tmp_path, 0)["status"] == "failed_or_invalid"


def test_isolated_offline_environment():
    env = hf_environment(offline=True)
    assert env["HF_DATASETS_OFFLINE"] == "1"
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == ""
    assert PROTOCOL["dataset_revision"] in env["HF_DATASETS_CACHE"]


def test_cancel_targets_only_owned_process_group(monkeypatch):
    signals = []

    class Process:
        pid = 43210
        calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise KeyboardInterrupt
            return 130

    def popen(*args, **kwargs):
        assert kwargs["start_new_session"] is True
        return Process()

    monkeypatch.setattr(bench.subprocess, "Popen", popen)
    monkeypatch.setattr(bench.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    with pytest.raises(KeyboardInterrupt):
        bench.execute_owned(["aiperf"], {}, None)
    assert signals == [(43210, bench.signal.SIGINT)]
