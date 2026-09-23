"""CPU-only wire integration; tiny synthetic fixture, NEVER an AgentX result."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

import bench


@pytest.mark.integration
def test_official_replay_to_streaming_mock(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    from transformers import PreTrainedTokenizerFast

    raw = Tokenizer(models.BPE(unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    raw.train_from_iterator(
        ["def hello(x):\n    return x + 1\n# inspect the function and fix tests\n"],
        trainers.BpeTrainer(vocab_size=300, special_tokens=["[UNK]", "[EOS]"],
                            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()),
    )
    tokenizer = tmp_path / "tokenizer"
    PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", eos_token="[EOS]").save_pretrained(tokenizer)
    requests = [{"t": i * .1, "type": "s", "model": "fixture", "in": 128,
                 "out": 4, "hash_ids": [1, i + 2], "api_time": .01}
                for i in range(30)]
    fixture = tmp_path / "trace.json"
    fixture.write_text(json.dumps({"id": "cpu-fixture", "models": ["fixture"],
                                  "block_size": 64, "hash_id_scope": "local",
                                  "requests": requests}))
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"data":[{"id":"fixture"}]}')

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            n = body.get("max_completion_tokens", body.get("max_tokens", 4))
            assert body["stream"] is True
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                for i in range(n):
                    event = {"id": "mock", "object": "chat.completion.chunk",
                             "model": "fixture", "choices": [{"index": 0,
                             "delta": {"content": " x"}, "finish_reason": None}]}
                    self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(.005)
                event = {"id": "mock", "choices": [{"index": 0, "delta": {},
                          "finish_reason": "length"}], "usage": {
                          "prompt_tokens": 128, "completion_tokens": n,
                          "total_tokens": 128 + n}}
                self.wfile.write(f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    target = {"url": f"http://127.0.0.1:{server.server_port}", "model": "fixture",
              "tokenizer": str(tokenizer)}
    output = tmp_path / "output"
    argv = bench.command(target, "smoke", 1, output)
    index = argv.index("--public-dataset")
    argv[index:index + 2] = ["--input-file", str(fixture), "--custom-dataset-type", "weka_trace"]
    argv[argv.index("--benchmark-duration") + 1] = "2"
    argv += ["--unsafe-override"]  # test-only: not exposed by the production wrapper
    env = {**os.environ, "HF_HUB_OFFLINE": "", "HF_DATASETS_OFFLINE": "1",
           "TRANSFORMERS_OFFLINE": "",
           "TOKENIZERS_PARALLELISM": "false"}
    try:
        result = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=120)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    (tmp_path / "transport.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, (result.stdout + result.stderr)[-6000:]
    assert len(seen) > 10  # snapshot + extra warmup + profiling
    assert all(request.get("ignore_eos") is True for request in seen)
    summary = bench.summarize(output, 0)
    assert summary["status"] == "failed_or_invalid"  # never promote mock evidence
    assert summary["upstream_reports"]
    assert summary["upstream_reports"][0]["submission_valid"] is False
