import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest


SPEC = importlib.util.spec_from_file_location(
    "runpod_worker", Path(__file__).resolve().parents[2] / "scripts/runpod_worker.py"
)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


@pytest.fixture
def config():
    return {"run_id": "run-1", "deadline": time.time() + 300,
            "image": "image:test", "prompts": 4, "warmups": 2,
            "input_len": 16, "output_len": 8, "concurrency": 2,
            "seed": 0, "profile": False}


def test_manifest_records_actual_data_center(config, tmp_path, monkeypatch):
    monkeypatch.setenv('RUNPOD_DC_ID', 'US-KS-2')
    w = worker.Worker(config, 'secret', 'pod1', 'pod-key', tmp_path)
    assert w.manifest['data_center'] == 'US-KS-2'


def test_mi350x_is_checked_and_recorded(config, tmp_path, monkeypatch):
    import sys
    import types
    config['gpu'] = 'AMD Instinct MI350 OAM'
    torch = types.SimpleNamespace(version=types.SimpleNamespace(hip='7.0'),
        cuda=types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                                   get_device_name=lambda n: 'AMD Instinct MI350X'))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setattr(worker.importlib.metadata, 'version', lambda name: {'vllm': '0.17.0', 'transformers': '5.8.0', 'torch': '2.9.0'}[name])
    w = worker.Worker(config, 'secret', 'pod1', 'pod-key', tmp_path)
    w.check_compatibility()
    assert w.manifest['gpu']['type'] == 'AMD Instinct MI350 OAM'
    assert w.manifest['gpu']['detected_name'] == 'AMD Instinct MI350X'


def test_speculative_metrics_sum_labels_and_report_missing_or_zero():
    before = 'vllm:spec_decode_num_accepted_tokens_total{engine="0"} 10\nvllm:spec_decode_num_draft_tokens_total 20\n'
    after = 'vllm:spec_decode_num_accepted_tokens_total{engine="0"} 16\nvllm:spec_decode_num_draft_tokens_total 30\n'
    assert worker.speculative_deltas(before, after)["acceptance_rate"] == 0.6
    assert worker.speculative_deltas(before, before)["acceptance_rate"] is None
    missing = worker.speculative_deltas('', after)
    assert missing["acceptance_rate"] is None and missing["reason"]


def test_recipe_and_benchmark_target_local_chat_and_fixed_lengths(config, tmp_path):
    w = worker.Worker(config, "secret", "pod1", "pod-key", tmp_path)
    serve = w.serve_command()
    assert serve[:3] == ["vllm", "serve", "Qwen/Qwen3.8-27B-FP8"]
    assert serve[serve.index("--host") + 1] == "127.0.0.1"
    assert json.loads(serve[serve.index("--speculative-config") + 1]) == {"method": "mtp", "num_speculative_tokens": 3}
    bench = w.benchmark_command(4, "benchmark.json")
    for flag, value in [("--backend", "openai-chat"), ("--endpoint", "/v1/chat/completions"),
                        ("--host", "127.0.0.1"), ("--num-prompts", "4"),
                        ("--random-input-len", "16"), ("--random-output-len", "8"),
                        ("--random-range-ratio", "0")]:
        assert bench[bench.index(flag) + 1] == value
    assert "--profile" not in bench


def test_profile_trace_requires_gpu_kernel_activity(tmp_path):
    trace = tmp_path / 'trace.json.gz'
    with gzip.open(trace, 'wt') as f:
        json.dump({"traceEvents": [{"cat": "cpu_op", "name": "call"}]}, f)
    assert not worker.trace_has_gpu_activity(trace)
    with gzip.open(trace, 'wt') as f:
        json.dump({"traceEvents": [{"cat": "kernel", "name": "hip_kernel", "dur": 5}]}, f)
    assert worker.trace_has_gpu_activity(trace)


def test_watchdog_retries_transient_delete_and_uses_pod_key(monkeypatch, config, tmp_path):
    calls = []
    class Response:
        status = 204
        def __enter__(self): return self
        def __exit__(self, *args): pass
    def send(request, timeout):
        calls.append((request.full_url, request.get_header("Authorization"), request.method))
        if len(calls) == 1:
            raise HTTPError(request.full_url, 503, 'unavailable', {}, None)
        return Response()
    monkeypatch.setattr(worker, 'urlopen', send)
    monkeypatch.setattr(worker.time, 'sleep', lambda _: None)
    config['deadline'] = time.time() - 1
    worker.Worker(config, 'secret', 'pod1', 'pod-key', tmp_path).watchdog()
    assert calls == [('https://api.runpod.io/v2/pods/pod1', 'Bearer pod-key', 'DELETE')] * 2


def test_http_auth_chunk_bounds_and_no_filesystem_access(config, tmp_path):
    w = worker.Worker(config, 'secret', 'pod1', 'pod-key', tmp_path)
    w.artifact_path.write_bytes(b'abcdefgh')
    w.artifact_size = 8
    w.artifact_sha256 = hashlib.sha256(b'abcdefgh').hexdigest()
    server = worker.make_server(w, '127.0.0.1', 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = 'http://127.0.0.1:%s' % server.server_port
    def get(path, secret='secret'):
        return urlopen(Request(base + path, headers={'Authorization': 'Bearer ' + secret}), timeout=2)
    try:
        with pytest.raises(HTTPError) as err:
            get('/status', 'wrong')
        assert err.value.code == 401
        with get('/status') as response:
            status = json.load(response)
        assert status['artifact_size'] == 8 and status['artifact_sha256'] == w.artifact_sha256
        for secret, expected in [('wrong', 401), ('secret', 405)]:
            req = Request(base + '/artifact', method='POST', headers={'Authorization': 'Bearer ' + secret}, data=b'')
            with pytest.raises(HTTPError) as err:
                urlopen(req, timeout=2)
            assert err.value.code == expected
        with get('/artifact?offset=2&size=3') as response:
            assert response.read() == b'cde'
        for path in ['/artifact?offset=-1&size=1', '/artifact?offset=0&size=999999999', '/artifact?offset=0&size=1&path=/etc/passwd', '/etc/passwd']:
            with pytest.raises(HTTPError):
                get(path)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_artifact_contains_partial_manifest_and_redacted_bounded_logs(config, tmp_path):
    w = worker.Worker(config, 'secret-value', 'pod1', 'key-value', tmp_path)
    (tmp_path / 'server.log').write_text('secret-value key-value hf-value')
    w.redactions.append('hf-value')
    w.error = 'failure: key-value'
    w.finalize('failed')
    with tarfile.open(w.artifact_path, 'r:gz') as archive:
        manifest = json.load(archive.extractfile('manifest.json'))
        assert manifest['status'] == 'failed' and manifest['run_id'] == 'run-1'
        assert b'[REDACTED]' in archive.extractfile('server.log').read()
        for member in archive.getmembers():
            if member.isfile():
                contents = archive.extractfile(member).read()
                assert b'secret-value' not in contents and b'key-value' not in contents and b'hf-value' not in contents


def test_missing_pod_key_fails_before_model_loading_and_produces_artifact(config, tmp_path):
    w = worker.Worker(config, 'secret', 'pod1', '', tmp_path)
    w.run()
    assert w.state == 'failed' and 'watchdog' in w.error
    assert w.artifact_size and len(w.artifact_sha256) == 64
    with tarfile.open(w.artifact_path, 'r:gz') as archive:
        assert json.load(archive.extractfile('manifest.json'))['status'] == 'failed'


def test_failed_measured_requests_preserve_native_results(monkeypatch, config, tmp_path):
    w = worker.Worker(config, 'secret', 'pod1', 'pod-key', tmp_path)
    result = {'completed': 3, 'failed': 1, 'duration': 2, 'total_input_tokens': 48,
              'total_output_tokens': 20, 'request_throughput': 1.5, 'output_throughput': 10}
    (tmp_path / 'benchmark.json').write_text(json.dumps(result))
    class Process:
        returncode = 0
    monkeypatch.setattr(worker.subprocess, 'run', lambda *args, **kwargs: Process())
    with pytest.raises(RuntimeError, match='requests failed'):
        w.execute_benchmark(4, 'benchmark.json', 'benchmark.log')
    assert w.manifest['benchmark']['failed'] == 1
    w.finalize('failed')
    with tarfile.open(w.artifact_path, 'r:gz') as archive:
        assert json.load(archive.extractfile('benchmark.json')) == result


def test_profiler_is_stopped_when_profile_workload_fails(monkeypatch, config, tmp_path):
    config['profile'] = True
    w = worker.Worker(config, 'secret', 'pod1', 'pod-key', tmp_path)
    calls = []
    monkeypatch.setattr(w, 'request', lambda path, **kwargs: calls.append(path))
    def fail(*args):
        raise RuntimeError('request failed')
    monkeypatch.setattr(w, 'execute_benchmark', fail)
    with pytest.raises(RuntimeError, match='request failed'):
        w.profile()
    assert calls == ['/start_profile', '/stop_profile']


def test_profile_requires_gpu_trace_even_after_successful_workload(monkeypatch, config, tmp_path):
    config['profile'] = True
    w = worker.Worker(config, 'secret', 'pod1', 'pod-key', tmp_path)
    monkeypatch.setattr(w, 'request', lambda *args, **kwargs: b'')
    monkeypatch.setattr(w, 'execute_benchmark', lambda *args: {})
    with pytest.raises(RuntimeError, match='GPU kernel activity'):
        w.profile()


def test_secret_in_native_output_is_excluded_and_run_fails(config, tmp_path):
    w = worker.Worker(config, 'secret-value', 'pod1', 'key-value', tmp_path)
    (tmp_path / 'benchmark.json').write_text('{"text":"key-value"}')
    w.finalize('completed')
    assert w.state == 'failed'
    with tarfile.open(w.artifact_path, 'r:gz') as archive:
        assert 'benchmark.json' not in archive.getnames()
        assert json.load(archive.extractfile('manifest.json'))['status'] == 'failed'


def test_completed_artifact_is_reported_as_done(config, tmp_path):
    w = worker.Worker(config, 'secret', 'pod1', 'pod-key', tmp_path)
    w.finalize('completed')
    assert w.state == 'done'
    assert w.artifact_size == w.artifact_path.stat().st_size
    assert w.artifact_sha256 == hashlib.sha256(w.artifact_path.read_bytes()).hexdigest()


def test_worker_runs_warmup_then_measured_phase_with_metrics(monkeypatch, config, tmp_path):
    w = worker.Worker(config, 'secret', 'pod1', 'pod-key', tmp_path)
    events = []
    stopped = threading.Event()
    def watchdog():
        w.watchdog_started.set()
        stopped.wait(10)
    class Process:
        def __init__(self, *args, **kwargs): self.alive = True
        def poll(self): return None if self.alive else 0
        def terminate(self): self.alive = False
        def wait(self, timeout=None): return 0
    def bench(count, filename, logname):
        events.append(filename)
        result = {'completed': count, 'failed': 0, 'duration': 2,
                  'total_input_tokens': 64, 'total_output_tokens': 32,
                  'request_throughput': 2, 'output_throughput': 16}
        (tmp_path / filename).write_text(json.dumps(result))
        if filename == 'benchmark.json':
            w.record_benchmark(result)
        return result
    def request(path, **kwargs):
        events.append(path)
        return (b'vllm:spec_decode_num_accepted_tokens_total 2\n'
                b'vllm:spec_decode_num_draft_tokens_total 4\n' if events.count(path) == 1 else
                b'vllm:spec_decode_num_accepted_tokens_total 5\n'
                b'vllm:spec_decode_num_draft_tokens_total 10\n')
    monkeypatch.setattr(w, 'watchdog', watchdog)
    monkeypatch.setattr(w, 'check_compatibility', lambda: None)
    monkeypatch.setattr(w, 'wait_ready', lambda: None)
    monkeypatch.setattr(w, 'record_revision', lambda: None)
    monkeypatch.setattr(w, 'execute_benchmark', bench)
    monkeypatch.setattr(w, 'request', request)
    monkeypatch.setattr(worker.subprocess, 'Popen', Process)
    try:
        w.run()
    finally:
        stopped.set()
    assert w.state == 'done'
    assert events == ['warmup.json', '/metrics', 'benchmark.json', '/metrics']
    with tarfile.open(w.artifact_path, 'r:gz') as archive:
        manifest = json.load(archive.extractfile('manifest.json'))
        assert manifest['speculative_decoding']['acceptance_rate'] == 0.5
        assert manifest['benchmark']['completed'] == 4
        assert 'profile' not in manifest


def test_missing_required_measured_metrics_fails(monkeypatch, config, tmp_path):
    w = worker.Worker(config, 'secret', 'pod1', 'pod-key', tmp_path)
    (tmp_path / 'benchmark.json').write_text(json.dumps({'completed': 4, 'failed': 0}))
    class Process:
        returncode = 0
    monkeypatch.setattr(worker.subprocess, 'run', lambda *args, **kwargs: Process())
    with pytest.raises(RuntimeError, match='missing or invalid measured'):
        w.execute_benchmark(4, 'benchmark.json', 'benchmark.log')


def test_version_requirement_rejects_prereleases_at_minimum():
    assert not worker.version_at_least('0.17.0rc1', (0, 17, 0))
    assert not worker.version_at_least('0.16.9', (0, 17, 0))
    assert worker.version_at_least('0.17.0+rocm', (0, 17, 0))


def test_profile_pass_records_bounded_workload_and_gpu_trace(monkeypatch, config, tmp_path):
    config.update(profile=True, input_len=512, output_len=256, concurrency=8)
    w = worker.Worker(config, 'secret', 'pod1', 'pod-key', tmp_path)
    calls = []
    monkeypatch.setattr(w, 'request', lambda path, **kwargs: calls.append(path))
    def bench(count, filename, logname):
        assert count == 2
        with gzip.open(tmp_path / 'profile' / 'rank0.trace.json.gz', 'wt') as stream:
            json.dump({'traceEvents': [{'cat': 'kernel', 'name': 'compute', 'dur': 1}]}, stream)
    monkeypatch.setattr(w, 'execute_benchmark', bench)
    w.profile()
    assert calls == ['/start_profile', '/stop_profile']
    assert w.manifest['profile']['gpu_activity_verified'] is True
    cmd = w.manifest['profile']['arguments']
    assert cmd[cmd.index('--random-input-len') + 1] == '128'
    assert cmd[cmd.index('--random-output-len') + 1] == '32'
    assert cmd[cmd.index('--max-concurrency') + 1] == '2'
