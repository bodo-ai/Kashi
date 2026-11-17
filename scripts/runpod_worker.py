#!/usr/bin/env python3
"""Standalone Runpod benchmark worker, executed inside the selected image."""
import gzip
import hashlib
import hmac
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, urlsplit
from urllib.request import Request, urlopen

MODEL = "Qwen/Qwen3.8-27B-FP8"
MAX_CHUNK = 4 * 1024 * 1024
LOG_LIMIT = 256 * 1024
UNAVAILABLE = "unavailable"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def speculative_deltas(before, after):
    names = ("vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_draft_tokens_total")
    def counters(text):
        values = {}
        for line in text.splitlines():
            match = re.match(r'^([^\s{]+)(?:\{[^}]*\})?\s+([^\s]+)', line)
            if match and match.group(1) in names:
                try:
                    value = float(match.group(2))
                    if not math.isfinite(value):
                        continue
                    values[match.group(1)] = values.get(match.group(1), 0) + value
                except ValueError:
                    continue
        return values
    a, b = counters(before), counters(after)
    result = {"accepted_tokens_delta": None, "draft_tokens_delta": None,
              "acceptance_rate": None, "reason": None}
    if any(n not in a or n not in b for n in names):
        result["reason"] = "speculative decoding counters unavailable"
        return result
    accepted, draft = (b[n] - a[n] for n in names)
    result.update(accepted_tokens_delta=accepted, draft_tokens_delta=draft)
    if accepted < 0 or draft < 0:
        result["reason"] = "speculative decoding counters decreased"
    elif draft == 0:
        result["reason"] = "no draft tokens recorded"
    else:
        result["acceptance_rate"] = accepted / draft
    return result


def trace_has_gpu_activity(path):
    try:
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            trace = json.load(stream)
        events = trace.get("traceEvents", []) if isinstance(trace, dict) else trace
        return any(isinstance(event, dict) and str(event.get("cat", "")).lower() in
                   ("kernel", "gpu_kernel", "gpu kernel") and
                   isinstance(event.get("dur"), (int, float)) and event["dur"] > 0
                   for event in events)
    except (OSError, ValueError, TypeError):
        return False


def version_at_least(version, minimum):
    # Reject prereleases at the minimum as they do not satisfy >= stable.
    match = re.match(r'^(\d+)\.(\d+)(?:\.(\d+))?([^+]*)', version)
    if not match:
        return False
    parts = tuple(int(x or 0) for x in match.group(1, 2, 3))
    return parts > minimum or (parts == minimum and not re.search(r'(?:a|b|rc|dev)', match.group(4), re.I))


class Worker:
    def __init__(self, config, secret, pod_id, api_key, directory):
        self.config = config
        self.secret = secret
        self.pod_id = pod_id
        self.api_key = api_key
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.artifact_path = self.directory / 'artifact.tar.gz'
        self.artifact_size = None
        self.artifact_sha256 = None
        self.state = 'starting'
        self.error = None
        self.started = utc_now()
        self.server_process = None
        self.watchdog_started = threading.Event()
        self.lock = threading.Lock()
        self.redactions = [s for s in (secret, api_key, os.environ.get('HF_TOKEN')) if s]
        self.manifest = {
            'schema_version': 1, 'run_id': config.get('run_id'), 'pod_id': pod_id,
            'started_at': self.started, 'ended_at': None, 'status': 'starting',
            'gpu': {'type': config.get('gpu'), 'count': 1},
            'data_center': os.environ.get('RUNPOD_DC_ID') or config.get('data_center') or UNAVAILABLE,
            'image': {'reference': config.get('image'), 'digest': UNAVAILABLE},
            'model': {'id': MODEL, 'revision': UNAVAILABLE},
            'versions': {'vllm': UNAVAILABLE, 'torch': UNAVAILABLE, 'rocm': UNAVAILABLE, 'transformers': UNAVAILABLE},
            'workload': {k: config.get(k) for k in ('prompts', 'warmups', 'input_len', 'output_len', 'concurrency', 'seed', 'profile')},
            'serve_arguments': None, 'benchmark_arguments': None,
            'measurement_units': {'duration': 'seconds', 'request_throughput': 'requests/second',
                                  'output_throughput': 'tokens/second', 'latencies': 'milliseconds'},
            'speculative_decoding': {'acceptance_rate': None, 'reason': 'measurement not completed'},
        }

    def clean(self, value):
        text = str(value)
        for secret in self.redactions:
            if secret:
                text = text.replace(secret, '[REDACTED]')
        return text

    def set_state(self, state, error=None):
        with self.lock:
            self.state = state
            if error is not None:
                self.error = self.clean(error)
        self.log(f'{state}: {self.clean(error) if error is not None else ""}')

    def log(self, message):
        try:
            with (self.directory / 'worker.log').open('a', encoding='utf-8') as stream:
                stream.write(f'{utc_now()} {self.clean(message)}\n')
        except OSError:
            pass

    def remaining(self):
        remaining = float(self.config['deadline']) - time.time()
        if remaining <= 0:
            raise TimeoutError('overall deadline expired')
        return remaining

    def validate(self):
        if not self.secret or not self.pod_id or not self.api_key:
            raise RuntimeError('missing BENCH_SECRET, RUNPOD_POD_ID or Pod-scoped RUNPOD_API_KEY; watchdog unavailable')
        deadline = self.config.get('deadline')
        if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
            raise ValueError('deadline must be a finite epoch timestamp')
        for key in ('prompts', 'warmups', 'input_len', 'output_len', 'concurrency'):
            if type(self.config.get(key)) is not int or self.config[key] <= 0:
                raise ValueError(f'{key} must be a positive integer')
        if type(self.config.get('seed')) is not int or self.config['seed'] < 0:
            raise ValueError('seed must be a nonnegative integer')
        if type(self.config.get('profile')) is not bool:
            raise ValueError('profile must be a boolean')
        if self.config.get('gpu') not in ('AMD Instinct MI300X OAM', 'AMD Instinct MI350 OAM'):
            raise ValueError('a supported GPU is required')
        if not self.config.get('run_id') or not self.config.get('image'):
            raise ValueError('run_id and image are required')
        self.remaining()

    def watchdog(self):
        self.watchdog_started.set()
        delay = float(self.config['deadline']) - time.time()
        if delay > 0:
            time.sleep(delay)
        self.set_state('failed', 'overall deadline expired; watchdog terminating Pod')
        if self.server_process and self.server_process.poll() is None:
            try:
                self.server_process.terminate()
            except OSError:
                pass
        url = 'https://api.runpod.io/v2/pods/' + quote(self.pod_id, safe='')
        while True:
            try:
                req = Request(url, method='DELETE', headers={'Authorization': 'Bearer ' + self.api_key})
                with urlopen(req, timeout=10) as response:
                    if 200 <= response.status < 300:
                        return
            except HTTPError as exc:
                if exc.code not in (408, 429) and exc.code < 500:
                    self.set_state('failed', f'watchdog termination rejected (HTTP {exc.code}); local cleanup required')
                    return
                try:
                    delay = min(max(float(exc.headers.get('Retry-After', '5')), 1), 60)
                except (ValueError, TypeError, AttributeError):
                    delay = 5
                time.sleep(delay)
                continue
            except (OSError, URLError):
                pass
            time.sleep(5)

    def check_compatibility(self):
        versions = self.manifest['versions']
        for name in ('vllm', 'torch', 'transformers'):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                raise RuntimeError(f'{name} is not installed in the selected image') from None
        for name, minimum in (('vllm', (0, 17, 0)), ('transformers', (5, 8, 0))):
            if not version_at_least(versions[name], minimum):
                raise RuntimeError(f'{name} {versions[name]} does not meet >= {".".join(map(str, minimum))}')
        try:
            import torch
            versions['rocm'] = torch.version.hip or UNAVAILABLE
            if not torch.version.hip or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError('one ROCm GPU must be visible to PyTorch')
            name = torch.cuda.get_device_name(0)
            expected = {'AMD Instinct MI300X OAM': 'mi300x', 'AMD Instinct MI350 OAM': 'mi350x'}.get(self.manifest['gpu']['type'])
            if not expected or expected not in name.lower():
                raise RuntimeError(f"expected {self.manifest['gpu']['type']}, found {name}")
            self.manifest['gpu']['detected_name'] = name
        except ImportError as exc:
            raise RuntimeError('PyTorch is unavailable in the selected image') from exc
        image = str(self.config['image'])
        if '@sha256:' in image:
            self.manifest['image']['digest'] = image.rsplit('@', 1)[1]

    def serve_command(self):
        cmd = ['vllm', 'serve', MODEL, '--host', '127.0.0.1', '--port', '8000',
               '--tensor-parallel-size', '1', '--enable-auto-tool-choice',
               '--tool-call-parser', 'qwen3_coder', '--reasoning-parser', 'qwen3',
               '--speculative-config', json.dumps({'method': 'mtp', 'num_speculative_tokens': 3}),
               '--language-model-only']
        if self.config['profile']:
            cmd += ['--profiler-config', json.dumps({'profiler': 'torch',
                    'torch_profiler_dir': str(self.directory / 'profile'),
                    'torch_profiler_use_gzip': True})]
        return cmd

    def benchmark_command(self, count, filename):
        c = self.config
        profile_workload = filename == 'profile-benchmark.json'
        input_len = min(c['input_len'], 128) if profile_workload else c['input_len']
        output_len = min(c['output_len'], 32) if profile_workload else c['output_len']
        concurrency = min(c['concurrency'], 2) if profile_workload else c['concurrency']
        return ['vllm', 'bench', 'serve', '--backend', 'openai-chat', '--host', '127.0.0.1',
                '--port', '8000', '--endpoint', '/v1/chat/completions', '--model', MODEL,
                '--dataset-name', 'random', '--num-prompts', str(count),
                '--random-input-len', str(input_len), '--random-output-len', str(output_len),
                '--random-range-ratio', '0', '--max-concurrency', str(concurrency),
                '--seed', str(c['seed']), '--percentile-metrics', 'ttft,tpot,itl,e2el',
                '--metric-percentiles', '50,95,99', '--save-result', '--result-dir', str(self.directory),
                '--result-filename', filename, '--disable-tqdm']

    def child_env(self):
        env = os.environ.copy()
        for name in ('BENCH_CONFIG', 'BENCH_SECRET', 'RUNPOD_API_KEY', 'RUNPOD_POD_ID'):
            env.pop(name, None)
        env['VLLM_USE_RUST_FRONTEND'] = '1'
        if self.config['profile']:
            env['VLLM_SERVER_DEV_MODE'] = '1'
        return env

    def request(self, path, method='GET', timeout=10):
        request = Request('http://127.0.0.1:8000' + path, method=method)
        with urlopen(request, timeout=min(timeout, self.remaining())) as response:
            return response.read()

    def wait_ready(self):
        while True:
            self.remaining()
            if self.server_process.poll() is not None:
                raise RuntimeError('vLLM exited before readiness; inspect server.log')
            try:
                data = json.loads(self.request('/v1/models', timeout=3))
                if any(item.get('id') == MODEL for item in data.get('data', [])):
                    return
            except (OSError, URLError, ValueError, TypeError):
                pass
            time.sleep(min(2, self.remaining()))

    def execute_benchmark(self, count, filename, logname):
        command = self.benchmark_command(count, filename)
        with (self.directory / logname).open('ab') as log:
            try:
                process = subprocess.run(command, env=self.child_env(), stdout=log, stderr=subprocess.STDOUT,
                                         timeout=self.remaining(), check=False)
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(f'{logname} exceeded the overall deadline') from exc
        if process.returncode:
            raise RuntimeError(f'{logname} failed with exit code {process.returncode}')
        result_path = self.directory / filename
        try:
            result = json.loads(result_path.read_text())
        except (OSError, ValueError) as exc:
            raise RuntimeError(f'{filename} missing or invalid') from exc
        if filename == 'benchmark.json':
            self.record_benchmark(result)
        completed = result.get('completed')
        failed = result.get('failed', count - completed if type(completed) is int else None)
        if type(completed) is not int or type(failed) is not int or failed != 0 or completed != count:
            raise RuntimeError(f'{filename}: benchmark requests failed or counts are incomplete')
        if filename == 'benchmark.json':
            for key in ('duration', 'total_input_tokens', 'total_output_tokens',
                        'request_throughput', 'output_throughput'):
                value = result.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise RuntimeError(f'{filename}: missing or invalid measured {key}')
            if result['duration'] <= 0:
                raise RuntimeError(f'{filename}: invalid benchmark duration')
        return result


    def record_benchmark(self, result):
        self.manifest['benchmark'] = {key: result.get(key, UNAVAILABLE) for key in
            ('duration', 'completed', 'failed', 'total_input_tokens', 'total_output_tokens',
             'request_throughput', 'output_throughput')}
        for metric in ('ttft', 'tpot', 'itl', 'e2el'):
            for statistic in ('mean', 'median', 'p50', 'p95', 'p99'):
                key = f'{statistic}_{metric}_ms'
                self.manifest['benchmark'][key] = result.get(key, UNAVAILABLE)

    def record_revision(self):
        try:
            from huggingface_hub import try_to_load_from_cache
            path = try_to_load_from_cache(MODEL, 'config.json')
            if isinstance(path, str):
                parts = Path(path).parts
                if 'snapshots' in parts and parts.index('snapshots') + 1 < len(parts):
                    revision = parts[parts.index('snapshots') + 1]
                    if re.fullmatch(r'[0-9a-f]{40}', revision):
                        self.manifest['model']['revision'] = revision
        except (ImportError, OSError, ValueError):
            pass

    def profile(self):
        (self.directory / 'profile').mkdir(exist_ok=True)
        count = min(2, self.config['prompts'])
        self.manifest['profile'] = {'workload': 'separate profiled pass', 'prompts': count,
                                    'arguments': self.benchmark_command(count, 'profile-benchmark.json'),
                                    'gpu_activity_verified': False}
        started = False
        try:
            # A failed start can still have partially activated the profiler.
            started = True
            self.request('/start_profile', method='POST', timeout=30)
            self.execute_benchmark(count, 'profile-benchmark.json', 'profile.log')
        finally:
            if started:
                self.request('/stop_profile', method='POST', timeout=self.remaining())
        traces = list((self.directory / 'profile').glob('*.json.gz'))
        if not traces or not any(trace_has_gpu_activity(path) for path in traces):
            raise RuntimeError('ROCm profiler did not produce a trace with GPU kernel activity')
        self.manifest['profile']['gpu_activity_verified'] = True

    def run(self):
        status = 'failed'
        try:
            self.validate()
            thread = threading.Thread(target=self.watchdog, daemon=True, name='deadline-watchdog')
            thread.start()
            if not self.watchdog_started.wait(timeout=5) or not thread.is_alive():
                raise RuntimeError('deadline watchdog failed to start')
            self.check_compatibility()
            self.set_state('model_loading')
            serve = self.serve_command()
            self.manifest['serve_arguments'] = serve
            self.manifest['benchmark_arguments'] = self.benchmark_command(self.config['prompts'], 'benchmark.json')
            with (self.directory / 'server.log').open('ab') as log:
                self.server_process = subprocess.Popen(serve, stdout=log, stderr=subprocess.STDOUT, env=self.child_env())
                self.wait_ready()
                self.record_revision()
                self.set_state('warmup')
                self.execute_benchmark(self.config['warmups'], 'warmup.json', 'warmup.log')
                self.set_state('benchmark')
                before = self.request('/metrics').decode('utf-8', 'replace')
                (self.directory / 'metrics-before.prom').write_text(before)
                try:
                    result = self.execute_benchmark(self.config['prompts'], 'benchmark.json', 'benchmark.log')
                finally:
                    after = self.request('/metrics').decode('utf-8', 'replace')
                    (self.directory / 'metrics-after.prom').write_text(after)
                    self.manifest['speculative_decoding'] = speculative_deltas(before, after)
                if self.config['profile']:
                    self.set_state('profiling')
                    self.profile()
                self.remaining()
                status = 'completed'
        except Exception as exc:
            self.error = self.clean(f'{type(exc).__name__}: {exc}')
            self.log(self.error)
        finally:
            if self.server_process and self.server_process.poll() is None:
                self.server_process.terminate()
                try:
                    self.server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.server_process.kill()
                    self.server_process.wait()
            try:
                self.finalize(status)
            except Exception as exc:
                self.set_state('failed', f'artifact creation failed: {self.clean(exc)}')

    def finalize(self, status):
        self.manifest.update(status=status, ended_at=utc_now(), error=self.clean(self.error) if self.error else None)
        self.log(f'finalizing {status}')
        # Include only explicit output names, never caches, weights or arbitrary paths.
        paths = [self.directory / name for name in ('benchmark.json', 'warmup.json', 'profile-benchmark.json',
                 'metrics-before.prom', 'metrics-after.prom', 'server.log', 'worker.log', 'warmup.log', 'benchmark.log', 'profile.log')]
        profile_dir = self.directory / 'profile'
        if profile_dir.is_dir():
            paths.extend(p for p in profile_dir.iterdir() if p.is_file() and
                         (p.name.endswith('.json.gz') or p.name.endswith('.txt')))
        # Secret-bearing native results/traces are omitted, never modified or leaked.
        safe = []
        for path in paths:
            if not path.is_file() or path.is_symlink():
                continue
            if path.suffix == '.log' or path.suffix == '.txt':
                safe.append(path)
                continue
            opener = gzip.open if path.name.endswith('.gz') else open
            overlap = max((len(s.encode()) for s in self.redactions), default=1) - 1
            tail = b''
            with opener(path, 'rb') as stream:
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        safe.append(path)
                        break
                    data = tail + chunk
                    if any(s.encode() in data for s in self.redactions if s):
                        self.error = 'artifact output contained a secret and was omitted'
                        status = 'failed'
                        self.manifest.update(status=status, error=self.error)
                        break
                    tail = data[-overlap:] if overlap else b''
        manifest_bytes = json.dumps(self.manifest, indent=2, default=str).encode()
        manifest_bytes = self.clean(manifest_bytes.decode()).encode()
        tmp = self.directory / 'artifact.tmp'
        with tarfile.open(tmp, 'w:gz') as archive:
            import io
            info = tarfile.TarInfo('manifest.json')
            info.size = len(manifest_bytes)
            archive.addfile(info, io.BytesIO(manifest_bytes))
            for path in safe:
                name = str(path.relative_to(self.directory))
                if path.suffix in ('.log', '.txt'):
                    with path.open('rb') as stream:
                        stream.seek(0, 2)
                        stream.seek(max(0, stream.tell() - LOG_LIMIT))
                        data = self.clean(stream.read().decode('utf-8', 'replace')).encode()
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
                else:
                    archive.add(path, arcname=name, recursive=False)
        digest = hashlib.sha256()
        with tmp.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        if status == 'completed' and time.time() >= float(self.config['deadline']):
            raise TimeoutError('overall deadline expired while creating artifact')
        tmp.replace(self.artifact_path)
        with self.lock:
            self.artifact_size = self.artifact_path.stat().st_size
            self.artifact_sha256 = digest.hexdigest()
            self.state = 'done' if status == 'completed' else status


def make_server(worker, host='0.0.0.0', port=8765):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def reply(self, status, body, content_type='application/json'):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            authorization = self.headers.get('Authorization', '')
            if not worker.secret or not hmac.compare_digest(authorization.encode(), ('Bearer ' + worker.secret).encode()):
                self.reply(401, b'{"error":"unauthorized"}')
                return
            url = urlsplit(self.path)
            if url.path == '/status' and not url.query:
                with worker.lock:
                    data = {'state': worker.state, 'error': worker.error, 'artifact_size': worker.artifact_size,
                            'artifact_sha256': worker.artifact_sha256}
                self.reply(200, json.dumps(data).encode())
            elif url.path == '/artifact':
                try:
                    params = parse_qs(url.query, strict_parsing=True)
                    if set(params) != {'offset', 'size'} or any(len(v) != 1 for v in params.values()):
                        raise ValueError()
                    offset, size = int(params['offset'][0]), int(params['size'][0])
                    if offset < 0 or size <= 0 or size > MAX_CHUNK:
                        raise ValueError()
                    with worker.lock:
                        artifact_size = worker.artifact_size
                    if artifact_size is None:
                        self.reply(409, b'{"error":"artifact not ready"}')
                        return
                    if offset > artifact_size:
                        raise ValueError()
                    with worker.artifact_path.open('rb') as stream:
                        stream.seek(offset)
                        data = stream.read(min(size, artifact_size - offset))
                    self.reply(200, data, 'application/octet-stream')
                except (ValueError, OSError):
                    self.reply(400, b'{"error":"invalid artifact range"}')
            else:
                self.reply(404, b'{"error":"not found"}')

        def reject_method(self):
            authorization = self.headers.get('Authorization', '')
            if not worker.secret or not hmac.compare_digest(authorization.encode(), ('Bearer ' + worker.secret).encode()):
                self.reply(401, b'{"error":"unauthorized"}')
            else:
                self.reply(405, b'{"error":"method not allowed"}')

        do_POST = do_PUT = do_DELETE = do_PATCH = reject_method
    return ThreadingHTTPServer((host, port), Handler)


def main():
    try:
        config = json.loads(os.environ.get('BENCH_CONFIG', '{}'))
    except ValueError:
        config = {}
    if not isinstance(config, dict):
        config = {}
    directory = tempfile.mkdtemp(prefix='runpod-benchmark-')
    worker = Worker(config, os.environ.get('BENCH_SECRET', ''), os.environ.get('RUNPOD_POD_ID', ''),
                    os.environ.get('RUNPOD_API_KEY', ''), directory)
    server = make_server(worker)
    threading.Thread(target=worker.run, daemon=True, name='benchmark').start()
    server.serve_forever()


if __name__ == '__main__':
    main()
