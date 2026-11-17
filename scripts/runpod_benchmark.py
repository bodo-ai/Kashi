#!/usr/bin/env python3
"""Run the specified vLLM benchmark on a temporary Runpod GPU Pod."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_recovery as recovery

GPU_OPTIONS = ('AMD Instinct MI300X OAM', 'AMD Instinct MI350 OAM')
IMAGE = 'vllm/vllm-openai-rocm:latest'
# Keep the existing journal location so outstanding Pods remain recoverable.
STATE = Path.home() / '.local' / 'state' / 'runpod-mi300x-benchmark'


def positive(value):
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return n


def nonnegative(value):
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError('must be nonnegative')
    return n


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api-key', default=os.environ.get('RUNPOD_API_KEY'))
    parser.add_argument('--state-dir', type=Path, default=STATE)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run', help='create, benchmark and terminate one Pod')
    run.add_argument('--image', default=IMAGE)
    run.add_argument('--gpu', nargs='+', choices=GPU_OPTIONS, default=list(GPU_OPTIONS), help='GPU types in preference order')
    run.add_argument('--disk', type=positive, default=100)
    run.add_argument('--deadline', type=positive, default=90, help='overall deadline in minutes')
    run.add_argument('--data-center')
    run.add_argument('--wait-for-capacity', type=positive, metavar='MINUTES', help='poll locally for capacity before creating a Pod')
    run.add_argument('--output-dir', type=Path, default=Path('.'))
    run.add_argument('--prompts', type=positive, default=128)
    run.add_argument('--warmups', type=positive, default=5)
    run.add_argument('--input-len', type=positive, default=512)
    run.add_argument('--output-len', type=positive, default=256)
    run.add_argument('--concurrency', type=positive, default=8)
    run.add_argument('--seed', type=nonnegative, default=0)
    run.add_argument('--profile', action='store_true')
    sub.add_parser('recover', help='terminate journaled Pods and report unresolved records')
    attest = sub.add_parser('attest', help='clear an unresolved record after manual console verification')
    attest.add_argument('run_id')
    attest.add_argument('--name', required=True)
    return parser.parse_args(argv)


def download_artifact(get_chunk, size, digest, target):
    if not isinstance(size, int) or size <= 0 or len(digest) != 64:
        raise ValueError('invalid artifact metadata')
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.part')
    checksum = hashlib.sha256()
    try:
        with temporary.open('xb') as f:
            offset = 0
            while offset < size:
                requested = min(1024 * 1024, size - offset)
                chunk = get_chunk(offset, requested)
                if not chunk or len(chunk) > requested:
                    raise ValueError('invalid artifact chunk')
                f.write(chunk)
                checksum.update(chunk)
                offset += len(chunk)
        if checksum.hexdigest() != digest:
            raise ValueError('artifact checksum mismatch')
        os.replace(temporary, target)
        return target
    finally:
        temporary.unlink(missing_ok=True)


def worker_request(pod_id, secret, path, deadline):
    remaining = deadline - time.time()
    if remaining <= 0:
        raise TimeoutError('overall deadline reached')
    url = f'https://{pod_id}-8765.proxy.runpod.net{path}'
    req = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + secret, 'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=min(20, remaining)) as response:
        return response.read()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def report_unresolved(records):
    for r in records:
        print(f"Unconfirmed cleanup: name={r.get('name')} id={r.get('pod_id') or 'unknown'} account={r.get('account')} elapsed={max(0, int(time.time() - r.get('started_at', time.time())))}s error={r.get('last_error') or 'outcome unconfirmed'}. Terminate this exact Pod in https://www.runpod.io/console/pods; run recover, or attest after verifying termination.", file=sys.stderr)

def pod_body(name, image, disk, code, env, gpu_id, data_center=None):
    body = {'name': name, 'image': image, 'gpu': {'id': gpu_id, 'count': 1}, 'disk': disk,
            'entrypoint': ['python3', '-u', '-c'], 'cmd': [code], 'ports': ['8765/http'], 'env': env}
    if data_center:
        body['dataCenterIds'] = [data_center]
    return body


def wait_for_capacity(api, seconds, data_center, gpu_options):
    deadline = time.monotonic() + seconds
    while True:
        catalog = api.request('GET', '/catalog/gpus?include=AVAILABILITY&product=POD&count=1')
        by_id = {gpu.get('id'): gpu for gpu in catalog['gpus']}
        for gpu_id in gpu_options:
            gpu = by_id.get(gpu_id)
            if not gpu or gpu.get('availability') not in ('NONE', 'LOW', 'MEDIUM', 'HIGH'):
                raise RuntimeError(f'{gpu_id} capacity information is unavailable')
            availability = gpu['availability']
            if data_center and availability != 'NONE':
                centers = gpu.get('dataCenters')
                if not isinstance(centers, list):
                    raise RuntimeError('data center capacity information is unavailable')
                center = next((c for c in centers if c.get('id') == data_center), None)
                availability = center.get('availability') if center else 'NONE'
                if availability not in ('NONE', 'LOW', 'MEDIUM', 'HIGH'):
                    raise RuntimeError('data center capacity information is unavailable')
            if availability != 'NONE':
                print(f'Selected {gpu_id}: capacity {availability}; attempting placement.', flush=True)
                return gpu_id
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('capacity wait expired without creating a Pod')
        print('Waiting for preferred GPU capacity...', flush=True)
        time.sleep(min(30, remaining))


def main(argv=None):
    args = parse_args(argv)
    if not args.api_key:
        print('RUNPOD_API_KEY or --api-key is required', file=sys.stderr)
        return 1
    api = recovery.RunpodAPI(args.api_key)
    try:
        with recovery.exclusive_lock(args.state_dir / 'lock'):
            journal = recovery.Journal(args.state_dir / 'journal.json')
            account = api.request('GET', '/account')['id']
            if args.command == 'attest':
                journal.attest(args.run_id, args.name, account)
                return 0
            unresolved = recovery.reconcile(journal, api, account)
            if unresolved:
                report_unresolved(unresolved)
                return 1
            if args.command == 'recover':
                print('No unresolved journal records.')
                return 0
            gpu_id = wait_for_capacity(api, (args.wait_for_capacity or 0) * 60, args.data_center, args.gpu)
            return run_benchmark(args, api, journal, account, gpu_id)
    except (Exception, KeyboardInterrupt) as error:
        print(f'error: {type(error).__name__}: {error}', file=sys.stderr)
        return 1


def create(api, journal, account, name, deadline, body, run_id):
    journal.begin(run_id, name, deadline, account)
    journal.mark_sent(run_id)
    try:
        pod = api.request('POST', '/pods', body)
    except recovery.APIError as error:
        if error.status == 400:
            journal.remove(run_id)
            gpu_id = body.get('gpu', {}).get('id', 'unknown GPU')
            availability = 'unavailable'
            try:
                catalog = api.request('GET', '/catalog/gpus?include=AVAILABILITY&product=POD&count=1')
                for gpu in catalog['gpus']:
                    if gpu.get('id') == gpu_id and gpu.get('availability') in ('NONE', 'LOW', 'MEDIUM', 'HIGH'):
                        availability = gpu['availability']
                        break
            except (recovery.APIError, OSError, KeyError, TypeError, AttributeError):
                pass
            print(f'Current secure-cloud {gpu_id} Pod capacity: {availability}. '
                  'This is a later catalog snapshot and does not establish the cause of the HTTP 400; '
                  'availability may change and may differ by data center.', file=sys.stderr)
        raise
    pod_id = pod['id']
    journal.set_pod_id(run_id, pod_id)
    return pod


def cleanup(api, journal, run_id, name, account, pod_id):
    if not pod_id:
        report_unresolved(journal.records())
        return False
    try:
        if api.delete_pod(pod_id):
            journal.remove(run_id)
            print('Pod termination confirmed.')
            return True
    except Exception as error:
        print(f'Termination error: {error}', file=sys.stderr)
    report_unresolved(journal.records())
    return False




def run_benchmark(args, api, journal, account, gpu_id):
    run_id = secrets.token_hex(16)
    name = 'runpod-benchmark-' + run_id
    deadline = time.time() + args.deadline * 60
    secret = secrets.token_urlsafe(32)
    config = {'run_id': run_id, 'deadline': deadline, 'image': args.image, 'prompts': args.prompts,
              'warmups': args.warmups, 'input_len': args.input_len, 'output_len': args.output_len,
              'concurrency': args.concurrency, 'seed': args.seed, 'profile': args.profile,
              'data_center': args.data_center, 'gpu': gpu_id}
    worker = (Path(__file__).resolve().parent / 'runpod_worker.py').read_bytes()
    code = "import base64;exec(compile(base64.b64decode('" + base64.b64encode(worker).decode() + "'),'<worker>','exec'))"
    env = {'BENCH_SECRET': secret, 'BENCH_CONFIG': json.dumps(config)}
    if os.environ.get('HF_TOKEN'):
        env['HF_TOKEN'] = os.environ['HF_TOKEN']
    pod_id = None
    success = False
    downloaded = None
    downloaded_digest = None
    cleanup_ok = False
    start = time.time()
    rate = None
    try:
        print(f'Provisioning one {gpu_id} Pod...')
        body = pod_body(name, args.image, args.disk, code, env, gpu_id, args.data_center)
        pod = create(api, journal, account, name, deadline, body, run_id)
        pod_id = pod['id']
        rate = pod.get('cost')
        print(f'Pod {pod_id}; hourly rate: {rate if rate is not None else "unavailable"}.')
        previous = None
        state = None
        while time.time() < deadline:
            info = api.request('GET', '/pods/' + urllib.parse.quote(pod_id, safe=''))
            if info.get('status') in ('ERROR', 'EXITED', 'TERMINATED'):
                raise RuntimeError('Pod status: ' + str(info.get('status')))
            try:
                status = json.loads(worker_request(pod_id, secret, '/status', deadline))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                time.sleep(min(5, max(0, deadline - time.time())))
                continue
            state = status.get('state')
            if state != previous:
                print('Worker:', state, flush=True)
                previous = state
            if state in ('done', 'failed', 'error'):
                break
            time.sleep(min(5, max(0, deadline - time.time())))
        else:
            raise TimeoutError('overall deadline reached')
        if status.get('artifact_size') and status.get('artifact_sha256'):
            print('Downloading artifact...')
            partial = state != 'done'
            target = args.output_dir / (name + ('.partial.tar.gz' if partial else '.tar.gz'))
            downloaded = download_artifact(lambda offset, size: worker_request(pod_id, secret, f'/artifact?offset={offset}&size={size}', deadline), status['artifact_size'], status['artifact_sha256'], target)
            downloaded_digest = status['artifact_sha256']
            print('Artifact:', downloaded)
        if state != 'done':
            raise RuntimeError('worker failed: ' + str(status.get('error', 'unknown error')))
        if downloaded is None:
            raise RuntimeError('worker did not produce an artifact')
        success = True
    finally:
        print('Cleaning up Pod...', flush=True)
        cleanup_ok = cleanup(api, journal, run_id, name, account, pod_id)
        sidecar = args.output_dir / (name + '.json')
        save_json(sidecar, {'run_id': run_id, 'pod_id': pod_id, 'artifact': str(downloaded) if downloaded else None,
                            'sha256': downloaded_digest,
                            'cleanup_confirmed': cleanup_ok, 'benchmark_success': success,
                            'estimated_compute_cost': rate * ((time.time() - start) / 3600) if isinstance(rate, (int, float)) else None,
                            'cost_note': 'estimate excludes unknown charges and overrun'})
    return 0 if success and cleanup_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
