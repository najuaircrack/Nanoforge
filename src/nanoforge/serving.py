"""Shared bounded serving policy, workspace paths and local job lifecycle."""

from __future__ import annotations

import argparse
import asyncio
import collections
import hmac
import ipaddress
import json
import math
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Iterator
from urllib.parse import urlparse


def is_loopback(host: str) -> bool:
    if host.lower() == 'localhost':
        return True
    try:
        return ipaddress.ip_address(host.strip('[]')).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class ServingPolicy:
    host: str = '127.0.0.1'
    api_key: str | None = None
    allowed_origins: tuple[str, ...] = ()
    max_request_bytes: int = 2 * 1024 * 1024
    max_concurrency: int = 2
    request_timeout: float = 120.0
    max_new_tokens: int = 4096
    max_output_bytes: int = 1024 * 1024

    def __post_init__(self):
        if not is_loopback(self.host) and (self.api_key is None or len(self.api_key) < 16):
            raise ValueError('Non-loopback serving requires an API key of at least 16 characters.')
        if self.api_key is not None and len(self.api_key) < 16:
            raise ValueError('API keys must contain at least 16 characters.')
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in (self.max_request_bytes, self.max_concurrency, self.max_new_tokens, self.max_output_bytes)) or isinstance(self.request_timeout, bool) or not math.isfinite(self.request_timeout) or self.request_timeout <= 0:
            raise ValueError('Serving limits must be positive.')


class SecurityMiddleware:
    """ASGI body/auth/origin limits; no credentials appear in responses or logs."""

    def __init__(self, app, policy: ServingPolicy):
        self.app, self.policy = app, policy
        self._requests = threading.BoundedSemaphore(max(4, policy.max_concurrency * 4))

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        if not self._requests.acquire(blocking=False):
            return await self._reject(send, 429, 'Concurrent HTTP request limit reached.')
        try:
            return await self._http(scope, receive, send)
        finally:
            self._requests.release()

    async def _http(self, scope, receive, send):
        headers = {key.decode('latin1').lower(): value.decode('latin1') for key, value in scope['headers']}
        origin = headers.get('origin')
        if origin:
            parsed = urlparse(origin)
            same_host = parsed.netloc.lower() == headers.get('host', '').lower()
            local_origin = is_loopback(parsed.hostname or '') and is_loopback(self.policy.host) and same_host
            if origin not in self.policy.allowed_origins and not local_origin:
                return await self._reject(send, 403, 'Origin is not allowed.')
        protected = scope['path'].startswith(('/api/', '/v1/', '/stream/')) or scope['path'] in ('/metrics', '/health')
        if self.policy.api_key and protected:
            expected = 'Bearer ' + self.policy.api_key
            if not hmac.compare_digest(headers.get('authorization', ''), expected):
                return await self._reject(send, 401, 'Authentication required.')
        try:
            if int(headers.get('content-length', '0')) > self.policy.max_request_bytes:
                return await self._reject(send, 413, 'Request exceeds the configured byte limit.')
        except ValueError:
            return await self._reject(send, 400, 'Invalid Content-Length.')
        body = bytearray()
        deadline = time.monotonic() + self.policy.request_timeout
        while True:
            try:
                message = await asyncio.wait_for(receive(), timeout=max(.001, deadline - time.monotonic()))
            except asyncio.TimeoutError:
                return await self._reject(send, 408, 'Request body deadline exceeded.')
            if message['type'] == 'http.disconnect':
                return
            body.extend(message.get('body', b''))
            if len(body) > self.policy.max_request_bytes:
                return await self._reject(send, 413, 'Request exceeds the configured byte limit.')
            if not message.get('more_body', False):
                break
        delivered = False

        async def bounded_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {'type': 'http.request', 'body': bytes(body), 'more_body': False}
            return await receive()

        await self.app(scope, bounded_receive, send)

    @staticmethod
    async def _reject(send, status, message):
        body = json.dumps({'ok': False, 'error': message}).encode()
        await send({'type': 'http.response.start', 'status': status, 'headers': [(b'content-type', b'application/json')]})
        await send({'type': 'http.response.body', 'body': body})


class WorkspacePaths:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise NotADirectoryError(root)

    def resolve(self, value: str | Path, *, allow_absolute: bool = True) -> Path:
        text = str(value)
        windows = PureWindowsPath(text)
        path = Path(text)
        if not allow_absolute and (path.is_absolute() or windows.drive or windows.is_absolute()):
            raise ValueError('Absolute, drive and UNC paths are not allowed.')
        if '..' in path.parts or '..' in windows.parts:
            raise ValueError('Parent traversal is not allowed.')
        resolved = (path if path.is_absolute() else self.root / path).resolve()
        if not resolved.is_relative_to(self.root):
            raise PermissionError('Path is outside the selected workspace.')
        return resolved

    def relative(self, path: str | Path) -> str:
        return self.resolve(path).relative_to(self.root).as_posix()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def count_tokens(engine: Any, text: str) -> int | None:
    if hasattr(engine, 'count_tokens'):
        return int(engine.count_tokens(text))
    tokenizer = getattr(engine, 'tokenizer', None)
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_bos=False, add_eos=False))
        except TypeError:
            return len(tokenizer.encode(text, add_special_tokens=False))
    model = getattr(engine, 'model', None)
    if model is not None and hasattr(model, 'tokenize'):
        return len(model.tokenize(text.encode('utf-8'), add_bos=False))
    return None


class EngineRunner:
    """Bound calls without releasing capacity for a timed-out still-running worker.

    Python cannot terminate an arbitrary engine thread. The caller's deadline is
    enforced; capacity stays occupied until actual engine work settles. Engines
    with interrupt support additionally receive the cancellation event.
    """

    def __init__(self, policy: ServingPolicy):
        self.policy = policy
        self._slots = threading.BoundedSemaphore(policy.max_concurrency)
        self._lock = threading.Lock()
        self.active = 0
        self._cancel: set[threading.Event] = set()
        self._engine_locks: dict[int, list] = {}

    def stream(self, engine: Any, prompt: str, max_new_tokens: int, sampling=None, **kwargs) -> Iterator[str]:
        if max_new_tokens > self.policy.max_new_tokens:
            raise ValueError('Generation token limit exceeds server policy.')
        if not self._slots.acquire(blocking=False):
            raise RuntimeError('Model is busy; configured concurrency limit reached.')
        cancel = threading.Event()
        result: queue.Queue = queue.Queue(maxsize=16)
        with self._lock:
            self.active += 1
            self._cancel.add(cancel)
            execution = self._engine_locks.setdefault(id(engine), [threading.Lock(), 0])
            execution[1] += 1

        def put(value):
            while not cancel.is_set():
                try:
                    result.put(value, timeout=.05)
                    return
                except queue.Full:
                    pass

        def run():
            acquired = False
            try:
                while not cancel.is_set():
                    if execution[0].acquire(timeout=.05):
                        acquired = True
                        break
                if not acquired:
                    return
                import inspect
                method = kwargs.pop("stream_fn", None) or engine.stream
                arguments = inspect.signature(method).parameters
                if "interrupt" in arguments or any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in arguments.values()):
                    kwargs["interrupt"] = cancel.is_set
                if "max_wall_seconds" in arguments:
                    kwargs["max_wall_seconds"] = self.policy.request_timeout
                output_bytes = 0
                for piece in method(prompt, max_new_tokens, sampling, **kwargs):
                    if cancel.is_set():
                        break
                    if not isinstance(piece, str):
                        raise TypeError('Engine stream must yield strings.')
                    output_bytes += len(piece.encode('utf-8'))
                    if output_bytes > self.policy.max_output_bytes:
                        raise ValueError('Model output exceeds server byte budget.')
                    put(('piece', piece))
                put(('done', None))
            except Exception as exc:
                put(('error', exc))
            finally:
                if acquired:
                    execution[0].release()
                with self._lock:
                    self.active -= 1
                    self._cancel.discard(cancel)
                    execution[1] -= 1
                    if not execution[1]:
                        self._engine_locks.pop(id(engine), None)
                self._slots.release()

        threading.Thread(target=run, daemon=True, name='latticekin-inference').start()
        deadline = time.monotonic() + self.policy.request_timeout
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Model request exceeded its deadline.')
                try:
                    kind, value = result.get(timeout=remaining)
                except queue.Empty as exc:
                    raise TimeoutError('Model request exceeded its deadline.') from exc
                if kind == 'done':
                    return
                if kind == 'error':
                    raise value
                yield value
        finally:
            cancel.set()

    def close(self):
        with self._lock:
            for cancel in self._cancel:
                cancel.set()


def terminate_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


class JobManager:
    def __init__(self, root: Path, *, max_log_lines: int = 2000, max_jobs: int = 64, max_concurrency: int = 2, deadline_seconds: float = 86400):
        self.root = Path(root).resolve()
        self.max_log_lines, self.max_jobs = max_log_lines, max_jobs
        self.max_concurrency, self.deadline_seconds = max_concurrency, deadline_seconds
        self._jobs: dict[str, dict] = {}
        self._lock = threading.RLock()
        self._closed = False

    def _create(self, job_type: str, run_dir: str | None = None):
        with self._lock:
            if self._closed:
                raise RuntimeError('Job manager is closed.')
            if sum(not job['done'] for job in self._jobs.values()) >= self.max_concurrency:
                raise RuntimeError('Job concurrency limit reached.')
            while len(self._jobs) >= self.max_jobs:
                finished = next((key for key, job in self._jobs.items() if job['done']), None)
                if finished is None:
                    raise RuntimeError('Job retention limit reached.')
                self._jobs.pop(finished)
            job_id = uuid.uuid4().hex
            entry = {'job_id': job_id, 'type': job_type, 'run_dir': run_dir, 'lines': collections.deque(maxlen=self.max_log_lines), 'log_lines_seen': 0, 'logs_truncated': False, 'done': False, 'code': None, 'error': None, 'result': None, 'created': time.time(), 'cancel': threading.Event(), 'proc': None}
            self._jobs[job_id] = entry
            return job_id, entry

    def start_process(self, command: list[str], *, job_type: str, run_dir: str | None = None) -> str:
        job_id, entry = self._create(job_type, run_dir)
        try:
            if len(command) >= 3 and command[1] == '-m' and command[2] in {'nanoforge', 'nanoforge.serving'}:
                worker = str(Path(__file__).resolve())
                tail = ['run-cli', '--', *command[3:]] if command[2] == 'nanoforge' else command[3:]
                command = [command[0], '-I', worker, *tail]
            flags = {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt' else {'start_new_session': True}
            process = subprocess.Popen(command, cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', env={**os.environ, 'PYTHONUNBUFFERED': '1'}, **flags)
            entry['proc'] = process
        except Exception:
            with self._lock:
                self._jobs.pop(job_id, None)
            raise

        def read():
            try:
                assert process.stdout is not None
                while True:
                    line = process.stdout.readline(8192)
                    if not line:
                        break
                    with self._lock:
                        entry['log_lines_seen'] += 1
                        entry['logs_truncated'] = entry['log_lines_seen'] > self.max_log_lines
                        entry['lines'].append(line.rstrip()[:8192])
                process.wait()
                entry['code'] = process.returncode
            except Exception as exc:
                entry['error'], entry['code'] = str(exc), -1
                terminate_process_tree(process)
            finally:
                if process.stdout:
                    process.stdout.close()
                entry['done'] = True

        def watch():
            deadline = time.monotonic() + self.deadline_seconds
            while not entry['done']:
                if entry['cancel'].wait(.1) or time.monotonic() >= deadline:
                    entry['error'] = 'Cancelled.' if entry['cancel'].is_set() else 'Job deadline exceeded.'
                    terminate_process_tree(process)
                    return

        threading.Thread(target=read, daemon=True, name='latticekin-job-log').start()
        threading.Thread(target=watch, daemon=True, name='latticekin-job-budget').start()
        return job_id

    def start_callable(self, callback: Callable[[threading.Event], Any], *, job_type: str, run_dir: str | None = None):
        job_id, entry = self._create(job_type, run_dir)
        def run():
            try:
                entry['result'], entry['code'] = callback(entry['cancel']), 0
            except Exception as exc:
                entry['error'], entry['code'] = str(exc), -1
            finally:
                entry['done'] = True
        threading.Thread(target=run, daemon=True, name='latticekin-background-run').start()
        return job_id

    def status(self, job_id: str) -> dict:
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry is None:
                raise KeyError(job_id)
            return {key: list(value) if key == 'lines' else value for key, value in entry.items() if key not in ('cancel', 'proc')}

    def list(self):
        with self._lock:
            return [self.status(key) for key in reversed(self._jobs)]

    def cancel(self, job_id: str):
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            self._jobs[job_id]['cancel'].set()

    def close(self):
        with self._lock:
            self._closed = True
            jobs = list(self._jobs.values())
        for entry in jobs:
            entry['cancel'].set()
            if entry['proc']:
                terminate_process_tree(entry['proc'])


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'run-cli':
        # Import the installed/checkout framework, never a same-named module in
        # the selected (potentially untrusted) workspace. -I removes cwd and
        # PYTHONPATH from initial discovery; this path is derived from this file.
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        from nanoforge.cli import main as cli_main
        cli_main()
        return
    parser = argparse.ArgumentParser(description='Fixed dataset download worker.')
    sub = parser.add_subparsers(dest='command', required=True)
    download = sub.add_parser('download-dataset')
    download.add_argument('--source', required=True)
    download.add_argument('--out', required=True)
    download.add_argument('--config')
    args = parser.parse_args()
    from datasets import load_dataset
    dataset = load_dataset(args.source, args.config) if args.config else load_dataset(args.source)
    dataset.save_to_disk(args.out)
    print(json.dumps({'completed': True, 'splits': {key: len(value) for key, value in dataset.items()}}))


if __name__ == '__main__':
    main()
