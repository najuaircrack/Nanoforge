"""Cancellable trusted-callable processes, with process-tree resource settlement.

This is not a filesystem/network sandbox. Registered Python callables retain the
user's permissions. Windows Job Objects (or POSIX process groups) bound and settle
their process trees. Callables are serialized only from a trusted registration.
"""

from __future__ import annotations

import os
import signal
import time
from multiprocessing import get_context
from typing import Any

from nanoforge.agent.protocol import CancellationToken, ToolContext, _bounded, json_bytes


class ExecutionCancelled(RuntimeError):
    pass


class IsolationError(RuntimeError):
    pass


def _worker_main(connection: Any, payload: bytes, cancelled: Any, memory_bytes: int | None) -> None:
    # Do not inherit ambient API credentials into ordinary local tool workers.
    # This is defence in depth; process isolation does not restrict filesystem reads.
    for name in list(os.environ):
        if any(marker in name.upper() for marker in ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "CREDENTIAL", "AUTHORIZATION")):
            os.environ.pop(name, None)
    if os.name != "nt":
        os.setsid()
    connection.send(("ready", None))
    if connection.recv() != "start":
        return
    try:
        if memory_bytes is not None and os.name != "nt":
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        import cloudpickle
        fn, factory = cloudpickle.loads(payload)
        if factory:
            fn = fn()
        connection.send(("loaded", None))
        while True:
            command = connection.recv()
            if command is None:
                break
            try:
                token = CancellationToken(cancelled)
                token.raise_if_cancelled()
                args, kwargs = command.get("args", ()), command.get("kwargs", {})
                if command.get("context") is not None:
                    context = command["context"]
                    kwargs["context"] = ToolContext(token, context["deadline"], context["max_output_bytes"])
                if command.get("mode") == "provider":
                    result = fn.complete(args[0], cancellation=token)
                else:
                    result = fn(*args, **kwargs)
                token.raise_if_cancelled()
                if command.get("bound_json"):
                    original_size = len(json_bytes(result))
                    result = {"output": _bounded(result, command["max_output_bytes"]), "truncated": original_size > command["max_output_bytes"]}
                encoded = cloudpickle.dumps(result)
                if len(encoded) > command.get("max_message_bytes", 1024 * 1024):
                    raise ValueError("worker response exceeds transport limit")
                connection.send(("ok", encoded))
            except BaseException as exc:
                connection.send(("error", f"{type(exc).__name__}: {exc}"[:4096]))
    except BaseException as exc:
        try:
            connection.send(("error", f"{type(exc).__name__}: {exc}"[:4096]))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def _windows_job(process: Any, memory_bytes: int | None) -> Any:
    import ctypes
    from ctypes import wintypes

    class Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class IO(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount", "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class Extended(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", IO),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.SetInformationJobObject.restype = wintypes.BOOL
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.CreateJobObjectW(None, None)
    if not handle:
        raise IsolationError(f"CreateJobObject failed: {ctypes.get_last_error()}")
    limits = Extended()
    limits.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
    if memory_bytes is not None:
        limits.BasicLimitInformation.LimitFlags |= 0x200  # JOB_MEMORY
        limits.JobMemoryLimit = memory_bytes
    if not kernel.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)) or not kernel.AssignProcessToJobObject(handle, wintypes.HANDLE(process.sentinel)):
        error = ctypes.get_last_error()
        kernel.CloseHandle(handle)
        raise IsolationError(f"Could not enforce Windows Job Object: {error}")
    return kernel, handle


class IsolatedWorker:
    """One persistent worker; close always settles its process tree."""

    def __init__(self, fn: Any, *, deadline: float, cancellation: CancellationToken, max_memory_bytes: int | None = None, factory: bool = False):
        self.cancellation = cancellation
        self.closed = False
        self.effects_settled = False
        self._job = None
        self._process = None
        self._connection = None
        self._cancelled = None
        if cancellation.cancelled:
            raise ExecutionCancelled(cancellation.reason or "cancelled")
        if deadline <= time.monotonic():
            raise TimeoutError("execution deadline exceeded")
        try:
            import cloudpickle
        except ImportError as exc:
            raise IsolationError("trusted process isolation requires cloudpickle; no thread fallback is permitted") from exc
        payload = cloudpickle.dumps((fn, factory))
        if len(payload) > 64 * 1024 * 1024:
            raise IsolationError("callable payload exceeds 64 MiB; use a worker factory that loads the model in the child")
        context = get_context("spawn")
        parent, child = context.Pipe()
        self._connection = parent
        self._cancelled = context.Event()
        self._process = context.Process(target=_worker_main, args=(child, payload, self._cancelled, max_memory_bytes), daemon=False)
        try:
            self._process.start()
            child.close()
            kind, _ = self._receive(deadline)
            if kind != "ready":
                raise IsolationError("worker did not reach process setup")
            if os.name == "nt":
                self._job = _windows_job(self._process, max_memory_bytes)
            parent.send("start")
            kind, value = self._receive(deadline)
            if kind != "loaded":
                raise IsolationError(value or "worker initialization failed")
        except BaseException:
            child.close()
            self.close()
            raise

    def _receive(self, deadline: float) -> tuple[str, Any]:
        while True:
            if self.cancellation.cancelled:
                self._cancelled.set()
                raise ExecutionCancelled(self.cancellation.reason or "cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._cancelled.set()
                raise TimeoutError("execution deadline exceeded")
            if self._connection.poll(min(.02, remaining)):
                try:
                    result = self._connection.recv()
                except (EOFError, OSError) as exc:
                    raise IsolationError("worker exited without a result (possibly resource limit)") from exc
                if self.cancellation.cancelled:
                    raise ExecutionCancelled(self.cancellation.reason or "cancelled")
                if time.monotonic() >= deadline:
                    raise TimeoutError("execution deadline exceeded")
                return result
            if not self._process.is_alive():
                raise IsolationError(f"worker exited without a result, exit={self._process.exitcode} (possibly resource limit)")

    def call(self, *args: Any, deadline: float, kwargs: dict[str, Any] | None = None, context: ToolContext | None = None, mode: str = "call", bound_json: bool = False, max_output_bytes: int = 65536, max_message_bytes: int = 1024 * 1024) -> Any:
        if self.closed:
            raise IsolationError("worker is closed")
        command = {"args": args, "kwargs": kwargs or {}, "mode": mode, "bound_json": bound_json,
                   "max_output_bytes": max_output_bytes, "max_message_bytes": max_message_bytes,
                   "context": None if context is None else {"deadline": context.deadline, "max_output_bytes": context.max_output_bytes}}
        try:
            self._connection.send(command)
            kind, value = self._receive(deadline)
            if kind != "ok":
                raise IsolationError(value or "worker call failed")
            import cloudpickle
            return cloudpickle.loads(value)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self._cancelled is not None:
            self._cancelled.set()
        process = self._process
        if self._job is not None:
            kernel, handle = self._job
            kernel.CloseHandle(handle)  # includes every child and descendant in the Job
            self._job = None
        elif process is not None and process.pid is not None:
            if os.name != "nt":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if process.is_alive():
                process.terminate()
        if process is not None and process.pid is not None:
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
            self.effects_settled = not process.is_alive()
            if not self.effects_settled:
                raise IsolationError("worker termination could not be confirmed")
            process.close()
        else:
            self.effects_settled = True
        if self._connection is not None:
            self._connection.close()

