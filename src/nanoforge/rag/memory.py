"""Principal/scoped local memory, separate from generic training corpora."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    owner: str
    scope: str
    content: str
    provenance: dict[str, Any]
    created_at: float
    expires_at: float
    personal: bool
    consent: bool
    training_allowed: bool


class MemoryStore:
    """Owner selection is an authorized principal, not model-provided input.

    SQLite transactions bind writes and content-free audit events. Every read
    selects the principal and explicit scope, excludes expired facts, and never
    exports personal content to training without a second explicit selection.
    Deletion removes content from the logical store; secure disk erasure is not
    promised for filesystem/SSD snapshots or externally copied data.
    """

    def __init__(self, path: str | Path, *, owner: str,
                 clock: Callable[[], float] = time.time, max_record_bytes: int = 65536):
        if not isinstance(owner, str) or not owner.strip() or max_record_bytes < 1:
            raise ValueError("memory requires a principal and positive byte limit")
        self.path = Path(path)
        if self.path.is_symlink():
            raise ValueError("memory database cannot be a symlink")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.owner, self.clock, self.max_record_bytes = owner, clock, max_record_bytes
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS memory (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, scope TEXT NOT NULL,
                    content TEXT NOT NULL, provenance TEXT NOT NULL,
                    created REAL NOT NULL, expires REAL NOT NULL,
                    personal INTEGER NOT NULL, consent INTEGER NOT NULL,
                    training_allowed INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS memory_scope ON memory(owner,scope,expires);
                CREATE TABLE IF NOT EXISTS audit (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner TEXT NOT NULL, action TEXT NOT NULL, memory_id TEXT NOT NULL,
                    scope TEXT NOT NULL, timestamp REAL NOT NULL
                );
                PRAGMA user_version = 1;
            """)

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA secure_delete = ON")
        return connection

    def _scope(self, scope):
        if not isinstance(scope, str) or not scope.strip():
            raise ValueError("memory scope must be explicit nonempty text")
        return scope

    def _record(self, row):
        if row is None:
            return None
        return MemoryRecord(row["id"], row["owner"], row["scope"], row["content"],
                            json.loads(row["provenance"]), row["created"], row["expires"],
                            bool(row["personal"]), bool(row["consent"]), bool(row["training_allowed"]))

    def _audit(self, connection, action, memory_id, scope):
        connection.execute("INSERT INTO audit(owner,action,memory_id,scope,timestamp) VALUES(?,?,?,?,?)",
                           (self.owner, action, memory_id, scope, self.clock()))

    def put(self, content: str, *, scope: str, provenance: dict[str, Any],
            personal: bool = False, consent: bool = False, training_allowed: bool = False,
            retention_seconds: float = 30 * 86400) -> MemoryRecord:
        self._scope(scope)
        if not isinstance(content, str) or not content.strip() or len(content.encode("utf-8")) > self.max_record_bytes:
            raise ValueError("memory content is empty or exceeds byte limit")
        if not isinstance(provenance, dict) or not isinstance(provenance.get("source"), str) or not provenance["source"]:
            raise ValueError("memory requires source provenance")
        if personal and not consent:
            raise PermissionError("personal memory requires explicit consent")
        if training_allowed and not consent:
            raise PermissionError("training selection requires explicit consent")
        if isinstance(retention_seconds, bool) or not isinstance(retention_seconds, (int, float)) or not 0 < retention_seconds <= 10 * 365 * 86400:
            raise ValueError("retention must be positive and at most ten years")
        identifier, created = str(uuid.uuid4()), self.clock()
        serialized = json.dumps(provenance, sort_keys=True, allow_nan=False)
        if len(serialized.encode("utf-8")) > self.max_record_bytes:
            raise ValueError("provenance exceeds byte limit")
        record = MemoryRecord(identifier, self.owner, scope, content, provenance, created,
                              created + retention_seconds, bool(personal), bool(consent), bool(training_allowed))
        with self._connect() as connection:
            connection.execute("INSERT INTO memory VALUES(?,?,?,?,?,?,?,?,?,?)",
                               (identifier, self.owner, scope, content, serialized, created, record.expires_at,
                                int(personal), int(consent), int(training_allowed)))
            self._audit(connection, "created", identifier, scope)
        return record

    def get(self, memory_id: str, *, scope: str) -> MemoryRecord | None:
        self._scope(scope)
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM memory WHERE id=? AND owner=? AND scope=? AND expires>?",
                                     (memory_id, self.owner, scope, self.clock())).fetchone()
        return self._record(row)

    def retrieve(self, query: str, *, scope: str, max_records: int = 8, max_bytes: int = 8192) -> list[MemoryRecord]:
        self._scope(scope)
        if not isinstance(query, str) or len(query.encode("utf-8")) > self.max_record_bytes:
            raise ValueError("invalid memory query")
        if not 1 <= max_records <= 1000 or max_bytes < 1:
            raise ValueError("invalid memory retrieval budget")
        # This is explicit lexical matching, not semantic retrieval.
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM memory WHERE owner=? AND scope=? AND expires>? AND content LIKE ? ESCAPE '\\' ORDER BY created DESC LIMIT ?",
                                      (self.owner, scope, self.clock(), "%" + escaped + "%", max_records)).fetchall()
        result, used = [], 0
        for row in rows:
            record = self._record(row)
            size = len(json.dumps(asdict(record), ensure_ascii=False).encode("utf-8"))
            if used + size > max_bytes:
                break
            result.append(record)
            used += size
        return result

    def delete(self, memory_id: str, *, scope: str) -> bool:
        self._scope(scope)
        with self._connect() as connection:
            changed = connection.execute("DELETE FROM memory WHERE id=? AND owner=? AND scope=?", (memory_id, self.owner, scope)).rowcount
            if changed:
                self._audit(connection, "deleted", memory_id, scope)
        return bool(changed)

    def purge_expired(self) -> int:
        with self._connect() as connection:
            rows = connection.execute("SELECT id,scope FROM memory WHERE owner=? AND expires<=?", (self.owner, self.clock())).fetchall()
            connection.execute("DELETE FROM memory WHERE owner=? AND expires<=?", (self.owner, self.clock()))
            for row in rows:
                self._audit(connection, "expired", row["id"], row["scope"])
        return len(rows)

    def export_training(self, *, scope: str, consent: bool = False, max_records: int = 1000) -> list[dict[str, Any]]:
        self._scope(scope)
        if not consent:
            raise PermissionError("memory training export requires explicit consent")
        if not 1 <= max_records <= 10000:
            raise ValueError("invalid export record budget")
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM memory WHERE owner=? AND scope=? AND expires>? AND consent=1 AND training_allowed=1 ORDER BY created LIMIT ?",
                                      (self.owner, scope, self.clock(), max_records)).fetchall()
            for row in rows:
                self._audit(connection, "training_export", row["id"], scope)
        return [asdict(self._record(row)) for row in rows]

    def audit_events(self, *, limit: int = 1000) -> list[dict[str, Any]]:
        if not 1 <= limit <= 10000:
            raise ValueError("invalid audit limit")
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM audit WHERE owner=? ORDER BY sequence DESC LIMIT ?", (self.owner, limit)).fetchall()
        return [dict(row) for row in reversed(rows)]
