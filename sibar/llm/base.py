"""Shared pieces of the LLM backends: a disk cache and JSON extraction."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import sqlite3
import threading
import time
from typing import Optional

log = logging.getLogger(__name__)


class DiskCache:
    """Persistent prompt -> completion cache, so replays and reruns cost nothing extra."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self._pid = None
        self._lock = threading.Lock()

    @property
    def _db(self) -> sqlite3.Connection:
        if self._pid != os.getpid():           # reconnect in forked worker processes
            self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=120)
            self._conn.execute("PRAGMA busy_timeout=120000")
            self._locked(lambda: self._conn.execute("PRAGMA journal_mode=WAL"))   # many concurrent writers
            self._locked(lambda: self._conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)"))
            self._pid = os.getpid()
        return self._conn

    @staticmethod
    def key(*parts: str) -> str:
        return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()

    @staticmethod
    def _locked(fn, attempts: int = 20):
        """Retry an operation that hits 'database is locked' under heavy write contention."""
        for i in range(attempts):
            try:
                return fn()
            except sqlite3.OperationalError as e:
                if "locked" not in str(e) or i == attempts - 1:
                    raise
                time.sleep(0.2 * (i + 1) * (1 + random.random()))

    def get(self, k: str) -> Optional[str]:
        with self._lock:
            row = self._locked(lambda: self._db.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone())
        return row[0] if row else None

    def put(self, k: str, v: str) -> None:
        def write():
            self._db.execute("INSERT OR REPLACE INTO kv VALUES (?, ?)", (k, v))
            self._db.commit()
        with self._lock:
            try:
                self._locked(write)
            except sqlite3.OperationalError as e:   # never abort a run over caching
                log.warning("cache write skipped: %s", e)


def extract_json(text: str):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    for opener, closer in (("{", "}"), ("[", "]")):
        s, e = text.find(opener), text.rfind(closer)
        if s != -1 and e > s:
            try:
                return json.loads(text[s:e + 1])
            except json.JSONDecodeError:
                continue
    return None


class ChatBackend:
    """complete(system, user) -> text, cached on disk."""

    name = "base"

    def __init__(self, model: str, cache: DiskCache):
        self.model = model
        self.cache = cache

    def _complete(self, system: str, user: str) -> str:
        raise NotImplementedError

    max_attempts = 6

    def complete(self, system: str, user: str) -> str:
        k = DiskCache.key(self.name, self.model, system, user)
        hit = self.cache.get(k)
        if hit is not None:
            return hit
        for attempt in range(self.max_attempts):
            try:
                out = self._complete(system, user)
            except Exception as e:  # malformed gateway responses, timeouts, rate limits
                wait = min(60.0, 2.0 ** attempt) * (1 + random.random())
                log.warning("%s call failed (%s: %s); retry %d in %.0fs", self.name, type(e).__name__,
                            str(e)[:120], attempt + 1, wait)
                time.sleep(wait)
                continue
            self.cache.put(k, out)
            return out
        log.error("%s call failed %d times; answering empty (not cached)", self.name, self.max_attempts)
        return ""
