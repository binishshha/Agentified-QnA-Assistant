import hashlib
import json
import time
from typing import Any


class ResponseCache:
    """Redis cache with an in-process fallback for local development."""
    def __init__(self, redis_url: str | None, ttl: int, max_size: int = 256):
        self.ttl = ttl
        self.max_size = max_size
        self._memory: dict[str, tuple[float, dict[str, Any]]] = {}
        self._redis = None
        if redis_url:
            try:
                import redis
                self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
            except Exception:
                self._redis = None

    @staticmethod
    def key(question: str, top_k: int, version: str = "1") -> str:
        raw = f"{version}:{top_k}:{' '.join(question.lower().split())}"
        return "rag:answer:" + hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        try:
            raw = self._redis.get(key) if self._redis else None
            if raw:
                return json.loads(raw)
        except Exception:
            pass
        item = self._memory.get(key)
        if item and item[0] > time.time():
            return item[1]
        self._memory.pop(key, None)
        return None

    def set(self, key: str, value: dict[str, Any]) -> None:
        try:
            if self._redis:
                self._redis.setex(key, self.ttl, json.dumps(value))
                return
        except Exception:
            pass
        self._memory[key] = (time.time() + self.ttl, value)
        while len(self._memory) > self.max_size:
            self._memory.pop(next(iter(self._memory)))
