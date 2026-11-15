import time
from collections import defaultdict, deque
from threading import Lock


class RateLimiter:
    def __init__(self, limit: int, window_seconds: int = 60, redis_url: str | None = None):
        self.limit = limit
        self.window = window_seconds
        self._hits: defaultdict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()
        self._redis = None
        if redis_url:
            try:
                import redis
                self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
            except Exception:
                self._redis = None

    def allow(self, identity: str) -> tuple[bool, int]:
        now = time.time()
        if self._redis:
            try:
                bucket = int(now // self.window)
                key = f"rag:limit:{identity}:{bucket}"
                count = int(self._redis.incr(key))
                if count == 1:
                    self._redis.expire(key, self.window)
                if count > self.limit:
                    return False, self.window - int(now % self.window)
                return True, 0
            except Exception:
                pass
        with self._lock:
            hits = self._hits[identity]
            while hits and hits[0] <= now - self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return False, max(1, int(self.window - (now - hits[0])))
            hits.append(now)
            return True, 0
