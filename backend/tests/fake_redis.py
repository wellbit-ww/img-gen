from __future__ import annotations


class FakeRedis:
    """Minimal Redis stand-in for worker batch/cancel tests."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def set(self, name: str, value: object, ex: int | None = None, nx: bool = False) -> bool:
        if nx and name in self.store:
            return False
        self.store[str(name)] = str(value)
        return True

    def get(self, name: str) -> str | None:
        return self.store.get(str(name))

    def incr(self, name: str) -> int:
        key = str(name)
        current = int(self.store.get(key, "0")) + 1
        self.store[key] = str(current)
        return current

    def delete(self, name: str) -> None:
        self.store.pop(str(name), None)

    def setex(self, name: str, _ttl: int, value: object) -> None:
        self.store[str(name)] = str(value)

    def exists(self, name: str) -> int:
        return 1 if str(name) in self.store else 0
