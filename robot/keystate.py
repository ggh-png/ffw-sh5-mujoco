"""GLFW key-repeat 기반 key-down 상태 추적.

MuJoCo viewer의 key_callback은 Callable[[int], None] — PRESS+REPEAT만 수신.
첫 PRESS 후 REPEAT 시작 전 ~0.5 s gap을 _INIT_TIMEOUT으로 커버한다.
REPEAT 단계(≥0.5 s)에서는 이벤트 간격 ~33 ms를 _REPEAT_TIMEOUT으로 커버.
"""
import time

_INIT_TIMEOUT   = 0.58   # PRESS → 첫 REPEAT 사이 gap 허용 (0.5 s + margin)
_REPEAT_TIMEOUT = 0.12   # REPEAT 간격 허용 (~33 ms × 3.6)


class KeyState:
    def __init__(self):
        self._first: dict[int, float] = {}
        self._last:  dict[int, float] = {}

    def on_key(self, key: int):
        t = time.perf_counter()
        # 이전 이벤트가 없거나 오래됐으면 새 세션으로 취급
        if key not in self._last or (t - self._last[key]) > _INIT_TIMEOUT:
            self._first[key] = t
        self._last[key] = t

    def is_down(self, key: int) -> bool:
        t = time.perf_counter()
        if key not in self._last:
            return False
        age_since_first = t - self._first.get(key, t)
        age_since_last  = t - self._last[key]
        if age_since_first < _INIT_TIMEOUT:
            return age_since_last < _INIT_TIMEOUT
        return age_since_last < _REPEAT_TIMEOUT
