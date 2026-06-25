"""GLFW key-repeat 기반 key-down 상태 추적.

key_callback은 PRESS + REPEAT만 수신 (RELEASE 없음).
- PRESS 직후 ~ 첫 REPEAT 사이 갭(OS 기본 ~0.5s)을 _INIT_HOLD로 커버
- REPEAT 수신 후에는 _REPEAT_HOLD(250ms)로 커버 — OS repeat 주기 ~33ms의 7.5배 버퍼
"""
import time

_INIT_HOLD   = 0.60   # 첫 PRESS 이후 이 시간(s) 동안 down 유지
_REPEAT_HOLD = 0.25   # REPEAT 이벤트 이후 이 시간(s) 동안 down 유지


class KeyState:
    def __init__(self):
        self._first: dict[int, float] = {}
        self._last:  dict[int, float] = {}

    def on_key(self, key: int):
        t = time.perf_counter()
        if key not in self._last or (t - self._last[key]) > _INIT_HOLD:
            self._first[key] = t
        self._last[key] = t

    def is_down(self, key: int) -> bool:
        if key not in self._last:
            return False
        t = time.perf_counter()
        age_first = t - self._first.get(key, t)
        age_last  = t - self._last[key]
        # 초기 PRESS 구간: INIT_HOLD 안이면 다운
        if age_first < _INIT_HOLD:
            return age_last < _INIT_HOLD
        # REPEAT 구간: 마지막 이벤트 이후 REPEAT_HOLD 안이면 다운
        return age_last < _REPEAT_HOLD
