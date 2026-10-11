"""pytest 공용 설정 — 프로젝트 루트를 sys.path에 등록.

목적: 어느 cwd에서 pytest를 실행해도 "from common.xxx import ..."가 동작하게 한다.
실행: (자동) pytest가 tests/ 수집 시 로드
산출: 없음 (sys.path 부작용만)
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent  # = UR5e_Sim2Real/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
