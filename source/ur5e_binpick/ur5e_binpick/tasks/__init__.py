"""ur5e_binpick.tasks — 태스크 구현 패키지 (import 시 gym 등록).

공식 템플릿(IsaacLab/tools/template `__init__tasks`)은 `import_packages`로 하위
태스크를 자동 수집하지만, 이 프로젝트는 단일 태스크 패키지(binpick)라 명시적으로
import한다(결정론적·isaaclab_tasks.utils 비의존). binpick/__init__.py가 gym.register 수행.
"""

from . import binpick  # noqa: F401  (gym.register 트리거)
