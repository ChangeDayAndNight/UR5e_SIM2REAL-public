"""ur5e_binpick — UR5e depth-vision bin-picking Isaac Lab 확장 패키지.

`import ur5e_binpick`만으로 태스크가 gym에 등록되도록 tasks 하위를 끌어온다
(scripts/rsl_rl/train.py는 `import ur5e_binpick.tasks`로 직접 등록을 트리거).
"""

from . import tasks  # noqa: F401  (gym.register 트리거)
