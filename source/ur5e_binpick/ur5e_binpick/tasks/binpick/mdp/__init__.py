"""binpick mdp 패키지 — scene / observations / events(DR) / ur5e_assets.

⚠️ 이 하위 모듈들은 isaaclab을 import한다 → AppLauncher 이후(= Hydra가 env cfg를
지연 로드할 때)에만 import된다. binpick/__init__.py(gym.register)는 이 패키지를
직접 import하지 않는다(문자열 entry_point로만 참조).
"""
