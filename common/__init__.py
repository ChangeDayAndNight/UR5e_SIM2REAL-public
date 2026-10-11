# common — sim/real 공용 순수 패키지.
# ★불변 규칙: 이 패키지 어디서도 isaaclab / isaacsim / ur_rtde / pyrealsense2를
#   import하지 않는다 (numpy / torch / yaml / stdlib만).★
# 그래야 isaac51(학습 PC)과 env_ur_real(배포 PC) 양쪽에서 동일하게 import되고,
# 전체가 sim 없이 pytest 가능하다.
