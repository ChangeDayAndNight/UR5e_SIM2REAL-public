# sim — Phase3 리허설용 Isaac Lab 어댑터 패키지 (isaac51 전용).
#
# ⚠️★불변 규칙: 이 패키지의 모듈은 isaaclab을 import하므로 반드시
#   AppLauncher(SimulationApp) 기동 "이후"에만 import할 것.★
#   (스크립트 최상단에서 import하면 Kit 미기동 상태에서 실패한다.)
#
# 구성 (환경/태스크 정의는 확장 패키지 ur5e_binpick.tasks.binpick 으로 이관됨):
#   sim_robot.py  : SimRobot(RobotInterface)  — Phase3 SimDeploy_Rehearsal 어댑터
#   sim_camera.py : SimCamera(CameraInterface) — Phase3 SimDeploy_Rehearsal 어댑터
#
# real/ 의 real_robot/real_camera와 대칭 — 같은 common.rollout.run_rollout 루프를
# sim 백엔드로 돌려 배포 전 리허설한다. 이 파일 자체는 아무것도 import하지 않는다.
