# real — 실기(배포 PC) 전용 어댑터 패키지. ★env_ur_real 전용, Isaac 불필요★
#
# ★불변 규칙: 이 패키지 어디서도 isaaclab / isaacsim을 import하지 않는다 —
#   Isaac이 설치되지 않은 env_ur_real(별도 배포 PC)에서 그대로 import되어야 한다.★
#
# 하드웨어 SDK(ur_rtde / pyrealsense2)는 모듈 최상위가 아니라 "연결 시점"에
# 지연 import한다 — 덕분에 SDK가 없는 학습 PC(isaac51)에서도 모듈 import와
# 순수 수학(rotvec_to_quat 등)의 pytest가 가능하다.
#
# 배포 번들은 canonical `deployment/build_bundle.py`가 common/real/Phase4/deployment,
# 고정 config·자산·Student release·hashed wheelhouse만 선별해 생성한다(README 배포 계약).
