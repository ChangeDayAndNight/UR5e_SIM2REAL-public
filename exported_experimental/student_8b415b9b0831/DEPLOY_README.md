> ★이 번들에 대한 정정(2026-09-25)★ 아래 본문은 실험 레인 공통 템플릿이라 "채택 게이트 미통과"로 적혀 있지만,
> 이 student는 **Phase2 봉인 평가를 통과했습니다**: placed 94.53% (seed 42~45 × 512 = 2048 에피소드,
> Wilson 95% 하한 93.46%, 최저 시드 480/512). 게이트는 시드당 384/512·Wilson 0.75였습니다.
> `adopted: false`는 정식 릴리스 레인(패리티·지연 벤치마크·커밋 동결) 미승인을 뜻합니다. 폴더 최상위
> `README_DEPLOY.md`를 먼저 읽으십시오.

# 실험용 정책 배포 폴더 — ★릴리스 아님★

이 폴더는 **배포 승인을 받지 않은** 실험 산출물입니다. 랩 환경에서 사람이 e-stop 앞에
있는 상태에서만 사용하십시오.

- checkpoint: `logs/rsl_rl/ur5e_binpick_vision/2026-09-22_20-27-07-272017/model_24000.pt` (iter 24000)
- sha256: `8b415b9b08315407…`
- 관측/행동: 50206 / 7
- 학습 커밋 `5368499ae3ee` / export 커밋 `5368499ae3ee`
- **채택 게이트 미통과** — 성공률은 낮습니다. 목표는 "학습 정책이 실기를 구동한다"입니다.
- `configs/calib/hand_eye.yaml` 의 `calibrated`: **false**

## 배포 PC에서 할 일

1. 이 폴더를 배포 PC로 복사합니다.
2. **캘리브레이션 값을 반영**합니다 (`configs/calib/tcp_offset.yaml`, `hand_eye.yaml`의
   `meta.calibrated: true`). 이것이 없으면 `policy` 범위 모션은 코드가 거부합니다.
3. **모션 인가 파일**을 만듭니다 — `configs/motion_authorization.yaml`
   (`configs/motion_authorization.example.yaml` 복사 후 채우기).
   - `operator`: 입회자 이름
   - `expires_at`: acknowledged_at + 72시간 이내
   - `scopes`: 필요한 최소 범위만 (`policy_dryrun` → 확인 후 `policy`)
4. 단계적으로 올립니다:
   - ① 수신 전용 연결 확인
   - ② `policy_dryrun` — 정책이 추론하되 명령은 보내지 않음
   - ③ `policy` + 낮은 speed_scale — 실제 모션

## 안전

`configs/safety.yaml` 의 workspace AABB · TCP z 바닥 · 관절속도 상한이 그대로 적용됩니다.
이탈 시 즉시 trip합니다. **작업 내내 e-stop 도달 거리 안에 사람이 있어야 합니다.**
