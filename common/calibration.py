# ★요소4: 캘리브레이션 모듈★ — hand-eye(T_flange_cam, eye-in-hand) / TCP offset /
# NuRec 장면 정합(T_base_nurec) yaml I/O와
# 좌표 변환. 순수 numpy — sim(카메라 배치)과 real(역투영·보정)이 같은 파일·같은
# 함수를 쓴다. 실측 스크립트는 deployment/Calib_*.py (이 모듈은 수학·I/O만).
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

# ---------- quaternion 유틸 (wxyz 고정 — Isaac Lab 관례와 동일) ----------


def quat_to_mat(quat_wxyz: np.ndarray) -> np.ndarray:
    """(4,) wxyz → (3,3) 회전행렬."""
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        raise ValueError("영(0) quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def mat_to_quat(rot: np.ndarray) -> np.ndarray:
    """(3,3) 회전행렬 → (4,) wxyz (w>=0 정규화)."""
    m = np.asarray(rot, dtype=np.float64)
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q if q[0] >= 0 else -q


def pose7_to_mat(pose: np.ndarray) -> np.ndarray:
    """(7,) pos+quat(wxyz) → (4,4)."""
    pose = np.asarray(pose, dtype=np.float64).reshape(7)
    T = np.eye(4)
    T[:3, :3] = quat_to_mat(pose[3:])
    T[:3, 3] = pose[:3]
    return T


def mat_to_pose7(T: np.ndarray) -> np.ndarray:
    """(4,4) → (7,) pos+quat(wxyz)."""
    T = np.asarray(T, dtype=np.float64)
    return np.concatenate([T[:3, 3], mat_to_quat(T[:3, :3])])


# ---------- yaml I/O (configs/calib/*.yaml 계약) ----------


def _load_T(path: Path | str, key: str) -> np.ndarray:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    T = np.asarray(data[key], dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"{path}:{key} 형상 오류 {T.shape} (기대 (4,4))")
    return T


def _save_T(path: Path | str, key: str, T: np.ndarray, meta: dict) -> None:
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"{key} 형상 오류 {T.shape}")
    data = {key: [[float(v) for v in row] for row in T], "meta": meta}
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def load_hand_eye(path: Path | str) -> np.ndarray:
    """configs/calib/hand_eye.yaml → T_flange_cam (4,4).

    ★2026-07-14 eye-in-hand 전환(사용자 확정)★: 카메라가 손목(플랜지) 장착으로
    바뀌어 키가 T_base_cam(고정) → T_flange_cam(플랜지→카메라)이다.
    """
    return _load_T(path, "T_flange_cam")


def save_hand_eye(path: Path | str, T_flange_cam: np.ndarray, meta: dict) -> None:
    _save_T(path, "T_flange_cam", T_flange_cam, meta)


def load_tcp_offset(path: Path | str) -> np.ndarray:
    """configs/calib/tcp_offset.yaml → T_flange_tcp (4,4)."""
    return _load_T(path, "T_flange_tcp")


def save_tcp_offset(path: Path | str, T_flange_tcp: np.ndarray, meta: dict) -> None:
    _save_T(path, "T_flange_tcp", T_flange_tcp, meta)


def load_scene_registration(path: Path | str) -> tuple[np.ndarray, float]:
    """configs/calib/scene_registration.yaml → (T_base_nurec (4,4), scale).

    ★2026-07-17 NuRec 실셀 정합(B안)★: 3DGS 스캔(Workplace.usdz) 볼륨을 로봇
    base frame에 놓는 유일한 변환 소스. T_base_nurec = base←NuRec 원점(similarity의
    강체부), scale = NuRec 단위→meter 균일 스케일(3DGS 스캔은 미터 보장이 없어
    스케일 자유도 필수 — Umeyama 산출). sim 스폰(scene.make_nurec_background_cfg)과
    배치 GUI(place_nurec_gui.py)가 같은 파일을 읽는다(§4-5: 생성물이지만 커밋 의무).
    """
    T = _load_T(path, "T_base_nurec")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    scale = float(data.get("scale", 1.0))
    if scale <= 0.0:
        raise ValueError(f"{path}: scale은 양수여야 함 ({scale})")
    return T, scale


def save_scene_registration(
    path: Path | str, T_base_nurec: np.ndarray, scale: float, meta: dict
) -> None:
    T = np.asarray(T_base_nurec, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T_base_nurec 형상 오류 {T.shape}")
    data = {
        "T_base_nurec": [[float(v) for v in row] for row in T],
        "scale": float(scale),
        "meta": meta,
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def umeyama_similarity(
    src: np.ndarray, dst: np.ndarray, with_scale: bool = True
) -> tuple[np.ndarray, float, float]:
    """대응점 쌍으로 similarity 변환 추정: dst ≈ scale·R·src + t.

    Umeyama(1991) 폐형해 — src/dst (N,3), N≥3(비공선). 반환: (T (4,4 — R,t만),
    scale, rms_residual[m]). NuRec 정합의 수치 초기해(랜드마크: 로봇 베이스·벨트
    모서리 등)와 tests의 왕복 검증에 사용. 순수 numpy(§4-2).
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError(f"대응점 오류: src {src.shape}, dst {dst.shape} (N≥3 필요)")
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    cs, cd = src - mu_s, dst - mu_d
    cov = cd.T @ cs / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    # ★공선(collinear) 퇴화 검출(2026-07-17 테스트 에이전트 지적)★: 점들이 한
    # 직선 위면 그 축 둘레 회전(roll) 1자유도가 미관측 — SVD는 임의 해를 rms≈0으로
    # 반환해 '조용히 틀린' T_base_nurec이 나온다. 2번째 특이값이 1번째 대비
    # 소실이면 거부(랜드마크를 비공선 배치로 다시 찍는 것이 정답).
    if D[0] < 1e-12 or D[1] < 1e-8 * D[0]:
        raise ValueError(
            f"퇴화 배치(공선 의심): 특이값 {D.tolist()} — 회전 1자유도 미관측. "
            f"랜드마크를 한 직선 위가 아닌 3점 이상으로 배치할 것"
        )
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0  # 반사 방지 (Umeyama 부호 보정)
    R = U @ S @ Vt
    var_s = (cs**2).sum() / src.shape[0]
    scale = float((D * np.diag(S)).sum() / var_s) if with_scale else 1.0
    if scale <= 0.0 or var_s < 1e-12:
        raise ValueError(f"퇴화 배치(공선/중복점 의심): scale={scale}, var={var_s}")
    t = mu_d - scale * R @ mu_s
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    residual = dst - (scale * (src @ R.T) + t)
    rms = float(np.sqrt((residual**2).sum(axis=1).mean()))
    return T, scale, rms


# ---------- 좌표 변환 ----------


# ═════════════════════════════════════════════════════════════════════════════
# ★UR 컨트롤러 Base ↔ URDF base_link 정합 (2026-07-28, 감사 C-2)★
#   URDF(assets/ur5e_2fg7/ur5e_tcw1_2fg7.urdf)가 스스로 명시한다:
#     <joint name="base_link-base_fixed_joint" type="fixed">
#       <!-- Note the rotation over Z of pi radians: as base_link is REP-103 aligned
#            ... this is needed to correctly align 'base' with the 'Base' coordinate
#            system of the UR controller. -->
#       <origin rpy="0 0 3.141592653589793" xyz="0 0 0"/>
#   즉 ★UR 컨트롤러가 보고하는 Base 프레임 = base_link를 z축으로 180° 돌린 것★이다.
#   sim은 base_link 기준(scene.py ee_frame source = Robot/base_link)이고 real RTDE
#   getActualTCPPose()는 컨트롤러 Base 기준이므로, 변환 없이 쓰면 ★x·y 부호가 반대★가 되어
#   tcp_pose 관측·target_pos·workspace AABB가 전부 뒤집힌다(정책이 거울상 좌표를 받는다).
#   ⚠️ 이 변환은 대합(involution): 두 번 적용하면 원복된다.
#   ⚠️ ★실기 확인 필수★ — 홈 자세에서 getActualTCPPose()를 실측해 sim FK와 나란히 대조할 것
#      (Phase1-1이 요구하는 바로 그 검증). 대조 결과가 '변환 불필요'로 나오면
#      configs/real.local.yaml robot.tcp_frame_convention을 base_link로 바꾼다.
# ═════════════════════════════════════════════════════════════════════════════
_RZ_PI = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])


def ur_base_to_base_link_pose7(pose7_urbase: np.ndarray) -> np.ndarray:
    """UR 컨트롤러 Base 기준 pose7 → URDF base_link 기준 pose7 (Rz(π) 적용).

    위치는 (x,y,z) → (−x,−y,z), 자세는 R → Rz(π)·R.
    """
    T = pose7_to_mat(np.asarray(pose7_urbase, dtype=np.float64).reshape(7))
    T_out = np.eye(4)
    T_out[:3, :3] = _RZ_PI @ T[:3, :3]
    T_out[:3, 3] = _RZ_PI @ T[:3, 3]
    return mat_to_pose7(T_out)


def ur_base_to_base_link_wrench(wrench6: np.ndarray) -> np.ndarray:
    """UR Base 기준 wrench (6,)=[Fx,Fy,Fz,Tx,Ty,Tz] → base_link 기준.

    force·torque 모두 base frame의 자유벡터라 회전만 적용한다(모멘트 팔 이동 없음 —
    원점이 같은 순수 회전이므로).
    """
    w = np.asarray(wrench6, dtype=np.float64).reshape(6)
    return np.concatenate([_RZ_PI @ w[:3], _RZ_PI @ w[3:]])


def camera_to_base(p_cam: np.ndarray, T_base_cam: np.ndarray) -> np.ndarray:
    """카메라 프레임 점 (3,)|(N,3) → base 프레임 (같은 형상 반환)."""
    p = np.asarray(p_cam, dtype=np.float64)
    single = p.ndim == 1
    p2 = p.reshape(-1, 3)
    out = p2 @ np.asarray(T_base_cam)[:3, :3].T + np.asarray(T_base_cam)[:3, 3]
    return out[0] if single else out


def apply_tcp_offset(flange_pose7: np.ndarray, T_flange_tcp: np.ndarray) -> np.ndarray:
    """flange 포즈 (7,) → TCP 포즈 (7,). real_robot이 RobotState.tcp_pose를 만들 때 사용."""
    return mat_to_pose7(pose7_to_mat(flange_pose7) @ np.asarray(T_flange_tcp))
