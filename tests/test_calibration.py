"""요소4(캘리브레이션 수학·yaml I/O) 단위테스트 — 순수 numpy.

quat(wxyz)↔회전행렬 왕복(<1e-9), w>=0 정규화, pose7↔mat 왕복,
camera_to_base 기지값+배치, apply_tcp_offset 기지값, yaml save/load 왕복,
형상 오류 ValueError, 실제 configs/calib/*.yaml 로드 스모크를 검증한다.

실행: python -m pytest tests/test_calibration.py -q
산출: 없음 (pass/fail)
"""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from common.calibration import (
    apply_tcp_offset,
    camera_to_base,
    load_hand_eye,
    load_scene_registration,
    load_tcp_offset,
    mat_to_pose7,
    mat_to_quat,
    pose7_to_mat,
    quat_to_mat,
    save_hand_eye,
    save_scene_registration,
    save_tcp_offset,
    umeyama_similarity,
)
from common.config_io import configs_dir


def _random_quats(n: int, seed: int = 0) -> list[np.ndarray]:
    """w>=0로 정규화한 랜덤 단위 quat (w≈0 부호 모호 케이스는 제외)."""
    rng = np.random.default_rng(seed)
    quats = []
    while len(quats) < n:
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        if abs(q[0]) < 0.05:  # w≈0은 부호 모호 → 별도 축 테스트에서 다룬다
            continue
        quats.append(q if q[0] >= 0 else -q)
    return quats


RANDOM_QUATS = _random_quats(20)
# mat_to_quat의 4개 분기(trace>0 / m00 / m11 / m22 최대)를 모두 밟는 축 회전
AXIS_QUATS = [
    np.array([1.0, 0.0, 0.0, 0.0]),  # 항등 (trace>0)
    np.array([0.0, 1.0, 0.0, 0.0]),  # 180° x (m00 분기)
    np.array([0.0, 0.0, 1.0, 0.0]),  # 180° y (m11 분기)
    np.array([0.0, 0.0, 0.0, 1.0]),  # 180° z (m22 분기)
]


# ------------------------------------------------------------ quat <-> mat ----


@pytest.mark.parametrize("q", RANDOM_QUATS)
def test_quat_mat_roundtrip_random(q):
    """랜덤 회전 다수: quat → mat → quat 왕복 오차 < 1e-9."""
    q2 = mat_to_quat(quat_to_mat(q))
    assert np.linalg.norm(q2 - q) < 1e-9, f"왕복 오차: {q} → {q2}"


@pytest.mark.parametrize("q", AXIS_QUATS)
def test_quat_mat_roundtrip_axis_branches(q):
    """180° 축 회전(4개 분기 커버): 행렬 수준 왕복 < 1e-9."""
    R = quat_to_mat(q)
    R2 = quat_to_mat(mat_to_quat(R))
    assert np.max(np.abs(R2 - R)) < 1e-9


def test_quat_to_mat_is_rotation():
    """quat_to_mat 결과는 정규 직교 + det=+1 (회전행렬 성질)."""
    for q in RANDOM_QUATS[:5]:
        R = quat_to_mat(q)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)


def test_quat_to_mat_normalizes_input():
    """비정규 quat도 내부 정규화 — 2배 스케일 quat과 같은 행렬."""
    q = RANDOM_QUATS[0]
    np.testing.assert_allclose(quat_to_mat(q), quat_to_mat(2.0 * q), atol=1e-12)


def test_mat_to_quat_w_nonnegative():
    """mat_to_quat은 항상 w>=0 표현을 반환 (이중 표현 정규화 계약)."""
    for q in RANDOM_QUATS:
        q_neg = -q  # 같은 회전의 w<0 표현
        out = mat_to_quat(quat_to_mat(q_neg))
        assert out[0] >= 0.0
        assert np.linalg.norm(out - q) < 1e-9  # w>=0 표현으로 복원


def test_zero_quat_raises():
    """영(0) quaternion → ValueError."""
    with pytest.raises(ValueError):
        quat_to_mat(np.zeros(4))


# ------------------------------------------------------------ pose7 <-> mat ----


@pytest.mark.parametrize("q", RANDOM_QUATS[:8])
def test_pose7_mat_roundtrip(q):
    """pose7 → 4x4 → pose7 왕복 < 1e-9 (랜덤 위치+회전)."""
    rng = np.random.default_rng(int(abs(q[1]) * 1e6))
    pose = np.concatenate([rng.uniform(-1, 1, size=3), q])
    pose2 = mat_to_pose7(pose7_to_mat(pose))
    assert np.linalg.norm(pose2 - pose) < 1e-9


def test_pose7_to_mat_structure():
    """4x4 구조: 상단 3x3=회전, 4열=위치, 하단행=[0,0,0,1]."""
    pose = np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0])
    T = pose7_to_mat(pose)
    np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(T[:3, 3], [0.1, 0.2, 0.3], atol=1e-12)
    np.testing.assert_allclose(T[3], [0, 0, 0, 1], atol=1e-12)


def test_pose7_wrong_shape_raises():
    """pose 길이 != 7 → ValueError (reshape 실패)."""
    with pytest.raises(ValueError):
        pose7_to_mat(np.zeros(6))


# ------------------------------------------------------------ 좌표 변환 ----

# hand_eye.yaml 공칭값과 같은 구조의 기지값 T (테스트는 파일과 독립)
T_KNOWN = np.array(
    [
        [0.0, -1.0, 0.0, 0.45],
        [-1.0, 0.0, 0.0, 0.00],
        [0.0, 0.0, -1.0, 0.80],
        [0.0, 0.0, 0.0, 1.00],
    ]
)


def test_camera_to_base_known_value():
    """기지값: p_cam=[0.1,0.2,0.3] → R@p+t = [0.25,-0.1,0.5]."""
    out = camera_to_base(np.array([0.1, 0.2, 0.3]), T_KNOWN)
    assert out.shape == (3,)
    np.testing.assert_allclose(out, [0.25, -0.1, 0.5], atol=1e-12)


def test_camera_to_base_batch():
    """(N,3) 배치: 같은 형상 반환 + 행별로 단건 결과와 일치."""
    rng = np.random.default_rng(2)
    pts = rng.uniform(-1, 1, size=(5, 3))
    out = camera_to_base(pts, T_KNOWN)
    assert out.shape == (5, 3)
    expected = (T_KNOWN[:3, :3] @ pts.T).T + T_KNOWN[:3, 3]
    np.testing.assert_allclose(out, expected, atol=1e-12)
    for i in range(5):
        np.testing.assert_allclose(camera_to_base(pts[i], T_KNOWN), out[i], atol=1e-12)


def test_apply_tcp_offset_identity_rotation():
    """기지값: 항등 회전 flange + z=0.1493 offset → 위치만 +z, quat 유지."""
    T_off = np.eye(4)
    T_off[2, 3] = 0.1493
    flange = np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0])
    tcp = apply_tcp_offset(flange, T_off)
    np.testing.assert_allclose(tcp[:3], [0.1, 0.2, 0.3 + 0.1493], atol=1e-12)
    np.testing.assert_allclose(tcp[3:], [1.0, 0.0, 0.0, 0.0], atol=1e-12)


def test_apply_tcp_offset_rotated_flange():
    """기지값: x축 +90° 회전 flange — offset은 flange 프레임 기준으로 회전되어
    base에서 -y 방향이 된다 (T_base_tcp = T_base_flange @ T_flange_tcp 계약)."""
    half = np.sqrt(0.5)
    flange = np.array([0.0, 0.0, 0.5, half, half, 0.0, 0.0])  # Rx(90°)
    T_off = np.eye(4)
    T_off[2, 3] = 0.1
    tcp = apply_tcp_offset(flange, T_off)
    np.testing.assert_allclose(tcp[:3], [0.0, -0.1, 0.5], atol=1e-9)
    np.testing.assert_allclose(tcp[3:], flange[3:], atol=1e-9)  # 회전은 그대로


# ---------------------------------------------------------------- yaml I/O ----


def test_hand_eye_save_load_roundtrip(tmp_path):
    """save_hand_eye → load_hand_eye 왕복 (meta 포함 유효 yaml)."""
    path = tmp_path / "hand_eye.yaml"
    save_hand_eye(path, T_KNOWN, meta={"calibrated": False, "method": "test"})
    T2 = load_hand_eye(path)
    assert T2.shape == (4, 4)
    np.testing.assert_allclose(T2, T_KNOWN, atol=1e-15)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["meta"]["method"] == "test"


def test_tcp_offset_save_load_roundtrip(tmp_path):
    """save_tcp_offset → load_tcp_offset 왕복."""
    path = tmp_path / "tcp_offset.yaml"
    T = np.eye(4)
    T[2, 3] = 0.1493
    save_tcp_offset(path, T, meta={"gripper_model": "robotiq_2f85_like"})
    np.testing.assert_allclose(load_tcp_offset(path), T, atol=1e-15)


def test_save_wrong_shape_raises(tmp_path):
    """(4,4)가 아닌 행렬 저장 시도 → ValueError."""
    with pytest.raises(ValueError):
        save_hand_eye(tmp_path / "bad.yaml", np.eye(3), meta={})


def test_load_wrong_shape_raises(tmp_path):
    """yaml 안의 행렬이 (4,4)가 아니면 ValueError."""
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.safe_dump({"T_flange_cam": [[1.0, 0.0], [0.0, 1.0]], "meta": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_hand_eye(path)


# ------------------------------------------------- umeyama_similarity ----
# ★NuRec 실셀 정합(2026-07-17 B안)★ — dst ≈ scale·R·src + t 폐형해 회귀 가드


def _random_similarity(seed: int):
    """랜덤 (R, t, s) 합성 — RANDOM_QUATS 재사용으로 회전 다양성 확보."""
    rng = np.random.default_rng(seed)
    R = quat_to_mat(RANDOM_QUATS[seed % len(RANDOM_QUATS)])
    t = rng.uniform(-2.0, 2.0, size=3)
    s = float(rng.uniform(0.3, 3.0))
    return R, t, s


@pytest.mark.parametrize("seed", range(8))
def test_umeyama_roundtrip_random(seed):
    """랜덤 R·t·s 합성 대응점 → (T, scale) 복원 오차 < 1e-9, rms ≈ 0."""
    rng = np.random.default_rng(100 + seed)
    R, t, s = _random_similarity(seed)
    src = rng.uniform(-1.0, 1.0, size=(10, 3))
    dst = s * (src @ R.T) + t
    T, scale, rms = umeyama_similarity(src, dst)
    assert np.max(np.abs(T[:3, :3] - R)) < 1e-9
    assert np.max(np.abs(T[:3, 3] - t)) < 1e-9
    assert scale == pytest.approx(s, abs=1e-9)
    assert rms < 1e-9
    np.testing.assert_allclose(T[3], [0, 0, 0, 1], atol=1e-15)


def test_umeyama_with_scale_false_rigid_only():
    """with_scale=False: scale은 1.0 고정 — 강체(s=1) 데이터는 정확 복원."""
    rng = np.random.default_rng(3)
    R, t, _ = _random_similarity(2)
    src = rng.uniform(-1.0, 1.0, size=(6, 3))
    dst = src @ R.T + t
    T, scale, rms = umeyama_similarity(src, dst, with_scale=False)
    assert scale == 1.0
    assert np.max(np.abs(T[:3, :3] - R)) < 1e-9
    assert rms < 1e-9


def test_umeyama_planar_points_det_correction_exact():
    """평면(rank-2) 배치 — 반사 보정(S 부호) 경로에서도 정확 복원 + det(R)=+1.

    실셀 랜드마크(벨트 모서리 등)는 거의 평면 — SVD 3번째 특이벡터 부호가
    임의라서 det 보정이 없으면 반사 행렬이 나올 수 있는 케이스다.
    """
    rng = np.random.default_rng(4)
    R, t, s = _random_similarity(5)
    src = np.c_[rng.uniform(-1.0, 1.0, size=(8, 2)), np.zeros(8)]  # z=0 평면
    dst = s * (src @ R.T) + t
    T, scale, rms = umeyama_similarity(src, dst)
    assert np.linalg.det(T[:3, :3]) == pytest.approx(1.0, abs=1e-9)
    assert np.max(np.abs(T[:3, :3] - R)) < 1e-9
    assert scale == pytest.approx(s, abs=1e-9)
    assert rms < 1e-9


def test_umeyama_mirrored_data_returns_proper_rotation():
    """거울상(dst = src의 z 반전) 입력 — 반환 R은 반드시 정회전(det=+1).

    반사가 최적이어도 Umeyama S 보정이 '최선의 정회전'을 반환해야 한다
    (캘리브 산출이 반사 좌표계가 되는 사고 방지). rms>0(정확 정합 불가)이 정상.
    """
    rng = np.random.default_rng(6)
    src = rng.uniform(-1.0, 1.0, size=(8, 3))
    dst = src.copy()
    dst[:, 2] *= -1.0
    T, scale, rms = umeyama_similarity(src, dst)
    assert np.linalg.det(T[:3, :3]) == pytest.approx(1.0, abs=1e-9)
    assert scale > 0.0
    assert rms > 0.01  # 거울상은 회전만으로 정합 불가 — 잔차가 남아야 정직하다


def test_umeyama_too_few_or_mismatched_points_raises():
    """N<3 또는 src/dst 개수 불일치 → ValueError."""
    pts2 = np.zeros((2, 3))
    with pytest.raises(ValueError):
        umeyama_similarity(pts2, pts2)
    with pytest.raises(ValueError):
        umeyama_similarity(np.zeros((4, 3)), np.zeros((5, 3)))


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_umeyama_degenerate_duplicate_points_raises():
    """전 점 동일(분산 0) 퇴화 배치 → ValueError (var_s 가드).

    frozen 코드가 가드 raise 직전 0/0 나눗셈(RuntimeWarning)을 지나므로 경고만 무시.
    """
    same = np.tile([0.1, 0.2, 0.3], (4, 1))
    with pytest.raises(ValueError):
        umeyama_similarity(same, same)


def test_umeyama_collinear_points_should_raise():
    """공선 3점: 축 회전이 미관측(해 비유일) → ValueError.

    이력(2026-07-17): 최초 구현은 공선을 통과시켜 임의 roll 해를 rms≈0으로
    반환했다(테스트 에이전트 xfail 기록) → calibration.py에 2번째 특이값 소실
    게이트(D[1] < 1e-8·D[0])를 추가해 해소. 실측 랜드마크는 비공선 배치 필수.
    """
    src = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    R, t, s = _random_similarity(1)
    dst = s * (src @ R.T) + t
    with pytest.raises(ValueError):
        umeyama_similarity(src, dst)


# ------------------------------------------------- scene_registration I/O ----


def test_scene_registration_save_load_roundtrip(tmp_path):
    """save → load 왕복: T_base_nurec + scale + meta 보존 (NuRec 정합 계약)."""
    path = tmp_path / "scene_registration.yaml"
    save_scene_registration(
        path, T_KNOWN, scale=0.037,
        meta={"calibrated": True, "method": "umeyama_landmarks", "residual_m": 0.004},
    )
    T2, scale2 = load_scene_registration(path)
    np.testing.assert_allclose(T2, T_KNOWN, atol=1e-15)
    assert scale2 == pytest.approx(0.037, abs=1e-15)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["meta"]["method"] == "umeyama_landmarks"


def test_scene_registration_missing_scale_defaults_to_one(tmp_path):
    """yaml에 scale 키가 없으면 1.0 (하위 호환 기본값)."""
    path = tmp_path / "scene_registration.yaml"
    path.write_text(
        yaml.safe_dump({"T_base_nurec": [[float(v) for v in row] for row in np.eye(4)]}),
        encoding="utf-8",
    )
    _, scale = load_scene_registration(path)
    assert scale == 1.0


@pytest.mark.parametrize("bad_scale", [0.0, -1.0])
def test_scene_registration_nonpositive_scale_raises(tmp_path, bad_scale):
    """scale <= 0 → ValueError (미터 변환 스케일은 양수 계약)."""
    path = tmp_path / "scene_registration.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "T_base_nurec": [[float(v) for v in row] for row in np.eye(4)],
                "scale": bad_scale,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_scene_registration(path)


def test_scene_registration_save_wrong_shape_raises(tmp_path):
    """(4,4)가 아닌 T 저장 시도 → ValueError."""
    with pytest.raises(ValueError):
        save_scene_registration(tmp_path / "bad.yaml", np.eye(3), scale=1.0, meta={})


def test_real_scene_registration_yaml_smoke():
    """실제 configs/calib/scene_registration.yaml 로드 스모크 — 구조 계약 검증."""
    T, scale = load_scene_registration(configs_dir() / "calib" / "scene_registration.yaml")
    assert T.shape == (4, 4)
    assert np.all(np.isfinite(T))
    np.testing.assert_allclose(T[3], [0, 0, 0, 1], atol=1e-12)
    R = T[:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert scale > 0.0


def test_real_calib_configs_load_smoke():
    """실제 configs/calib/*.yaml 로드 스모크: (4,4), 유한값, 하단행 [0,0,0,1],
    회전부 직교성 — 공칭값이더라도 구조 계약은 지켜야 한다."""
    calib = configs_dir() / "calib"
    for loader, fname in ((load_hand_eye, "hand_eye.yaml"), (load_tcp_offset, "tcp_offset.yaml")):
        T = loader(calib / fname)
        assert T.shape == (4, 4)
        assert np.all(np.isfinite(T))
        np.testing.assert_allclose(T[3], [0, 0, 0, 1], atol=1e-12)
        R = T[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
