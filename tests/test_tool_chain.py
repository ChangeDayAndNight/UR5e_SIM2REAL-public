"""툴 체인 계약 — 실기 5단 체인과 sim URDF/tcp_offset의 불일치를 드러낸다 (순수, sim 불필요).

배경(2026-07-30 사용자 제공): 실기 툴 체인이
      UR5e flange → tool flange → magbot TCW1 → ★OnRobot QC-RS v3★ → OnRobot 2FG7
      5단임이 사진(Real_Environment/KakaoTalk_20260730_*)으로 확인됐다.
      그런데 sim URDF는 ★3단(tool0 → TCW1 → 2FG7)★이라 QC-RS v3 한 단이 누락돼
      tcp_offset이 그만큼 짧다 — 실기에서 ★그 높이만큼 낮게 집으려 한다★는 뜻이고
      파지 실패나 벨트 충돌로 이어진다.

이 테스트는 ★불일치를 숨기지 않는다★:
  ①tool_chain.yaml의 미측정 단이 남아 있으면 xfail로 표시(수치를 채우면 자동 통과)
  ②측정이 끝난 단들의 합은 URDF joint origin과 정확히 일치해야 한다
  ③QC 해소 후에는 체인 합계 == tcp_offset.yaml z 여야 한다
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from common.calibration import load_tcp_offset
from common.config_io import configs_dir, load_yaml

_ROOT = Path(__file__).resolve().parent.parent
_URDF = _ROOT / "assets" / "ur5e_2fg7" / "ur5e_tcw1_2fg7.urdf"
_CHAIN = configs_dir() / "calib" / "tool_chain.yaml"


@pytest.fixture(scope="module")
def chain() -> dict:
    if not _CHAIN.is_file():
        pytest.skip(f"tool_chain.yaml 없음: {_CHAIN}")
    return load_yaml(_CHAIN)


def test_chain_lists_all_five_real_elements(chain: dict) -> None:
    """실기 5단이 전부 기록돼 있어야 한다 — 빠진 단이 곧 sim2real 오차다."""
    names = [str(e.get("name", "")) for e in chain.get("elements", [])]
    for must in ("tool_flange_plate", "magbot_TCW1", "onrobot_QC_RS_v3", "onrobot_2FG7"):
        assert must in names, f"체인에 {must}가 없다 — 실기 5단 중 한 단이 누락됐다"


def test_measured_elements_match_urdf(chain: dict) -> None:
    """★측정 완료된 단은 URDF와 정확히 일치해야 한다★ (문서-자산 드리프트 방지)."""
    if not _URDF.is_file():
        pytest.skip("URDF 없음")
    root = ET.parse(_URDF).getroot()

    def joint_z(parent: str, child: str) -> float | None:
        for j in root.iter("joint"):
            if j.find("parent").get("link") == parent and j.find("child").get("link") == child:
                return float(j.find("origin").get("xyz").split()[2])
        return None

    by = {str(e.get("name")): e for e in chain.get("elements", [])}
    # yaml 각 단 ↔ URDF joint origin z 대응 (5단 체인, 2026-07-30)
    pairs = {
        "tool_flange_plate": ("tool_flange_plate_link", "tcw1_link"),
        "magbot_TCW1":       ("tcw1_link", "qc_rs_v3_link"),
        "onrobot_QC_RS_v3":  ("qc_rs_v3_link", "onrobot_base_link"),
    }
    for name, (pa, ch) in pairs.items():
        e = by.get(name)
        if not e or not e.get("measured"):
            continue
        urdf_z = joint_z(pa, ch)
        assert urdf_z is not None, f"URDF에서 {pa}→{ch}를 찾지 못함 — 체인이 끊겼다"
        assert abs(urdf_z - float(e["height_m"])) < 1e-6, (
            f"{name} 높이 불일치 — yaml {e['height_m']} vs URDF {urdf_z}"
        )


def test_2fg7_grip_height_matches_urdf(chain: dict) -> None:
    """2FG7 base→pad중점 거리가 URDF(joint origin + collision origin)와 일치해야 한다."""
    if not _URDF.is_file():
        pytest.skip("URDF 없음")
    root = ET.parse(_URDF).getroot()
    jz = cz = None
    for j in root.iter("joint"):
        if j.get("name") == "left_finger_joint":
            jz = float(j.find("origin").get("xyz").split()[2])
    for l in root.iter("link"):
        if l.get("name") == "left_finger_link":
            cz = float(l.find("collision/origin").get("xyz").split()[2])
    assert jz is not None and cz is not None
    by = {str(e.get("name")): e for e in chain.get("elements", [])}
    yaml_h = float(by["onrobot_2FG7"]["height_to_grip_m"])
    assert abs((jz + cz) - yaml_h) < 1e-5, (
        f"2FG7 파지중심 높이 불일치 — yaml {yaml_h} vs URDF {jz + cz:.5f}"
    )


def test_chain_sum_matches_tcp_offset(chain: dict) -> None:
    """★체인 합계 == tcp_offset.yaml z★.

    현재는 QC-RS v3 미측정으로 ★의도적으로 실패(xfail)★한다 — 이 불일치가 곧
    해소해야 할 실기 오차다. QC 높이를 실측해 tool_chain.yaml에 넣고 URDF·tcp_offset을
    갱신하면 자동으로 통과한다.
    """
    if chain.get("verification", {}).get("qc_missing", False):
        pytest.xfail("QC-RS v3 미측정 — 실측 후 tool_chain.yaml qc_missing: false 로 변경")
    total = 0.0
    for e in chain.get("elements", []):
        h = e.get("height_to_grip_m")
        if h is None:
            h = e.get("height_m")
        if h is None:
            pytest.fail(f"{e.get('name')} 높이가 여전히 미측정이다")
        total += float(h)
    tcp_z = float(load_tcp_offset(configs_dir() / "calib" / "tcp_offset.yaml")[2, 3])
    assert abs(total - tcp_z) < 1e-4, (
        f"체인 합계 {total*1000:.1f}mm != tcp_offset {tcp_z*1000:.1f}mm "
        f"(차 {(total-tcp_z)*1000:+.1f}mm) — URDF/tcp_offset 갱신 필요"
    )


def test_photos_referenced_exist(chain: dict) -> None:
    """규격 근거 사진이 로컬에 있으면 검증한다.

    ⚠️Real_Environment/는 대용량 사진이라 .gitignore 대상이다 — 클론 직후엔 없다.
      그래서 폴더가 없으면 skip한다(테스트가 신규 클론에서 깨지지 않게).
      단 폴더가 있는데 2026-07-30 사진이 없으면 ★규격 근거 소실★로 실패시킨다.
    """
    folder = _ROOT / "Real_Environment"
    if not folder.is_dir():
        pytest.skip("Real_Environment/ 없음(.gitignore 대상) — 사진 근거는 로컬 보관")
    photos = sorted(folder.glob("KakaoTalk_20260730_*.jpg"))
    assert photos, (
        "Real_Environment/에 2026-07-30 툴 체인 사진이 없다 — tool_chain.yaml의 "
        "QC-RS v3 수치 근거가 소실됐다(사진을 복원하거나 실측값으로 교체할 것)"
    )
