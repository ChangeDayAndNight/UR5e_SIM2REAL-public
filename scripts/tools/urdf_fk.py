"""URDF 순기구학(FK) — 배치 GUI 고스트의 관절 포즈 계산용 (pure: stdlib+numpy, Isaac 불필요).

배경: URDF→USD 변환본은 링크가 루트 아래 '평면(sibling)'으로 놓이고 관절은 PhysX 전용
프림이라, 물리 시뮬레이션 없이는 기즈모로 관절을 움직일 수 없다(체인이 안 따라옴).
→ 배치 GUI(place_workspace_gui.py)는 이 모듈의 FK로 각 링크의 루트기준 pose를 계산해
   링크 Xform을 직접 다시 쓴다(슬라이더 → FK → 링크 재저작).

⚠️ 시각화 전용 — 학습/실기 kinematics의 소스는 Isaac(PhysX)이다.
검증: tests/test_urdf_fk.py — 홈 포즈 FK가 실측 파지점(_PREGRASP_GRIP)과 일치해야 한다.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# UR5e 팔 관절(슬라이더 노출 순서). 그리퍼는 left_finger_joint 하나로 조작(right=mimic).
ARM_JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


@dataclass(frozen=True)
class Joint:
    """URDF joint 1개 — origin은 4x4 동차변환(부모링크frame→joint frame)."""

    name: str
    jtype: str  # revolute / prismatic / fixed
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float
    mimic_joint: str | None = None
    mimic_mult: float = 1.0
    mimic_off: float = 0.0


def _rpy_mat(r: float, p: float, y: float) -> np.ndarray:
    """URDF rpy(고정축 XYZ) → 회전행렬 R = Rz(y)·Ry(p)·Rx(r)."""
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _axis_rot(axis: np.ndarray, q: float) -> np.ndarray:
    """축-각(Rodrigues) 회전행렬."""
    a = axis / (np.linalg.norm(axis) or 1.0)
    c, s = math.cos(q), math.sin(q)
    x, y, z = a
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) * c + s * k + (1 - c) * np.outer(a, a)


def load_joints(urdf_path: str | Path) -> list[Joint]:
    """URDF의 모든 joint 파싱(고정 관절 포함 — FK 체인에 필요)."""
    root = ET.parse(str(urdf_path)).getroot()
    joints: list[Joint] = []
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
        rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
        origin = np.eye(4)
        origin[:3, :3] = _rpy_mat(*rpy)
        origin[:3, 3] = xyz
        a = j.find("axis")
        axis = np.array([float(v) for v in (a.get("xyz", "1 0 0") if a is not None else "1 0 0").split()])
        lim = j.find("limit")
        lower = float(lim.get("lower", "0")) if lim is not None else 0.0
        upper = float(lim.get("upper", "0")) if lim is not None else 0.0
        m = j.find("mimic")
        joints.append(
            Joint(
                name=j.get("name"),
                jtype=j.get("type"),
                parent=j.find("parent").get("link"),
                child=j.find("child").get("link"),
                origin=origin,
                axis=axis,
                lower=lower,
                upper=upper,
                mimic_joint=(m.get("joint") if m is not None else None),
                mimic_mult=float(m.get("multiplier", "1.0")) if m is not None else 1.0,
                mimic_off=float(m.get("offset", "0.0")) if m is not None else 0.0,
            )
        )
    return joints


def root_link(joints: list[Joint]) -> str:
    """부모로만 등장하는 링크 = 루트(base_link)."""
    parents = {j.parent for j in joints}
    children = {j.child for j in joints}
    roots = sorted(parents - children)
    if len(roots) != 1:
        raise ValueError(f"루트 링크가 유일하지 않음: {roots}")
    return roots[0]


def fk(joints: list[Joint], q: dict[str, float]) -> dict[str, np.ndarray]:
    """관절값 q[rad|m] → 링크별 4x4 pose(루트링크 frame). 미지정 관절 = 0, mimic 자동 해석."""

    def q_of(j: Joint) -> float:
        if j.mimic_joint is not None:
            return q.get(j.mimic_joint, 0.0) * j.mimic_mult + j.mimic_off
        return q.get(j.name, 0.0)

    base = root_link(joints)
    poses: dict[str, np.ndarray] = {base: np.eye(4)}
    pending = list(joints)
    while pending:
        progressed = False
        rest: list[Joint] = []
        for j in pending:
            if j.parent not in poses:
                rest.append(j)
                continue
            motion = np.eye(4)
            if j.jtype in ("revolute", "continuous"):
                motion[:3, :3] = _axis_rot(j.axis, q_of(j))
            elif j.jtype == "prismatic":
                motion[:3, 3] = j.axis / (np.linalg.norm(j.axis) or 1.0) * q_of(j)
            poses[j.child] = poses[j.parent] @ j.origin @ motion
            progressed = True
        if not progressed:
            raise ValueError(f"FK 체인 미해결(고아 관절): {[j.name for j in rest]}")
        pending = rest
    return poses


def mat_to_quat_wxyz(m: np.ndarray) -> tuple[float, float, float, float]:
    """3x3 회전행렬 → 쿼터니언(wxyz) — USD orient 저작용(수치 안정 분기)."""
    t = float(np.trace(m))
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        return (0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    i = int(np.argmax(np.diag(m)))
    if i == 0:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        return ((m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)
    if i == 1:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        return ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s)
    s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
    return ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s)
