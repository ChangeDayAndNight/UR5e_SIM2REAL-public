"""★2FG7 DO 펄스 드라이버(2026-08-12 채널 확정) 계약 테스트★ — ur_rtde 불필요(순수).

근거 = README에 보존한 실기 계약: 열기 DO6·닫기 DO5 순간 펄스,
펄스 전 양쪽 라인 LOW(동시 assert 방지), DO5·6 외 신호 금지(§07 — 툴 체인저 낙하 위험).
"""
from __future__ import annotations

import yaml
import threading
from pathlib import Path

from real.real_robot import RealRobot, pulse_digital_out

_ROOT = Path(__file__).resolve().parent.parent


def test_pulse_sequence_matches_measured_procedure():
    """시퀀스 = ①상대핀 LOW ②대상핀 LOW ③대상핀 HIGH ④pulse_s 대기 ⑤대상핀 LOW."""
    events: list = []
    pulse_digital_out(
        lambda pin, level: events.append(("set", pin, level)),
        pin_fire=5, pin_other=6, pulse_s=0.3,
        sleep=lambda s: events.append(("sleep", s)),
    )
    assert events == [
        ("set", 6, False),      # 상대(열기) 라인 먼저 LOW — 동시 assert 방지 (report §04)
        ("set", 5, False),
        ("set", 5, True),
        ("sleep", 0.3),
        ("set", 5, False),      # 순간 펄스 — 켜둔 채 두지 않는다
    ]


def test_pulse_touches_only_the_two_gripper_pins():
    """★report §07 안전 계약★ — DO5·DO6 외 핀 접근은 0건이어야 한다(DO0=툴체인저 후보)."""
    touched: set[int] = set()
    pulse_digital_out(lambda pin, level: touched.add(pin), 6, 5, 0.0, sleep=lambda s: None)
    assert touched == {5, 6}


def test_preset_stop_event_never_asserts_high():
    events: list[tuple[int, bool]] = []
    stop = threading.Event()
    stop.set()
    pulse_digital_out(
        lambda pin, level: events.append((pin, level)),
        5,
        6,
        0.3,
        sleep=lambda _: None,
        stop_event=stop,
    )
    assert not any(level for _, level in events)
    assert events == [(6, False), (5, False)]


def test_real_yaml_gripper_channel_is_measured_mapping():
    """real.local.yaml이 실측 매핑(driver=do_pulse, open=DO6, close=DO5)에서 벗어나면 실패.

    배선을 실제로 바꾸는 경우에만 이 테스트와 real.example.yaml, RealRobot 생성자 검사를 ★함께★
    갱신한다 — 한 곳만 바꾸는 드리프트를 차단(§9-5-2).
    """
    cfg = yaml.safe_load((_ROOT / "configs" / "real.example.yaml").read_text(encoding="utf-8"))
    g = cfg["gripper"]
    assert g["driver"] == "do_pulse"
    assert int(g["do_open"]) == 6 and int(g["do_close"]) == 5
    assert 0.1 <= float(g["pulse_s"]) <= 1.5      # report §06 튜닝 범위
    assert float(g["dispatch_ack_timeout_s"]) == 0.040
    # 검증 플래그는 실기 1회 검증·관측 계약 재정의 전까지 false — true로 바뀌려면
    # canonical deployment readiness 게이트를 거쳐야 한다(검증 계약이 이 값으로 기동을 결정).
    assert g["driver_verified"] is False or g["width_feedback_verified"] is not None


def test_background_do_failure_trips_immediately_without_self_join(monkeypatch):
    robot = object.__new__(RealRobot)
    robot._estopped = False
    robot._receive_only = False
    robot._grip_worker_error = None
    robot._grip_stop_event = threading.Event()
    robot._grip_io_lock = threading.Lock()
    robot._grip_threads = set()
    robot._grip_stop_incomplete = False
    robot._grip_do_open = 6
    robot._grip_do_close = 5
    robot._grip_pulse_s = 0.0
    robot._rtde_io = type(
        "FailingIo", (), {"setStandardDigitalOut": lambda *_args: (_ for _ in ()).throw(OSError("DO lost"))}
    )()
    tripped = threading.Event()
    robot.emergency_stop = lambda: (setattr(robot, "_estopped", True), tripped.set())
    robot._start_gripper_dispatch_watchdog()

    robot._set_gripper_do_pulse(closed=True)

    assert tripped.wait(1.0)
    assert isinstance(robot._grip_worker_error, OSError)
    assert robot._estopped is True


def test_arm_command_gate_latches_existing_gripper_worker_failure():
    robot = object.__new__(RealRobot)
    robot._estopped = False
    robot._receive_only = False
    robot._grip_worker_error = OSError("DO lost")
    try:
        robot._assert_alive()
    except RuntimeError as exc:
        assert "gripper worker failed" in str(exc)
    else:
        raise AssertionError("worker failure did not block the next command")
    assert robot._estopped is True


def test_estop_between_pulse_check_and_high_suppresses_physical_high(monkeypatch):
    """A stop latch that wins immediately before HIGH must prevent the edge."""
    robot = object.__new__(RealRobot)
    robot._estopped = False
    robot._receive_only = False
    robot._grip_worker_error = None
    robot._grip_stop_event = threading.Event()
    robot._grip_io_lock = threading.Lock()
    robot._grip_threads = set()
    robot._grip_stop_incomplete = False
    robot._grip_do_open = 6
    robot._grip_do_close = 5
    robot._grip_pulse_s = 0.0
    events: list[tuple[int, bool]] = []
    robot._rtde_io = type(
        "RecordingIo",
        (),
        {
            "setStandardDigitalOut": lambda _self, pin, level: (
                events.append((pin, level)) or True
            )
        },
    )()
    robot.emergency_stop = lambda: (
        setattr(robot, "_estopped", True),
        robot._grip_stop_event.set(),
    )
    robot._start_gripper_dispatch_watchdog()

    def adversarial_pulse(io_set, pin_fire, pin_other, _pulse_s, **_kwargs):
        io_set(pin_other, False)
        io_set(pin_fire, False)
        # Model emergency_stop winning after pulse_digital_out's event check
        # but immediately before the nested SDK HIGH call.
        robot._estopped = True
        robot._grip_stop_event.set()
        io_set(pin_fire, True)

    monkeypatch.setattr("real.real_robot.pulse_digital_out", adversarial_pulse)
    robot._set_gripper_do_pulse(closed=True)
    for worker in tuple(robot._grip_threads):
        worker.join(timeout=1.0)

    assert not any(level for _, level in events)
    assert isinstance(robot._grip_worker_error, RuntimeError)
