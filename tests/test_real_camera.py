from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from real.real_camera import RealCamera


def _camera_cfg() -> dict:
    return {
        "serial": "example_camera_0001",
        "width": 848,
        "height": 480,
        "fps": 30,
        "frame_timeout_ms": 100,
        "enable_filters": False,
        "global_time_enabled": True,
        "frames_queue_size": 1,
    }


def test_started_pipeline_is_stopped_when_post_start_initialization_fails(monkeypatch):
    stopped: list[bool] = []

    class Device:
        def first_depth_sensor(self):
            raise OSError("depth sensor lookup failed")

    class Profile:
        def get_device(self):
            return Device()

    class Pipeline:
        def start(self, _config):
            return Profile()

        def stop(self):
            stopped.append(True)

    class Config:
        def enable_device(self, _serial):
            pass

        def enable_stream(self, *_args):
            pass

    fake_rs = SimpleNamespace(
        pipeline=Pipeline,
        config=Config,
        stream=SimpleNamespace(depth=object(), color=object()),
        format=SimpleNamespace(z16=object(), rgb8=object()),
    )
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake_rs)
    camera = RealCamera(_camera_cfg())

    with pytest.raises(OSError, match="depth sensor lookup failed"):
        camera.start()

    assert stopped == [True]
    assert camera._pipeline is None
    assert camera._profile is None


def test_frame_preserves_sdk_timestamp_domain_separate_from_host_arrival(monkeypatch):
    class Depth:
        def get_frame_number(self): return 17
        def get_timestamp(self): return 12_345.0  # SDK unit is milliseconds
        def get_frame_timestamp_domain(self):
            return SimpleNamespace(name="global_time")
        def get_data(self):
            return np.ones((2, 3), dtype=np.uint16)

    class Frames:
        def get_depth_frame(self): return Depth()

    class Pipeline:
        def wait_for_frames(self, _timeout_ms): return Frames()

    camera = RealCamera(_camera_cfg())
    camera._pipeline = Pipeline()
    camera._depth_scale = 0.001
    monkeypatch.setattr("real.real_camera.time.perf_counter", lambda: 77.25)

    frame = camera.get_frame()
    assert frame.timestamp == 77.25
    assert frame.source_timestamp_s == 12.345
    assert frame.source_timestamp_domain == "realsense_global_time"
    assert frame.host_wall_timestamp_s is not None
    assert frame.sequence_id == 17


def test_start_sets_and_reads_back_global_time_and_single_frame_queue(monkeypatch):
    values = {}
    waits = 0

    class Sensor:
        def get_depth_scale(self): return 0.001
        def supports(self, _option): return True
        def set_option(self, option, value): values[option] = value
        def get_option(self, option): return values[option]

    sensor = Sensor()

    class Device:
        def first_depth_sensor(self): return sensor
        def get_info(self, key):
            return "example_camera_0001" if key == "test_serial_key" else "5.16.0.1"

    class Profile:
        def get_device(self): return Device()

    class Pipeline:
        def start(self, _config): return Profile()
        def wait_for_frames(self, _timeout):
            nonlocal waits
            waits += 1
            return object()
        def stop(self): pass

    class Config:
        def enable_device(self, _serial): pass
        def enable_stream(self, *_args): pass

    fake_rs = SimpleNamespace(
        pipeline=Pipeline,
        config=Config,
        stream=SimpleNamespace(depth="depth", color="color"),
        format=SimpleNamespace(z16="z16", rgb8="rgb8"),
        camera_info=SimpleNamespace(serial_number="test_serial_key", firmware_version="firmware"),
        option=SimpleNamespace(
            global_time_enabled="global_time_enabled",
            frames_queue_size="frames_queue_size",
        ),
    )
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake_rs)
    camera = RealCamera(_camera_cfg())
    camera.start()

    assert values == {"global_time_enabled": 1.0, "frames_queue_size": 1.0}
    assert waits >= 480  # 16 s at 30 fps, covering global-time stabilization
    camera.stop()
