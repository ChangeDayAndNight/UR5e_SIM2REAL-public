"""Pure constants for the one supported Isaac training host contract.

This module deliberately imports no Isaac, CUDA, or packaging runtime.  Both the
preflight and immutable training/release receipts consume it so version policy
cannot drift between those gates.
"""
from __future__ import annotations

TRAINING_PYTHON_VERSION = "3.11.15"
TRAINING_OS = "Windows"
TRAINING_PYTHON_IMPLEMENTATION = "CPython"
MIN_WINDOWS_BUILD = 22000
MIN_TRAINING_VRAM_MIB = 7.5 * 1024.0

ISAAC_LAB_COMMIT = "37ddf626871758333d6ed89cf64ad702aef127d0"
ISAAC_LAB_RELEASE = "2.3.2"
ISAAC_LAB_TAG = "v2.3.2"
CUDA_RUNTIME = "12.8"

# Values are tuples because the NVIDIA top-level Isaac Sim metapackage has been
# observed with either the public release spelling or four-component metadata.
# A receipt records the one exact installed value; every other distribution is
# single-valued.
EXPECTED_TRAINING_DISTRIBUTIONS: dict[str, tuple[str, ...]] = {
    "isaacsim": ("5.1.0", "5.1.0.0"),
    "isaacsim-core": ("5.1.0.0",),
    "isaacsim-kernel": ("5.1.0.0",),
    "isaaclab": ("0.54.2",),
    "isaaclab_assets": ("0.2.4",),
    "isaaclab_rl": ("0.4.7",),
    "isaaclab_tasks": ("0.11.12",),
    "rsl-rl-lib": ("3.1.2",),
    "tensordict": ("0.13.0",),
    "torch": ("2.7.0+cu128",),
    "torchvision": ("0.22.0+cu128",),
    "torchaudio": ("2.7.0+cu128",),
    "numpy": ("1.26.0",),
    "gymnasium": ("1.2.1",),
    "matplotlib": ("3.10.3",),
    "openpyxl": ("3.1.5",),
    "pandas": ("2.2.3",),
    "packaging": ("23.0",),
    "setuptools": ("81.0.0",),
    "wheel": ("0.45.1",),
    "click": ("8.1.7",),
    "psutil": ("5.9.8",),
    "typing_extensions": ("4.12.2",),
    "starlette": ("0.45.3",),
    "onnx": ("1.21.0",),
    "ur5e_binpick": ("0.1.0",),
}

ISAAC_SIM_METADATA = {
    "isaacsim": "5.1.0.0",
    "isaacsim-core": "5.1.0.0",
    "isaacsim-kernel": "5.1.0.0",
}
ISAAC_LAB_DISTRIBUTIONS = {
    "isaaclab": "0.54.2",
    "isaaclab_assets": "0.2.4",
    "isaaclab_rl": "0.4.7",
    "isaaclab_tasks": "0.11.12",
}
