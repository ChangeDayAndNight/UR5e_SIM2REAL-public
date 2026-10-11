"""Installation script for the 'ur5e_binpick' Isaac Lab extension.

학습 PC(isaac51)에서 편집 설치:  pip install -e source/ur5e_binpick
(배포 PC에는 설치하지 않는다 — 확장은 sim 학습 전용, README.md §4)
"""

import os

import toml
from setuptools import find_packages, setup

EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# 설치 전 최소 의존성. isaaclab·isaaclab_rl·isaaclab_tasks는 isaac51에
# Isaac Lab 설치 시 이미 존재하므로 여기 나열하지 않는다(extension.toml [dependencies] 참조).
INSTALL_REQUIRES = [
    "psutil==5.9.8",
]

setup(
    name="ur5e_binpick",
    packages=find_packages(),
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="MIT",
    include_package_data=True,
    python_requires=">=3.11,<3.12",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 5.1.0",
    ],
    zip_safe=False,
)
