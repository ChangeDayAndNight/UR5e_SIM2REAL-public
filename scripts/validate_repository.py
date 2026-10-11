#!/usr/bin/env python
"""Fail-closed static repository, format, and disclosure checks.

The file set is the effective prospective tree: tracked files plus non-ignored
new files, minus paths already removed from the working tree.  This makes the
same command useful before ``git add`` and in a clean CI checkout.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
import ipaddress
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 developer shell only; supported runtime is exact 3.11.15.
    import toml as tomllib


ROOT = Path(__file__).resolve().parents[1]
MAX_TRACKED_BYTES = 100 * 1024 * 1024
MAX_TEXT_SCAN_BYTES = 8 * 1024 * 1024

PARSE_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".toml", ".urdf"}
TEXT_SUFFIXES = {
    ".cfg",
    ".env",
    ".html",
    ".ini",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".rst",
    ".toml",
    ".txt",
    ".urdf",
    ".usd",
    ".usda",
    ".xml",
    ".yaml",
    ".yml",
}

STALE_REFERENCE_TOKENS = (
    "CLAUDE.md",
    "PROGRESS.md",
    "Future_Plan.md",
    "Failure_Check.md",
    "DEPLOYMENT.md",
    "METHODOLOGY.md",
    "README_MOVED.md",
    "configs/real.yaml",
    "gripper_problem.png",
    "gripper_report.html",
    "assets/ur5e_2fg7/README.md",
    "assets/d435i/README.md",
    "Phase0/rtde_connect_test.py",
    "Phase0\\rtde_connect_test.py",
    "Phase0/realsense_capture_test.py",
    "Phase0\\realsense_capture_test.py",
    "Phase3/Calib_HandEye.py",
    "Phase3\\Calib_HandEye.py",
    "Phase3/Calib_TcpOffset.py",
    "Phase3\\Calib_TcpOffset.py",
    "Phase3/Calib_Verify.py",
    "Phase3\\Calib_Verify.py",
    "Phase3/Verify_ObsParity.py",
    "Phase3\\Verify_ObsParity.py",
    "sync_bundle.py",
)
STALE_SCAN_EXEMPT = {
    ".gitignore",
    "README.md",  # cleanup inventory and history-only privacy warning
    # ★F97★ 배포 저장소(GitHub 공개 패키지)에서는 같은 원장이 PROJECT_LOG.md라는 이름으로 실린다.
    #   내용이 같으므로 같은 면제를 받는다(README.md는 그쪽에서 빠른 시작 안내가 된다).
    "PROJECT_LOG.md",
    "scripts/validate_repository.py",  # owns the forbidden token table
    "tests/test_repository_validator.py",  # synthetic negative fixtures
    "tests/test_real_verify_bundle.py",  # asserts deletion of the legacy root
}

PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"
)
TOKEN_PATTERNS = (
    ("AWS access key", re.compile(r"(?<![A-Z0-9])AKIA[A-Z0-9]{16}(?![A-Z0-9])")),
    ("GitHub token", re.compile(r"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{20,}")),
    ("GitHub fine-grained token", re.compile(r"github_pat_[A-Za-z0-9_]{20,}")),
    ("Google API key", re.compile(r"AIza[0-9A-Za-z_-]{35}")),
    ("Slack token", re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}")),
)
GENERIC_SECRET_RE = re.compile(
    r"(?i)(?:['\"])?\b(?:api[_-]?key|client[_-]?secret|access[_-]?token|password|passwd)\b"
    r"(?:['\"])?\s*(?::|=)\s*['\"](?P<value>[^'\"\r\n]{8,})['\"]"
)
SERIAL_RE = re.compile(
    r"(?i)(?:['\"])?\b(?:camera[_-]?)?serial(?:[_-]?number)?\b(?:['\"])?\s*"
    r"(?::|=)\s*(?:"
    r"(?P<quote>['\"])(?P<quoted>[A-Za-z0-9_-]{6,})(?P=quote)"
    r"|(?P<numeric>[0-9]{6,})(?![A-Za-z0-9_]))"
)
IPV4_CANDIDATE_RE = re.compile(
    r"(?<![0-9.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9.])"
)
WINDOWS_USER_PATH_RE = re.compile(
    r"(?i)(?:[A-Z]:[\\/]+Users[\\/]+)(?P<user>[^\\/\s'\"<>]+)"
)
POSIX_USER_PATH_RE = re.compile(
    r"/(?:home|Users)/(?P<user>[^/\s'\"<>]+)"
)
PLACEHOLDER_VALUES = {
    "changeme",
    "example",
    "placeholder",
    "redacted",
    "test",
    "unknown",
    "your_api_key",
    "your_password",
    "your_token",
}
SYNTHETIC_SERIAL_PREFIXES = ("test_", "example_", "placeholder_")
PLACEHOLDER_USERS = {
    "<user>",
    "example",
    "name",
    "runner",
    "runneradmin",
    "test",
    "user",
    "username",
}
TEST_NETS = tuple(
    ipaddress.ip_network(value)
    for value in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24")
)
# Keep rule data out of the content pattern it enforces; concatenation prevents
# the validator source itself from looking like a configured private address.
RFC1918_NETWORKS = tuple(
    ipaddress.ip_network(cidr)
    for cidr in ("10." + "0.0.0/8", "172." + "16.0.0/12", "192." + "168.0.0/16")
)


@dataclass
class ValidationReport:
    counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _git(root: Path, *args: str) -> bytes:
    proc = subprocess.run(
        ["git", *args],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        message = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {message}")
    return proc.stdout


def effective_repository_files(root: Path = ROOT) -> list[Path]:
    """Return tracked + prospective files that exist, excluding ignored output."""

    raw = _git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
    relative_names = {
        item.decode("utf-8", errors="surrogateescape")
        for item in raw.split(b"\0")
        if item
    }
    files: list[Path] = []
    for name in sorted(relative_names):
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe Git path: {name!r}")
        path = root.joinpath(*relative.parts)
        if path.is_file() or path.is_symlink():
            files.append(path)
    return files


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _read_text_for_scan(path: Path) -> str | None:
    try:
        if path.stat().st_size > MAX_TEXT_SCAN_BYTES:
            return None
        data = path.read_bytes()
    except OSError:
        return None
    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None


def parse_repository_file(path: Path) -> str | None:
    """Parse a supported source/config file and return its counter key."""

    suffix = path.suffix.lower()
    if suffix == ".py":
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        return "python"
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - CI installs PyYAML
            raise RuntimeError("PyYAML is required for repository validation") from exc
        list(yaml.safe_load_all(path.read_text(encoding="utf-8-sig")))
        return "yaml"
    if suffix == ".json":
        json.loads(path.read_text(encoding="utf-8-sig"))
        return "json"
    if suffix == ".toml":
        tomllib.loads(path.read_text(encoding="utf-8-sig"))
        return "toml"
    if suffix == ".urdf":
        ET.parse(path)
        return "urdf"
    return None


def generated_artifact_reason(relative: str) -> str | None:
    """Return why a path is forbidden in the source tree, if applicable."""

    path = PurePosixPath(relative)
    parts = path.parts
    lowered = tuple(part.casefold() for part in parts)
    if not parts:
        return None
    if lowered[0] == "real_verify":
        return "legacy duplicate deployment bundle"
    if lowered[0] in {"dist", "outputs"}:
        return "generated distribution/output root"
    if lowered[0] == "build":
        return "generated build root"
    if any(part in {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"} for part in lowered):
        return "cache directory"
    if "logs" in lowered:
        return "runtime log directory"
    if len(lowered) >= 2 and re.fullmatch(r"phase[0-9]+", lowered[0]) and lowered[1] == "result":
        return "generated phase result"
    if len(lowered) >= 2 and lowered[:2] in {
        ("deployment", "dist"),
        ("deployment", "evidence"),
        ("deployment", "wheelhouse"),
    }:
        return "generated deployment material"
    if path.name.casefold().startswith("real_verify") and path.suffix.casefold() == ".zip":
        return "stale deployment archive"
    return None


def sensitive_text_findings(relative: str, text: str) -> list[str]:
    """Find high-confidence current-tree credentials or machine identity."""

    findings: list[str] = []
    if PRIVATE_KEY_RE.search(text):
        findings.append("private key material")
    for label, pattern in TOKEN_PATTERNS:
        if pattern.search(text):
            findings.append(label)
    for match in GENERIC_SECRET_RE.finditer(text):
        value = match.group("value").strip().casefold()
        if value not in PLACEHOLDER_VALUES and not value.startswith(("${", "{{", "<")):
            findings.append("non-placeholder credential assignment")
            break
    for match in SERIAL_RE.finditer(text):
        value = (match.group("quoted") or match.group("numeric")).strip().casefold()
        if (
            value not in PLACEHOLDER_VALUES
            and not value.startswith(("${", "{{", "<") + SYNTHETIC_SERIAL_PREFIXES)
        ):
            findings.append("hard-coded device serial")
            break
    for match in IPV4_CANDIDATE_RE.finditer(text):
        try:
            address = ipaddress.ip_address(match.group(0))
        except ValueError:
            continue
        if any(address in network for network in TEST_NETS):
            continue
        if any(address in network for network in RFC1918_NETWORKS):
            findings.append("private IPv4 address")
            break
    for pattern in (WINDOWS_USER_PATH_RE, POSIX_USER_PATH_RE):
        for match in pattern.finditer(text):
            user = match.group("user").strip().casefold()
            if user not in PLACEHOLDER_USERS:
                findings.append("user-specific absolute path")
                break
    return sorted(set(findings))


def _git_symlink_paths(root: Path) -> set[str]:
    raw = _git(root, "ls-files", "-s", "-z")
    found: set[str] = set()
    for record in raw.split(b"\0"):
        if not record:
            continue
        header, separator, name = record.partition(b"\t")
        if separator and header.startswith(b"120000 "):
            found.add(name.decode("utf-8", errors="surrogateescape"))
    return found


def validate_repository(root: Path = ROOT) -> ValidationReport:
    report = ValidationReport(
        counts={"python": 0, "yaml": 0, "json": 0, "toml": 0, "urdf": 0, "markdown": 0}
    )
    try:
        files = effective_repository_files(root)
        git_symlinks = _git_symlink_paths(root)
    except RuntimeError as exc:
        report.errors.append(str(exc))
        return report

    relative_files = {_relative(path, root): path for path in files}
    for relative, path in relative_files.items():
        if path.is_symlink() or relative in git_symlinks:
            report.errors.append(f"symlink is forbidden: {relative}")
        try:
            size = path.stat().st_size
        except OSError as exc:
            report.errors.append(f"cannot stat {relative}: {exc}")
            continue
        if size > MAX_TRACKED_BYTES:
            report.errors.append(f"file exceeds 100 MiB: {relative} ({size} bytes)")
        artifact_reason = generated_artifact_reason(relative)
        if artifact_reason:
            report.errors.append(f"generated/stale artifact ({artifact_reason}): {relative}")

        if path.suffix.casefold() == ".md":
            report.counts["markdown"] += 1
        if path.suffix.casefold() in PARSE_SUFFIXES:
            try:
                key = parse_repository_file(path)
            except Exception as exc:
                report.errors.append(f"parse failed: {relative}: {type(exc).__name__}: {exc}")
            else:
                if key:
                    report.counts[key] += 1

        # Scan every small UTF-8 text file, including extensionless scripts and
        # uncommon config suffixes. Binary/large assets return None above.
        text = _read_text_for_scan(path)
        if text is None:
            continue
        for finding in sensitive_text_findings(relative, text):
            report.errors.append(f"sensitive current-tree content ({finding}): {relative}")
        if relative not in STALE_SCAN_EXEMPT:
            for token in STALE_REFERENCE_TOKENS:
                if token in text:
                    report.errors.append(f"stale deleted-path reference {token!r}: {relative}")

    # ★F97★ Markdown 정책 — 학습 저장소는 README.md 하나다. 배포 저장소(GitHub 공개 패키지)는
    #   같은 트리에 빠른 시작(README.md) · 배포 절차(DEPLOY_GUIDE.md) · 원장(PROJECT_LOG.md)과
    #   번들별 DEPLOY_README.md를 함께 싣는다. 그 ★고정된 집합★만 허용하고 나머지는 그대로 막는다
    #   (문서 난립 방지라는 원래 의도는 유지된다).
    allowed_markdown = {"README.md", "DEPLOY_GUIDE.md", "PROJECT_LOG.md"}
    markdown = sorted(
        relative for relative, path in relative_files.items() if path.suffix.casefold() == ".md"
    )
    unexpected = [
        relative for relative in markdown
        if relative not in allowed_markdown
        and not (relative.startswith("exported_experimental/") and relative.endswith("/DEPLOY_README.md"))
    ]
    if unexpected or "README.md" not in markdown:
        report.errors.append(f"project Markdown policy violated: {markdown!r}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="validate the prospective repository tree")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    report = validate_repository(args.root.resolve())
    for error in report.errors:
        print(f"[repository][error] {error}")
    counts = " ".join(f"{key}={value}" for key, value in report.counts.items())
    if report.ok:
        print(f"[repository] REPOSITORY_STATIC_OK {counts}")
        return 0
    print(f"[repository] FAIL errors={len(report.errors)} {counts}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
