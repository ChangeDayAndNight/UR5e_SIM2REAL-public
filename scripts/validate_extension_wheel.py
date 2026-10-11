#!/usr/bin/env python
"""Import every project extension module from an isolated wheel target."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
from pathlib import Path
import pkgutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.training_env import preload_native_training_modules


_WHEEL_CHILD_OK = "WHEEL_KIT_CHILD_OK"
_WHEEL_CHILD_FAIL = "WHEEL_KIT_CHILD_FAIL"


def _within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def validate_installed_tree(
    target: Path,
    *,
    package: str = "ur5e_binpick",
    distribution: str = "ur5e_binpick",
    support_root: Path | None = None,
    expected_modules: int | None = None,
) -> list[str]:
    """Import the complete package tree and return imported module names."""

    target = target.resolve()
    if not target.is_dir():
        raise RuntimeError(f"wheel target does not exist: {target}")
    distributions = list(importlib.metadata.distributions(path=[str(target)]))
    matching = [
        item
        for item in distributions
        if (item.metadata.get("Name") or "").casefold() == distribution.casefold()
    ]
    if len(matching) != 1:
        raise RuntimeError(f"expected one {distribution!r} distribution in target; got {len(matching)}")
    direct_url = matching[0].read_text("direct_url.json") or ""
    if '"editable": true' in direct_url.casefold():
        raise RuntimeError("wheel gate received an editable install")

    previous_modules = {
        name: module
        for name, module in tuple(sys.modules.items())
        if name == package or name.startswith(package + ".")
    }
    for name in previous_modules:
        sys.modules.pop(name, None)

    inserted = [str(target)]
    if support_root is not None:
        inserted.append(str(support_root.resolve()))
    for value in reversed(inserted):
        sys.path.insert(0, value)
    imported: list[str] = []
    try:
        root_module = importlib.import_module(package)
        names = [package]
        names.extend(
            item.name
            for item in pkgutil.walk_packages(root_module.__path__, package + ".")
        )
        names = sorted(set(names))
        for name in names:
            module = importlib.import_module(name)
            origin = getattr(module, "__file__", None)
            if not origin or not _within(Path(origin), target):
                raise RuntimeError(f"module did not import from isolated wheel target: {name}: {origin}")
            imported.append(name)
        if expected_modules is not None and len(imported) != expected_modules:
            raise RuntimeError(
                f"wheel module count mismatch: expected={expected_modules} actual={len(imported)}"
            )
        return imported
    finally:
        for name in tuple(sys.modules):
            if name == package or name.startswith(package + "."):
                sys.modules.pop(name, None)
        sys.modules.update(previous_modules)
        for value in inserted:
            try:
                sys.path.remove(value)
            except ValueError:
                pass


def _emit_validation_result(imported: list[str], *, child: bool) -> None:
    print(f"[wheel-import] WHEEL_IMPORT_OK modules={len(imported)}", flush=True)
    for name in imported:
        print(f"[wheel-import][ok] {name}", flush=True)
    if child:
        print(_WHEEL_CHILD_OK, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()


def _emit_validation_failure(exc: Exception, *, child: bool) -> None:
    print(f"[wheel-import][FAIL] {type(exc).__name__}: {exc}", flush=True)
    if child:
        print(_WHEEL_CHILD_FAIL, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()


def _run_validation(args: argparse.Namespace, *, launch_app: bool, child: bool) -> int:
    """Complete validation and flush its verdict before closing a launched app."""

    simulation_app = None
    exit_code = 1
    try:
        if launch_app:
            preloaded = preload_native_training_modules()
            print(f"[wheel-import] native preload={','.join(preloaded)}", flush=True)
            from isaaclab.app import AppLauncher

            simulation_app = AppLauncher(headless=True).app
        imported = validate_installed_tree(
            args.target,
            support_root=args.support_root,
            expected_modules=args.expected_modules,
        )
        _emit_validation_result(imported, child=child)
        exit_code = 0
    except Exception as exc:
        _emit_validation_failure(exc, child=child)
        exit_code = 1

    if simulation_app is not None:
        try:
            simulation_app.close()
        except Exception as exc:  # pragma: no cover - actual Kit shutdown diagnosis
            print(f"[wheel-import][close-fail] {type(exc).__name__}: {exc}", flush=True)
            return 2
    return exit_code


def _run_isolated_kit_child(args: argparse.Namespace) -> int:
    command = [
        sys.executable,
        "-I",
        str(Path(__file__).resolve()),
        "--target",
        str(args.target.resolve()),
        "--app-child",
    ]
    if args.support_root is not None:
        command.extend(("--support-root", str(args.support_root.resolve())))
    if args.expected_modules is not None:
        command.extend(("--expected-modules", str(args.expected_modules)))
    try:
        proc = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=600,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[wheel-import][FAIL] isolated Kit child invocation failed: {exc}", flush=True)
        return 1

    output = proc.stdout or ""
    if output:
        sys.stdout.write(output)
        if not output.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()
    lines = [line.strip() for line in output.splitlines()]
    import_ok_count = sum(line.startswith("[wheel-import] WHEEL_IMPORT_OK modules=") for line in lines)
    child_ok_count = sum(line == _WHEEL_CHILD_OK for line in lines)
    child_fail_count = sum(line == _WHEEL_CHILD_FAIL for line in lines)
    if (
        proc.returncode != 0
        or import_ok_count != 1
        or child_ok_count != 1
        or child_fail_count != 0
    ):
        print(
            "[wheel-import][FAIL] isolated Kit child gate rejected result: "
            f"exit={proc.returncode} import_ok={import_ok_count} "
            f"child_ok={child_ok_count} child_fail={child_fail_count}",
            flush=True,
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="validate a non-editable extension wheel install")
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--support-root", type=Path)
    parser.add_argument("--expected-modules", type=int)
    parser.add_argument(
        "--launch-app",
        action="store_true",
        help="launch a headless Isaac application before importing Isaac-dependent modules",
    )
    parser.add_argument("--app-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.app_child:
        if args.launch_app:
            parser.error("--app-child and --launch-app are mutually exclusive")
        return _run_validation(args, launch_app=True, child=True)
    if args.launch_app:
        return _run_isolated_kit_child(args)
    return _run_validation(args, launch_app=False, child=False)


if __name__ == "__main__":
    raise SystemExit(main())
