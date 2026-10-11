from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_source_side_release_clis_self_reexec_in_frozen_training_env():
    for relative in ("scripts/build_parity_dataset.py", "deployment/build_bundle.py"):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert 'if __name__ == "__main__":' in source
        assert "reexec_in_training_python(__file__)" in source
        if relative.endswith("build_parity_dataset.py"):
            assert source.index("reexec_in_training_python(__file__)") < source.index(
                "import numpy as np"
            )


def test_canonical_posttrain_adoption_clis_self_reexec_before_pipeline_import():
    for relative in ("Phase1/BinPick_PostTrain.py", "Phase2/VisionPick_PostTrain.py"):
        source = (ROOT / relative).read_text(encoding="utf-8")
        guard = source.index("reexec_in_training_python(__file__)")
        pipeline = source.index("from scripts.adoption_pipeline import run_pipeline")
        assert guard < pipeline


def test_generated_runtime_verifier_does_not_require_training_environment():
    source = (ROOT / "deployment/verify_deployment.py").read_text(encoding="utf-8")
    assert "reexec_in_training_python" not in source


def test_non_static_preflight_reexecs_before_optional_dependency_imports():
    source = (ROOT / "scripts/preflight_training.py").read_text(encoding="utf-8")
    guard = source.index('if __name__ == "__main__" and "--static" not in sys.argv[1:]:')
    reexec = source.index("reexec_in_training_python(__file__)", guard)
    optional = source.index("from common.nurec_provenance import", reexec)
    assert guard < reexec < optional
