"""Fast regression tests for the local quality reconstruction contracts."""

from __future__ import annotations

import numpy as np

from panorama_spherical_stage import PanoramaValidationError, validate_panorama
from reconstruction_stage import GaussianData
from sharp_wrapper import _matrices_to_wxyz, _wxyz_to_matrices
from splat_compiler import compile_splat


def test_panorama_validation_accepts_cropped_and_full_layouts() -> None:
    validate_panorama(2048, 1024)
    validate_panorama(1347, 447)
    try:
        validate_panorama(512, 1024)
    except PanoramaValidationError:
        pass
    else:
        raise AssertionError("Portrait uploads must use Image to 3D")


def test_splat_compiler_preserves_valid_sharp_scales(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SPLAT_MAX_SCALE", "0.10")
    learned_scales = np.array([[0.08, 0.03, 0.005], [0.20, 0.01, 0.001]], dtype=np.float32)
    gaussians = GaussianData(
        positions=np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 3.0]], dtype=np.float32),
        scales=np.log(learned_scales),
        rotations=np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        colors=np.ones((2, 3), dtype=np.float32),
        opacities=np.ones((2, 1), dtype=np.float32),
    )
    path = compile_splat(gaussians, tmp_path / "quality.splat")
    records = np.fromfile(
        path,
        dtype=np.dtype([("pos_scale", np.float32, 6), ("rgba", np.uint8, 4), ("rot", np.uint8, 4)]),
    )
    np.testing.assert_allclose(records["pos_scale"][0, 3:], learned_scales[0], rtol=1e-6)
    np.testing.assert_allclose(records["pos_scale"][1, 3:], [0.10, 0.01, 0.001], rtol=1e-6)


def test_quaternion_matrix_conversion_remains_a_proper_rotation() -> None:
    source = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.9238795, 0.0, 0.3826834, 0.0]],
        dtype=np.float32,
    )
    matrices = _wxyz_to_matrices(source)
    recovered = _matrices_to_wxyz(matrices)
    roundtrip = _wxyz_to_matrices(recovered)
    np.testing.assert_allclose(roundtrip, matrices, atol=1e-5)
    np.testing.assert_allclose(np.linalg.det(roundtrip), np.ones(2), atol=1e-5)
