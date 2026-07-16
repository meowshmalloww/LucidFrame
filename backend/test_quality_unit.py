"""Fast regression tests for the local quality reconstruction contracts."""

from __future__ import annotations

import numpy as np
from PIL import Image

from cubediff_stage import prepare_conditioning_image
from panorama_spherical_stage import PanoramaValidationError, validate_panorama
from reconstruction_stage import GaussianData
from sharp_wrapper import (
    _matrices_to_wxyz,
    _repair_source_color_outliers,
    _wxyz_to_matrices,
)
from sharp360_wrapper import (
    _build_coverage_guard,
    _profile_side_count,
    _profile_target_width,
    _repair_directional_color_outliers,
    _seam_overlap_degrees,
    _spherical_latitude_mask,
)
from splat_compiler import compile_splat
from image_metadata import preserved_image_metadata
from image_restoration_stage import _denoise_strength, _target_scale, _wrap_panorama_for_tiling
from scene_camera import camera_metadata_from_quality


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
    learned_scales = np.array(
        [[0.08, 0.03, 0.005], [0.85, 0.01, 0.001], [1.50, 0.02, 0.002]],
        dtype=np.float32,
    )
    gaussians = GaussianData(
        positions=np.array(
            [[0.0, 0.0, 2.0], [1.0, 0.0, 3.0], [-1.0, 0.0, 3.0]],
            dtype=np.float32,
        ),
        scales=np.log(learned_scales),
        rotations=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (3, 1)),
        colors=np.ones((3, 3), dtype=np.float32),
        opacities=np.ones((3, 1), dtype=np.float32),
    )
    path = compile_splat(gaussians, tmp_path / "quality.splat")
    records = np.fromfile(
        path,
        dtype=np.dtype([("pos_scale", np.float32, 6), ("rgba", np.uint8, 4), ("rot", np.uint8, 4)]),
    )
    np.testing.assert_allclose(records["pos_scale"][0, 3:], learned_scales[0], rtol=1e-6)
    np.testing.assert_allclose(records["pos_scale"][1, 3:], learned_scales[1], rtol=1e-6)
    np.testing.assert_allclose(records["pos_scale"][2, 3:], learned_scales[2], rtol=1e-6)


def test_sharp_camera_metadata_preserves_lens_and_scales_nearby_motion() -> None:
    camera = camera_metadata_from_quality(
        {
            "source_resolution": [2308, 1100],
            "focal_px": 1772.773,
            "depth_percentiles_m": [1.40435, 2.04822, 4.30362, 135.66136],
        }
    )
    assert camera is not None
    assert 66.12 < camera["horizontal_fov_deg"] < 66.13
    assert 0.16 < camera["recommended_lateral_m"] < 0.17
    assert 0.21 < camera["recommended_forward_m"] < 0.22
    assert 0.08 <= camera["move_speed_mps"] < 0.09


def test_legacy_sharp_camera_metadata_uses_conservative_motion() -> None:
    camera = camera_metadata_from_quality(
        {"source_resolution": [1015, 831], "focal_px": 909.561}
    )
    assert camera is not None
    assert 58.31 < camera["horizontal_fov_deg"] < 58.33
    assert camera["move_speed_mps"] == 0.12


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


def test_restoration_scale_targets_low_resolution_without_bloating_large_inputs() -> None:
    assert _target_scale(512, 512) == 2.0
    assert 1.1 < _target_scale(1280, 720) < 2.0
    assert _target_scale(1920, 1200) == 1.0
    assert 1.8 < _target_scale(570, 428, "artwork") < 1.9
    assert _target_scale(1015, 831, "artwork") == 1.0


def test_restoration_denoise_strength_is_bounded(monkeypatch) -> None:
    monkeypatch.setenv("RESTORATION_DENOISE_STRENGTH", "2")
    assert _denoise_strength() == 1.0
    monkeypatch.setenv("RESTORATION_DENOISE_STRENGTH", "-1")
    assert _denoise_strength() == 0.0


def test_artwork_restoration_uses_gentler_default(monkeypatch) -> None:
    monkeypatch.delenv("RESTORATION_DENOISE_STRENGTH", raising=False)
    assert _denoise_strength("artwork") == 0.1
    assert _denoise_strength("photo") == 0.55


def test_lossless_metadata_helper_preserves_focal_exif_and_icc(tmp_path) -> None:
    image = Image.new("RGB", (32, 24), "white")
    exif = Image.Exif()
    exif[274] = 6
    exif[34665] = {41989: 50}
    source = tmp_path / "source.png"
    image.save(source, "PNG", exif=exif, icc_profile=b"test-profile")
    with Image.open(source) as reopened:
        metadata = preserved_image_metadata(reopened)
    assert metadata["exif"].startswith(b"Exif")
    assert metadata["icc_profile"] == b"test-profile"
    roundtrip = tmp_path / "metadata.png"
    image.save(roundtrip, "PNG", exif=metadata["exif"])
    with Image.open(roundtrip) as reopened:
        assert reopened.getexif().get_ifd(34665)[41989] == 50


def test_panorama_quality_profiles_change_actual_reconstruction_density(monkeypatch) -> None:
    monkeypatch.setenv("SHARP360_ERP_WIDTH", "auto")
    monkeypatch.setenv("SHARP360_SIDE_COUNT", "auto")
    assert (_profile_target_width("balanced"), _profile_side_count("balanced")) == (2048, 4)
    assert (_profile_target_width("detail"), _profile_side_count("detail")) == (2560, 6)

    monkeypatch.setenv("SHARP360_ERP_WIDTH", "2560")
    monkeypatch.setenv("SHARP360_SIDE_COUNT", "8")
    assert (_profile_target_width("balanced"), _profile_side_count("balanced")) == (2560, 8)


def test_panorama_restoration_padding_wraps_horizontal_seam() -> None:
    source = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
    padded = _wrap_panorama_for_tiling(source, 1)

    assert padded.shape == (5, 7, 3)
    np.testing.assert_array_equal(padded[1:4, 0], source[:, -1])
    np.testing.assert_array_equal(padded[1:4, -1], source[:, 0])
    np.testing.assert_array_equal(padded[1:4, 1:-1], source)


def test_sharp360_seam_overlap_is_bounded(monkeypatch) -> None:
    monkeypatch.setenv("SHARP360_SEAM_OVERLAP_DEGREES", "9.5")
    assert _seam_overlap_degrees() == 9.5
    monkeypatch.setenv("SHARP360_SEAM_OVERLAP_DEGREES", "999")
    assert _seam_overlap_degrees() == 16.0


def test_sharp360_horizon_cap_handoff_has_no_triangular_corner_gap() -> None:
    import torch

    def ray(azimuth_degrees: float, latitude_degrees: float) -> list[float]:
        azimuth = np.deg2rad(azimuth_degrees)
        latitude = np.deg2rad(latitude_degrees)
        return [
            float(np.cos(latitude) * np.sin(azimuth)),
            float(np.sin(latitude)),
            float(np.cos(latitude) * np.cos(azimuth)),
        ]

    # At the 45-degree corner of a four-face horizon ring, the former |Y/Z|
    # rectangle stopped at ~21.4 degrees. Spherical ownership reaches the full
    # 29-degree seam everywhere, then hands the next sample to the pole cap.
    means = torch.tensor(
        [ray(45.0, 28.9), ray(45.0, 29.1)],
        dtype=torch.float32,
    )
    assert _spherical_latitude_mask(means, 58.0).tolist() == [True, False]


def test_sharp360_coverage_guard_is_sparse_complete_and_finite(monkeypatch) -> None:
    monkeypatch.setenv("SHARP360_GUARD_MAX_POINTS", "50000")
    panorama = np.zeros((64, 128, 3), dtype=np.uint8)
    panorama[..., 0] = 180
    disparity = np.full((64, 128), 0.2, dtype=np.float32)
    guard = _build_coverage_guard(panorama, disparity, target_radius=4.0)
    assert guard.count == 64 * 128
    assert np.isfinite(guard.positions).all()
    assert np.isfinite(guard.scales).all()
    assert np.isfinite(guard.rotations).all()
    assert float(np.median(np.linalg.norm(guard.positions, axis=1))) > 4.0
    np.testing.assert_allclose(np.linalg.norm(guard.rotations, axis=1), 1.0, atol=1e-5)


def test_directional_color_repair_changes_only_severe_bright_contradictions() -> None:
    panorama = np.full((8, 16, 3), 255, dtype=np.uint8)
    gaussians = GaussianData(
        positions=np.array([[0.0, 0.0, 4.0], [0.1, 0.0, 4.0]], dtype=np.float32),
        scales=np.zeros((2, 3), dtype=np.float32),
        rotations=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (2, 1)),
        colors=np.array([[0.01, 0.01, 0.01], [0.5, 0.5, 0.5]], dtype=np.float32),
        opacities=np.ones((2, 1), dtype=np.float32),
    )
    repaired = _repair_directional_color_outliers(gaussians, panorama, chunk_size=1)
    assert repaired == 1
    assert float(gaussians.colors[0].mean()) > 0.8
    np.testing.assert_allclose(gaussians.colors[1], [0.5, 0.5, 0.5])


def test_source_color_repair_uses_only_projected_severe_outliers() -> None:
    source = np.full((8, 8, 3), 255, dtype=np.uint8)
    positions = np.array(
        [[0.0, 0.0, 2.0], [0.2, 0.0, 2.0], [20.0, 0.0, 2.0]],
        dtype=np.float32,
    )
    colors = np.array(
        [[0.01, 0.01, 0.01], [0.5, 0.5, 0.5], [0.01, 0.01, 0.01]],
        dtype=np.float32,
    )
    repaired = _repair_source_color_outliers(positions, colors, source, focal_px=8.0)
    assert repaired == 1
    assert float(colors[0].mean()) > 0.8
    np.testing.assert_allclose(colors[1], [0.5, 0.5, 0.5])
    np.testing.assert_allclose(colors[2], [0.01, 0.01, 0.01])


def test_cubediff_conditioning_preserves_aspect_without_black_bands() -> None:
    source = np.zeros((80, 160, 3), dtype=np.uint8)
    source[..., 0] = 220
    source[..., 1] = 90
    conditioned = np.asarray(prepare_conditioning_image(Image.fromarray(source), 128))
    assert conditioned.shape == (128, 128, 3)
    assert int(conditioned[..., 0].min()) >= 215
    assert int(conditioned[..., 1].min()) >= 85
    assert int(conditioned[..., 2].max()) <= 5
