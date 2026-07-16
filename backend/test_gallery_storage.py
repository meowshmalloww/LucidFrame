"""Regression tests for safe local Library cleanup."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

import main


def test_gallery_batch_delete_removes_output_and_upload(tmp_path, monkeypatch) -> None:
    outputs = tmp_path / "outputs"
    uploads = tmp_path / "uploads"
    (outputs / "scene-123").mkdir(parents=True)
    (uploads / "scene-123").mkdir(parents=True)
    (outputs / "scene-123" / "final.splat").write_bytes(b"x" * 32)
    (uploads / "scene-123" / "input.png").write_bytes(b"y" * 16)
    monkeypatch.setattr(main, "OUTPUTS_DIR", outputs)
    monkeypatch.setattr(main, "UPLOADS_DIR", uploads)

    result = main._delete_gallery_projects(["scene-123"])

    assert result == {"deleted": ["scene-123"], "missing": [], "freed_bytes": 48}
    assert not (outputs / "scene-123").exists()
    assert not (uploads / "scene-123").exists()


def test_gallery_delete_rejects_traversal(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main, "OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(main, "UPLOADS_DIR", tmp_path / "uploads")
    with pytest.raises(HTTPException) as caught:
        main._delete_gallery_projects(["../outside"])
    assert caught.value.status_code == 422


def test_gallery_delete_removes_legacy_standalone_splat(tmp_path, monkeypatch) -> None:
    outputs = tmp_path / "outputs"
    uploads = tmp_path / "uploads"
    outputs.mkdir()
    uploads.mkdir()
    legacy = outputs / "old-room.splat"
    legacy.write_bytes(b"z" * 64)
    monkeypatch.setattr(main, "OUTPUTS_DIR", outputs)
    monkeypatch.setattr(main, "UPLOADS_DIR", uploads)

    result = main._delete_gallery_projects(["old-room"])

    assert result == {"deleted": ["old-room"], "missing": [], "freed_bytes": 64}
    assert not legacy.exists()


def test_gallery_delete_removes_nested_legacy_project(tmp_path, monkeypatch) -> None:
    outputs = tmp_path / "outputs"
    uploads = tmp_path / "uploads"
    project = outputs / "artwork-sharp-ab" / "photo"
    project.mkdir(parents=True)
    uploads.mkdir()
    (project / "final.splat").write_bytes(b"s" * 96)
    monkeypatch.setattr(main, "OUTPUTS_DIR", outputs)
    monkeypatch.setattr(main, "UPLOADS_DIR", uploads)

    result = main._delete_gallery_projects(["artwork-sharp-ab/photo"])

    assert result == {
        "deleted": ["artwork-sharp-ab/photo"],
        "missing": [],
        "freed_bytes": 96,
    }
    assert not project.exists()
    assert not (outputs / "artwork-sharp-ab").exists()


@pytest.mark.parametrize("scene_id", ["../outside", "group/../../outside", "/outside", "group\\..\\outside"])
def test_nested_gallery_delete_rejects_traversal(tmp_path, monkeypatch, scene_id) -> None:
    monkeypatch.setattr(main, "OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(main, "UPLOADS_DIR", tmp_path / "uploads")
    with pytest.raises(HTTPException) as caught:
        main._delete_gallery_projects([scene_id])
    assert caught.value.status_code == 422
