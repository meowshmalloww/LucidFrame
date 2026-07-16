"""Small helpers for retaining measured image metadata through PNG stages."""

from __future__ import annotations

from typing import Any

from PIL import Image


def preserved_image_metadata(image: Image.Image) -> dict[str, Any]:
    """Return Pillow save arguments that survive a lossless format conversion.

    Apple SHARP reads focal length from EXIF. LucidFrame normalises uploads to
    PNG, so dropping EXIF here silently changes a measured camera into SHARP's
    generic 30 mm fallback and can distort depth and parallax.
    """
    metadata: dict[str, Any] = {}
    try:
        exif = image.getexif()
        if exif and len(exif) > 0:
            metadata["exif"] = exif.tobytes()
    except (AttributeError, OSError, TypeError, ValueError):
        # Invalid metadata must never make an otherwise readable upload fail.
        pass

    icc_profile = image.info.get("icc_profile")
    if icc_profile:
        metadata["icc_profile"] = icc_profile
    return metadata
