# LucidFrame — current product brief

This is the authoritative hackathon scope. Older experiments are not product claims.

## Keep the project

LucidFrame is a good Hack the Arts project when presented as an interactive artwork about crossing the edge of an image—not as a local clone of World Labs Marble.

The experience is simple: a visitor supplies an image, LucidFrame reconstructs a navigable Gaussian field, and bounded movement reveals the transition from photographed evidence to model-inferred structure. The technology is not hidden plumbing; its uncertainty is part of the work.

## The three modes

### Local Image to 3D — primary live demo

- Apple SHARP predicts metric anisotropic Gaussians at a fixed high inference resolution.
- The photographed viewpoint is crisp and nearby camera movement has real parallax.
- The viewer deliberately limits translation to the region the single view can support.
- It is not a complete 360° room and must not be described as one.

### Local Panorama to 3D — full look-around

- SPAG4D SHARP-360 reconstructs overlapping perspective faces and aligns them with a shared panoramic depth estimate.
- It produces learned 3D Gaussians rather than the old depth-projected spherical shell.
- True 2:1 input covers both poles. Wide and partial panoramas are accepted, but missing directions are extended and clearly labeled as inferred.
- Translation is bounded because a single panorama still has only one captured camera center.

### World Labs — optional hosted mode

- Uses a user-supplied key and explicit paid-generation confirmation.
- Keeps World Labs provenance visible and opens its native result in the supported viewer.
- It is not required for the local demo, and LucidFrame does not pretend hosted output was generated locally.

## Demo strategy

Use Local Image to 3D for the live generation. Keep one verified SHARP panorama in the Library for full-look-around demonstration, because a 121 MB, multi-face scene takes longer to transfer and render. Use a sharp, wide room photograph with clear foreground/midground separation; avoid motion blur, flat walls with no texture, and heavily compressed screenshots.

The safest pitch is:

> LucidFrame turns an image into millions of oriented particles, then makes the boundary between photographic evidence and machine inference physically explorable.

Do not claim perfect unseen geometry, unlimited walking, or World Labs-equivalent world generation. Those claims are not supported by any local single-view model on this 12 GB machine.

## Technical evidence

- [SHARP](https://apple.github.io/ml-sharp/) is the tested high-resolution single-image Gaussian predictor.
- [SPAG4D](https://github.com/cedarconnor/SPAG4d) supplies the tested SHARP-360 face alignment and merge path.
- [Scene4U](https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_Scene4U_Hierarchical_Layered_3D_Scene_Reconstruction_from_Single_Panoramic_Image_CVPR_2025_paper.pdf) demonstrates why convincing panorama reconstruction needs layered scene reasoning and inpainting; its reported A100-class workflow is not a safe live dependency for this laptop.
- [Matrix-3D](https://github.com/SkyworkAI/Matrix-3D) and other generative world systems require substantially more compute for full reconstruction than the tested local profile.

## Release constraint

The Apple SHARP model weights are non-commercial research only. Keep that disclosure in the UI and submission. A commercial product must replace or relicense the model.
