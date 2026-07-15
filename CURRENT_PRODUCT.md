# LucidFrame — current product brief

This is the authoritative hackathon scope. Older experiments are not product claims.

## Keep the project

LucidFrame is a good Hack the Arts project when presented as an interactive artwork about crossing the edge of an image—not as a local clone of World Labs Marble.

The experience is simple: a visitor supplies an image, LucidFrame reconstructs a navigable Gaussian field, and free camera movement reveals the transition from photographed evidence to model-inferred structure. The technology is not hidden plumbing; its uncertainty is part of the work.

## The four modes

### Local Image to 3D — primary live demo

- Apple SHARP predicts metric anisotropic Gaussians at a fixed high inference resolution.
- Detail mode applies conservative weak-denoise restoration to small/degraded sources before SHARP; it never claims to recover factual missing pixels.
- The photographed viewpoint is crisp and nearby camera movement has real parallax.
- The viewer offers honest free-flight without pretending the raw splat has collision geometry; quality is strongest near the source camera.
- Standard rendering remains the unchanged default. The optional HD render toggle uses bounded supersampling and a small subpixel-footprint adjustment to reduce visible dot spacing without modifying the generated scene.
- It is not a complete 360° room and must not be described as one.

### Local Image to 360 Dream — generated full sphere

- OpenCubeDiff jointly generates six connected cubemap directions from one normal image; it does not make six independent requests.
- The uploaded direction remains the visual anchor. Back, side, ceiling, floor, and occluded content are explicitly labeled as generated artistic hypotheses.
- SPAG4D SHARP-360 predicts overlapping Gaussian fields and aligns them with one panoramic depth reference, providing nearby-view support around the full sphere.
- The tested 12 GB Balanced path produces roughly 3.35 million Gaussians and a 107 MB scene. Warm end-to-end generation is about two minutes on the target RTX 4080 Laptop GPU; cold model loading can extend that substantially.
- Exact learned-face ownership avoids ghosting. A 131k-point depth-aligned underlay fills residual angular pinholes, and only severe dark color predictions that contradict a bright source direction are repaired.
- Complete angular coverage does not guarantee unlimited translation. Moving far enough can still reveal surfaces that no single camera center supplied.

### Local Panorama to 3D — full look-around

- SPAG4D SHARP-360 reconstructs overlapping perspective faces and aligns them with a shared panoramic depth estimate. Balanced uses four horizon views; Detail uses a 2K alignment field and six.
- It produces learned 3D Gaussians rather than the old depth-projected spherical shell.
- True 2:1 input covers both poles. Wide and partial panoramas are accepted, but missing directions are extended and clearly labeled as inferred.
- Translation is unrestricted in Explore mode, but a single panorama still has only one captured camera center and no collision mesh.

### World Labs — optional hosted mode

- Uses a user-supplied key and explicit paid-generation confirmation.
- Keeps World Labs provenance visible and opens its native result in the supported viewer.
- It is not required for the local demo, and LucidFrame does not pretend hosted output was generated locally.

## Demo strategy

Use Local Image to 3D for the fastest live generation. Pre-generate one verified Image to 360 Dream scene for the full-look-around demonstration because the multi-model local path is slower and produces a roughly 107 MB scene. Use a sharp room photograph with clear foreground/midground separation; avoid motion blur, flat walls with no texture, and heavily compressed screenshots.

The safest pitch is:

> LucidFrame turns an image into millions of oriented particles, then makes the boundary between photographic evidence and machine inference physically explorable.

Do not claim perfect unseen geometry, physical collision, or World Labs-equivalent world generation. Free-flight navigation is unlimited; reliable reconstructed evidence is not.

The Library is project-based: individual delete, multi-select, Select all, and reclaimed-space reporting remove both the generated scene and its matching local source.

## Technical evidence

- [SHARP](https://apple.github.io/ml-sharp/) is the tested high-resolution single-image Gaussian predictor.
- [SPAG4D](https://github.com/cedarconnor/SPAG4d) supplies the tested SHARP-360 face alignment and merge path.
- [CubeDiff](https://arxiv.org/abs/2501.17162) motivates joint six-face synthesis; LucidFrame uses the compact [OpenCubeDiff](https://github.com/Juan5713/OpenCubeDiff) reimplementation that was measured on the target laptop.
- [Scene4U](https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_Scene4U_Hierarchical_Layered_3D_Scene_Reconstruction_from_Single_Panoramic_Image_CVPR_2025_paper.pdf) demonstrates why convincing panorama reconstruction needs layered scene reasoning and inpainting; its reported A100-class workflow is not a safe live dependency for this laptop.
- [Matrix-3D](https://github.com/SkyworkAI/Matrix-3D) and other generative world systems require substantially more compute for full reconstruction than the tested local profile.

## Release constraint

The Apple SHARP model weights are non-commercial research only. Keep that disclosure in the UI and submission. A commercial product must replace or relicense the model.
