# Hack the Arts submission kit — LucidFrame

## One-line pitch

**LucidFrame turns an image into a place you can enter, then makes the boundary between photographic evidence and machine inference physically explorable.**

## Project description

A photograph gives us one position in space, but imagination immediately fills in the rest. LucidFrame turns that reflex into an interactive computational art medium.

Visitors upload a photograph, painting, or panorama. A local neural renderer predicts millions of oriented 3D Gaussian particles, and LucidFrame compiles them into a navigable field rendered live in the browser. Camera movement reveals both the convincing depth and the places where a single captured viewpoint can no longer provide evidence.

The work is about the ethics and poetry of spatial hallucination—the moment an image stops being proof and becomes a place. It could not exist without learned spatial priors, metric depth, Gaussian reconstruction, GPU inference, and the visitor's live movement.

## How it works

- Next.js provides the image workflow, Library, Settings, and immersive browser viewer.
- FastAPI runs local GPU inference and streams progress over WebSockets.
- For a normal image, Apple SHARP predicts about 1.18 million metric anisotropic Gaussians at high inference resolution.
- For a panorama, SPAG4D SHARP-360 reconstructs overlapping perspective Gaussian fields, aligns their depth, and merges them into one look-around scene.
- The backend filters invalid predictions, writes model provenance, and compiles a browser-ready `.splat` file.
- An optional World Labs route is clearly identified as a paid hosted service and is not required for the local artwork.

## Why it fits the theme

LucidFrame is not a static AI image and not a generic file converter. Its artistic form is the relationship among one image, a spatial prediction model, and a moving person. The visitor discovers where the image still supports a viewpoint and where a machine's learned geometry begins to speak.

## Suggested 90-second demo

1. Show the source: “Every photograph remembers only one place to stand.”
2. Start Local Image to 3D and show the live model activity.
3. Enter the finished field and drag to inspect fine depth and orientation.
4. Enable Move and make one short sideways translation to reveal parallax.
5. Open a pre-generated panorama from the Library and look behind the source direction.
6. End at the original viewpoint: “LucidFrame makes the instant where documentation becomes imagination navigable.”

## Submission checklist

- [ ] Record the complete local generation and viewer flow on the demo machine.
- [ ] Use a sharp interior or artwork with clear foreground, midground, and background structure.
- [ ] Include one true 2:1 panorama for the panorama demonstration.
- [ ] Publish the source repository and describe which mode produced each example.
- [ ] Do not claim unlimited movement, factual unseen geometry, or a World Labs-equivalent local world model.
- [ ] Disclose the Apple SHARP non-commercial research license.

## Third-party disclosure

- **Apple SHARP** — high-resolution single-image metric Gaussian prediction; released weights are non-commercial research only.
- **SPAG4D SHARP-360** — MIT-licensed panorama alignment/merge code using the SHARP predictor.
- **Depth Anything V2** — local panorama depth alignment and automatic fallback reconstruction.
- **gsplat.js** — real-time browser rendering of Gaussian splats.
- **World Labs** — optional hosted API mode using a visitor-supplied key and credits.

## Useful links

- [Hack the Arts](https://hackthearts.devpost.com/)
- [Apple SHARP project](https://apple.github.io/ml-sharp/)
- [Apple SHARP implementation](https://github.com/apple/ml-sharp)
- [SPAG4D](https://github.com/cedarconnor/SPAG4d)
- [3D Gaussian Splatting reference](https://github.com/graphdeco-inria/gaussian-splatting)
