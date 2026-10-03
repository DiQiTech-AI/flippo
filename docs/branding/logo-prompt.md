# Flippo logo generation prompt

The repository logo was generated as a raster PNG with the built-in ImageGen tool. The repository asset is stored at `docs/assets/flippo-logo.png` (2079 × 756 px, solid background).

## Prompt

```text
Use case: logo-brand
Asset type: GitHub README horizontal logo lockup for an open-source document question-answering project
Primary request: Create a polished playful brand logo for Flippo, a document Q&A tool whose answers cite their sources.
Scene/backdrop: clean solid ivory near-white background with ample balanced whitespace
Subject: a charming compact open-book/document mascot on the left; one page is caught mid-flip and its curl subtly suggests the letter F; give the mascot two small cheerful eyes and a restrained friendly smile. On the right, place the brand wordmark, with the Chinese name below.
Style/medium: crisp vector-like flat raster illustration, simple strong rounded shapes, professional open-source product identity, recognizable at README thumbnail size, warm and clever rather than corporate
Composition/framing: wide horizontal lockup, mascot about one-third of width, wordmark about two-thirds, vertically centered, generous outer margins
Color palette: deep teal and soft mint as main colors, one small warm coral bookmark accent, ivory backdrop
Text (verbatim): "Flippo" and "翻翻文档"
Typography: bold custom rounded wordmark EXACTLY "Flippo" (capital F followed by lowercase l-i-p-p-o); directly below it, a smaller neat Chinese line EXACTLY "翻翻文档"
Constraints: preserve exact spelling and character order; flat colors; crisp edges; high resolution; balanced spacing; suitable as the lead graphic in a GitHub README
Avoid: extra text, slogans, misspellings, gradients, shadows, mockups, devices, watermarks, busy detail, photorealism, 3D effects
```

Generation mode: built-in ImageGen, with `transparent_background=false`.

Before publishing another generated variant, inspect both text lines at original resolution. Image models can misspell Latin or Chinese text even when the prompt requires exact wording.
