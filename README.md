# Text2Ink — Handwritten PDF Renderer

Text2Ink converts digital PDFs into images that convincingly resemble handwritten pages. It is intended as a practical, easily extensible toolkit for generating stylized handwritten renderings for design, mockups, or creative workflows.

Key ideas implemented:

- Per-character variation (position, rotation, scale, kerning)
- Simulated pen pressure and ink pooling
- Local ink bleed and smudging
- Paper texture, slight perspective warp and vignette

## Features

- Extract text blocks from PDFs using PyMuPDF (`fitz`).
- Render glyphs with Pillow and apply per-character transformations.
- Post-process with OpenCV (ink bleed, noise, blur, perspective).
- Configurable parameters for spacing, pressure, bleed and texture.

## Quickstart

1. Create and activate a Python virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install the Python dependencies:

```bash
pip install -r requirements.txt
```

3. Put a TrueType font (.ttf) in the `TTF/` folder or update `font_path` in `main.py` to point to a font you prefer.

4. Run the renderer:

```bash
uv run main.py
```

Generated images are written to `output_images/` by default.

## Configuration & Tuning

Most of the tuning parameters live in `main.py` (search for `draw_handwritten_line`, `apply_pressure_variation`, `add_ink_bleed`, and `add_paper_texture`). Important knobs:

- `word_gap_base` — controls base spacing between words.
- Per-character jitter ranges (x/y offsets and rotation) — increase for more irregular handwriting.
- Pressure parameters — adjust stroke thickness and darkness.
- Ink-bleed probability and kernel sizes — larger values give stronger bleed.
- Paper texture intensity — higher values make the background more textured/aged.

Recommendation: change one parameter at a time and re-run to evaluate visual impact.

## Implementation notes

- `main.py` is organized as a pipeline: extract text → render per-block → apply distortions → add texture.
- Pillow is used for glyph rasterization and compositing; OpenCV and NumPy handle image-level filters and warps.
- The code is intentionally readable and editable: swap fonts, tune ranges, or add new post-processing steps.

## Examples

See the `output_images/` folder after running the script for rendered examples. (Add your own sample PDFs to the `pdf/` folder.)

## Contributing

Contributions are welcome. Good first changes:

- Add CLI flags to control the most-used parameters.
- Add style presets (e.g., "neat", "loopy", "scribbly").
- Implement multi-glyph alternates or small vector perturbations for higher realism.

Please open issues or pull requests with concise, focused changes.

## License

This project is released under the Apache License Version 2.0, January 2004.
