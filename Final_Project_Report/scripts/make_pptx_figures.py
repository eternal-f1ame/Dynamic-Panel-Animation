#!/usr/bin/env python3
"""Build the two paper figures sourced from the final presentation.

(1) sakuga_motivation.png -- a 2x3 grid of representative frames from the
    slide-2 motivation GIFs, illustrating the kind of stylized motion the
    project targets.

(2) qualitative_results.png -- a 2x4 strip of frames from the two trained-
    model output GIFs on slide 11. Each row shows one generated clip
    sampled at four evenly-spaced timesteps, demonstrating the keyframe
    -> continuation behaviour qualitatively.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

PPTX_MEDIA = Path("/tmp/pptx_extract/ppt/media")
OUT_DIR = Path(__file__).resolve().parent.parent / "images"


def best_frame(gif_path: Path) -> Image.Image:
    """Return a single representative RGB frame from a GIF.

    Picks the frame near the temporal midpoint to capture motion-in-progress
    rather than a static opening pose.
    """
    im = Image.open(gif_path)
    n = getattr(im, "n_frames", 1)
    im.seek(n // 2)
    return im.convert("RGB")


def even_frames(gif_path: Path, n: int) -> list[Image.Image]:
    """Sample ``n`` evenly-spaced RGB frames across the GIF's duration."""
    im = Image.open(gif_path)
    total = getattr(im, "n_frames", 1)
    indices = [round(i * (total - 1) / max(n - 1, 1)) for i in range(n)]
    out = []
    for idx in indices:
        im.seek(idx)
        out.append(im.convert("RGB"))
    return out


def labelled_panel(img: Image.Image, label: str, panel_size: tuple[int, int]) -> Image.Image:
    """Letterbox an image into a fixed panel and stamp a small caption."""
    img = ImageOps.contain(img, panel_size)
    panel = Image.new("RGB", panel_size, "white")
    paste_x = (panel_size[0] - img.size[0]) // 2
    paste_y = (panel_size[1] - img.size[1]) // 2
    panel.paste(img, (paste_x, paste_y))
    if label:
        draw = ImageDraw.Draw(panel)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except OSError:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), label, font=font)
        pad = 4
        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        # Bottom-left badge with white background for legibility.
        draw.rectangle(
            [(6, panel_size[1] - bh - 2 * pad - 6), (6 + bw + 2 * pad, panel_size[1] - 6)],
            fill="white",
        )
        draw.text((6 + pad, panel_size[1] - bh - pad - 6), label, fill="black", font=font)
    return panel


def build_grid(images: list[Image.Image], cols: int, panel_size: tuple[int, int], gap: int = 8) -> Image.Image:
    rows = (len(images) + cols - 1) // cols
    width = cols * panel_size[0] + (cols - 1) * gap
    height = rows * panel_size[1] + (rows - 1) * gap
    canvas = Image.new("RGB", (width, height), "white")
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        x = c * (panel_size[0] + gap)
        y = r * (panel_size[1] + gap)
        canvas.paste(img, (x, y))
    return canvas


def make_motivation_figure() -> None:
    """2x3 grid of representative sakuga frames from the motivation slide."""
    panel_size = (480, 270)  # 16:9 panels keep the canvas tight.
    sources = [
        (PPTX_MEDIA / "image1.gif",  "Sakuga 1"),
        (PPTX_MEDIA / "image2.gif",  "Sakuga 2"),
        (PPTX_MEDIA / "image12.gif", "Sakuga 3"),
        (PPTX_MEDIA / "image14.gif", "Sakuga 4"),
        (PPTX_MEDIA / "image16.gif", "Sakuga 5"),
        (PPTX_MEDIA / "image25.gif", "Sakuga 6"),
    ]
    panels = [labelled_panel(best_frame(p), label, panel_size) for p, label in sources]
    grid = build_grid(panels, cols=3, panel_size=panel_size, gap=10)
    out_path = OUT_DIR / "sakuga_motivation.png"
    grid.save(out_path, "PNG", optimize=True)
    print(f"[OK] wrote {out_path}")


def make_qualitative_results_figure() -> None:
    """2x4 strip: two generated clips, four evenly-spaced frames each."""
    panel_size = (384, 216)
    rows = []
    sources = [
        (PPTX_MEDIA / "image19.gif", "Clip A"),
        (PPTX_MEDIA / "image20.gif", "Clip B"),
    ]
    for gif_path, row_label in sources:
        frames = even_frames(gif_path, n=4)
        labels = [f"{row_label} -- t=0", f"t=1/3", f"t=2/3", f"t=T"]
        rows.extend(labelled_panel(f, lbl, panel_size) for f, lbl in zip(frames, labels))
    grid = build_grid(rows, cols=4, panel_size=panel_size, gap=8)
    out_path = OUT_DIR / "qualitative_results.png"
    grid.save(out_path, "PNG", optimize=True)
    print(f"[OK] wrote {out_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_motivation_figure()
    make_qualitative_results_figure()


if __name__ == "__main__":
    main()
