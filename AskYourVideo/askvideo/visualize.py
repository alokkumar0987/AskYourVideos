"""Presentation-quality result frames: clean boxes, a zoom inset for small
objects, and a caption that says what was asked, where it was found, and how
it was confirmed.

Kept separate from ``detect.draw_boxes``, which stays a fast overlay for the
live stream.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from askvideo.detect import Box

# Distinct, readable on both bright and dark footage (RGB).
PALETTE = [(16, 185, 129), (245, 158, 11), (99, 102, 241), (236, 72, 153), (14, 165, 233)]
CONTEXT = (226, 232, 240)  # large "context" objects such as a person: drawn quietly
CAPTION_BG = (15, 23, 42)
MUTED = (148, 163, 184)
INDIGO = (99, 102, 241)
GREEN = (16, 185, 129)

_FONTS = {
    "regular": ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "Helvetica.ttc"],
    "bold": ["seguisb.ttf", "segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf", "Helvetica.ttc"],
}
_FONT_DIRS = [Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu"), Path("/System/Library/Fonts")]


def _font(size: int, weight: str = "regular") -> ImageFont.ImageFont:
    for name in _FONTS[weight]:
        for d in _FONT_DIRS:
            try:
                return ImageFont.truetype(str(d / name), size)
            except OSError:
                continue
    return ImageFont.load_default(size)


def _area(b: Box) -> float:
    x0, y0, x1, y1 = b.xyxy
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _overlap(a: tuple, b: tuple) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def render_hit(
    rgb: np.ndarray,
    boxes: Sequence[Box],
    *,
    query: str,
    ts: float,
    labels: Sequence[str] = (),
    verified: bool | None = None,
    footnote: str = "",
) -> np.ndarray:
    """Return an RGB image: the frame with boxes and zoom inset, plus a caption strip.

    ``labels`` are the objects the query asked for; boxes with other labels are
    dropped. ``verified`` None means no detector was asked (plain retrieval).
    """
    h, w = rgb.shape[:2]
    s = w / 1280  # scale everything off a 720p design
    wanted = [b for b in boxes if not labels or b.label in labels]
    colors = {}
    for label in dict.fromkeys(b.label for b in sorted(wanted, key=_area)):
        colors[label] = PALETTE[len(colors) % len(PALETTE)]

    frame = Image.fromarray(rgb).convert("RGB")
    clean = frame.copy()  # the zoom inset is cut from the frame before anything is drawn on it
    draw = ImageDraw.Draw(frame, "RGBA")
    tag_font = _font(round(17 * s), "bold")

    # Big boxes (over a quarter of the frame) are context: thin and neutral.
    big = {id(b) for b in wanted if _area(b) > 0.25 * w * h}
    for b in sorted(wanted, key=_area, reverse=True):
        x0, y0, x1, y1 = b.xyxy
        color = CONTEXT if id(b) in big else colors[b.label]
        width = max(2, round((2 if id(b) in big else 4) * s))
        if id(b) not in big:
            draw.rectangle((x0, y0, x1, y1), fill=color + (40,))
        draw.rectangle((x0, y0, x1, y1), outline=color + (255,), width=width)
        tag = f"{b.label}  {b.conf:.2f}"
        tw, th = draw.textbbox((0, 0), tag, font=tag_font)[2:]
        pad = round(6 * s)
        ty = y0 - th - 2 * pad if y0 - th - 2 * pad > 0 else y0 + width
        draw.rounded_rectangle(
            (x0, ty, x0 + tw + 2 * pad, ty + th + 2 * pad), radius=round(5 * s),
            fill=color + (235,),
        )
        text_color = (15, 23, 42) if id(b) in big else (255, 255, 255)
        draw.text((x0 + pad, ty + pad - round(2 * s)), tag, font=tag_font, fill=text_color)

    small = [b for b in wanted if _area(b) < 0.03 * w * h]
    if small:
        _zoom_inset(frame, clean, draw, min(small, key=_area), colors, wanted, s)

    return np.asarray(_caption(frame, query, ts, labels, verified, footnote, s))


def _zoom_inset(frame, clean, draw, target: Box, colors, boxes, s) -> None:
    w, h = frame.size
    x0, y0, x1, y1 = target.xyxy
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    # Square crop around the object, shifted (not shrunk) at the frame edges.
    half = min(max(x1 - x0, y1 - y0) * 1.6 + 20 * s, h / 2)
    left = min(max(0, cx - half), w - 2 * half)
    top = min(max(0, cy - half), h - 2 * half)
    crop_box = tuple(round(v) for v in (left, top, left + 2 * half, top + 2 * half))
    side = round(h * 0.34)
    crop = clean.crop(crop_box).resize((side, side), Image.LANCZOS)
    k = side / (crop_box[2] - crop_box[0])
    ImageDraw.Draw(crop).rectangle(
        ((x0 - crop_box[0]) * k, (y0 - crop_box[1]) * k, (x1 - crop_box[0]) * k, (y1 - crop_box[1]) * k),
        outline=colors[target.label], width=max(2, round(3 * s)),
    )

    margin = round(18 * s)
    corners = [
        (margin, margin), (w - side - margin, margin),
        (margin, h - side - margin), (w - side - margin, h - side - margin),
    ]

    def cost(c):  # cover as little detected content as possible, stay away from the target
        rect = (c[0], c[1], c[0] + side, c[1] + side)
        covered = sum(_overlap(rect, b.xyxy) for b in boxes if b is not target) * 0.2
        covered += _overlap(rect, target.xyxy) * 10
        dist = abs(c[0] + side / 2 - cx) + abs(c[1] + side / 2 - cy)
        return covered - dist

    ix, iy = min(corners, key=cost)
    color = colors[target.label]
    # Connector from the box corner facing the inset, drawn first so the inset sits on top.
    mx, my = ix + side / 2, iy + side / 2
    start = (x0 if mx < cx else x1, y0 if my < cy else y1)
    draw.line((*start, mx, my), fill=color + (200,), width=max(2, round(2 * s)))
    border = max(3, round(4 * s))
    draw.rounded_rectangle(
        (ix - border, iy - border, ix + side + border, iy + side + border),
        radius=round(10 * s), fill=color + (255,),
    )
    frame.paste(crop, (ix, iy))
    font = _font(round(15 * s), "bold")
    label = f"zoom · {target.label}"
    tw, th = draw.textbbox((0, 0), label, font=font)[2:]
    pad = round(6 * s)
    draw.rounded_rectangle(
        (ix + pad, iy + side - th - 3 * pad, ix + tw + 3 * pad, iy + side - pad),
        radius=round(5 * s), fill=(15, 23, 42, 200),
    )
    draw.text((ix + 2 * pad, iy + side - th - 2.6 * pad), label, font=font, fill=(255, 255, 255))


def _caption(frame, query, ts, labels, verified, footnote, s) -> Image.Image:
    w, h = frame.size
    strip = round(118 * s)
    out = Image.new("RGB", (w, h + strip), CAPTION_BG)
    out.paste(frame, (0, 0))
    draw = ImageDraw.Draw(out)
    pad = round(26 * s)

    q_font = _font(round(28 * s), "bold")
    draw.text((pad, h + round(16 * s)), f"“{query}”", font=q_font, fill=(248, 250, 252))

    pill_font = _font(round(16 * s), "bold")
    x, y = pad, h + round(66 * s)

    def pill(text, color, check=False):
        nonlocal x
        tw, th = draw.textbbox((0, 0), text, font=pill_font)[2:]
        p = round(10 * s)
        icon = round(22 * s) if check else 0
        box = (x, y, x + tw + 2 * p + icon, y + th + round(12 * s))
        draw.rounded_rectangle(box, radius=round(14 * s), fill=color)
        tx = x + p
        if check:  # drawn check mark: fonts often lack the glyph
            cy = (box[1] + box[3]) / 2
            draw.line(
                [(tx, cy), (tx + 5 * s, cy + 5 * s), (tx + 13 * s, cy - 6 * s)],
                fill=(255, 255, 255), width=max(2, round(3 * s)),
            )
            tx += icon
        draw.text((tx, y + round(4 * s)), text, font=pill_font, fill=(255, 255, 255))
        x = box[2] + round(10 * s)

    m, sec = divmod(int(round(ts)), 60)
    pill(f"{m:02d}:{sec:02d}", INDIGO)
    if verified is True:
        pill("Verified: " + " + ".join(labels), GREEN, check=True)
    elif verified is False:
        pill("Not verified: " + " + ".join(labels), (71, 85, 105))
    if footnote:
        f_font = _font(round(16 * s))
        draw.text((x + round(4 * s), y + round(5 * s)), footnote, font=f_font, fill=MUTED)
    return out
