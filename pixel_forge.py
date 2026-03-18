#!/usr/bin/env python3
"""
pixel_forge.py — Convert a smooth icon font to pixel/bitmap style.

Rasterises each glyph at a fixed pixel grid, then re-vectorises the
bitmap as a rectilinear (all-right-angle, no curves) outline.

For glyphs that are left-right or top-bottom symmetric in their original
vector outline, a subpixel shift is applied before rasterisation so that
the symmetry axis lands on an integer or half-integer pixel boundary.
This prevents the grayscale anti-aliasing from rounding differently on
the two sides, which would produce visibly asymmetric pixel art.

Dependencies:
    pip install fonttools freetype-py numpy

Usage:
    python pixel_forge.py fonts/original.ttf
    python pixel_forge.py --grid 16 fonts/original.ttf
    python pixel_forge.py --grid 20 --output fonts/ fonts/original.ttf

Output filename: <input_stem>_<grid>.ttf  (in same dir as input by default)
"""

import argparse
import ctypes
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input", help="Input TTF file (e.g. fonts/original.ttf)")
    ap.add_argument(
        "--grid", "-g", type=int, default=16,
        help="Pixel grid height in pixels (default: 16)",
    )
    ap.add_argument(
        "--output", "-o", default=None,
        help="Output path: directory → auto-named file; file → used as-is "
             "(default: same dir as input, named PlurkIconFont_<grid>.ttf)",
    )
    ap.add_argument(
        "--threshold", "-t", type=int, default=96,
        help="Grayscale fill threshold 1-255: lower=fatter strokes, higher=thinner "
             "(default: 96)",
    )
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Symmetry detection from the original vector outline
# ---------------------------------------------------------------------------

def _on_curve_points(font, glyph_name: str) -> list:
    """
    Return all on-curve endpoints from a glyph's outline.
    Off-curve control points are intentionally excluded: they can be
    asymmetric even in a geometrically symmetric glyph.
    """
    from fontTools.pens.recordingPen import RecordingPen
    glyph_set = font.getGlyphSet()
    if glyph_name not in glyph_set:
        return []
    pen = RecordingPen()
    try:
        glyph_set[glyph_name].draw(pen)
    except Exception:
        return []

    pts = []
    for op, args in pen.value:
        if op == "moveTo" and args:
            pts.append(args[0])
        elif op == "lineTo" and args:
            pts.append(args[0])
        elif op in ("qCurveTo", "curveTo") and args:
            pts.append(args[-1])   # last arg is always the on-curve endpoint
    return [p for p in pts if p is not None]


def detect_symmetry(font, glyph_name: str, tolerance: int = 2, score_threshold: float = 0.9):
    """
    Detect left-right and top-bottom symmetry from on-curve outline points.

    For each on-curve point P, checks whether its mirror across the bounding-
    box centre exists in the outline (within ±tolerance font units).

    Returns
    -------
    lr : bool   — glyph is left-right symmetric
    tb : bool   — glyph is top-bottom symmetric
    cx : float  — x centre of the LR symmetry axis (font units)
    cy : float  — y centre of the TB symmetry axis (font units)
    """
    pts = _on_curve_points(font, glyph_name)
    if not pts:
        return False, False, 0.0, 0.0

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2

    # Build integer-rounded point set for fast lookup
    pt_set = {(round(x), round(y)) for x, y in pts}

    def has_mirror(mx, my):
        return any(
            (mx + dx, my + dy) in pt_set
            for dx in range(-tolerance, tolerance + 1)
            for dy in range(-tolerance, tolerance + 1)
        )

    lr_hits = sum(1 for x, y in pts if has_mirror(round(2 * cx - x), round(y)))
    tb_hits = sum(1 for x, y in pts if has_mirror(round(x), round(2 * cy - y)))
    n = len(pts)

    return (lr_hits / n >= score_threshold,
            tb_hits / n >= score_threshold,
            cx, cy)


# ---------------------------------------------------------------------------
# Alignment shift computation
# ---------------------------------------------------------------------------

def alignment_shift_26dot6(
    center_font: float,
    edge_font: float,      # the nearer outer edge (xMin for LR, yMin for TB)
    scale: float,          # font units per pixel = UPM / grid_size
    threshold: int = 96,   # grayscale threshold (used to judge edge coverage)
) -> int:
    """
    Compute the subpixel shift (in freetype 26.6 fixed-point pixels) needed
    to snap `center_font` to the nearest 0.5-pixel boundary.

    The shift direction is chosen so that the glyph's outer edge pixel keeps
    at least `threshold/255` coverage — i.e., the outermost pixel stays
    filled.  If only one direction satisfies this, that direction is used;
    otherwise the nearer snap is preferred.

    Returns delta_26dot6 (int).
    """
    center_px = center_font / scale

    lo = math.floor(center_px * 2) / 2   # nearest lower 0.5-pixel mark
    hi = math.ceil(center_px * 2) / 2    # nearest upper 0.5-pixel mark

    if lo == hi:
        return 0   # already exactly on a half-pixel boundary

    threshold_frac = threshold / 255.0

    def edge_coverage(delta_26dot6: int) -> float:
        """Coverage of the outermost pixel after applying this shift."""
        delta_font = delta_26dot6 * scale / 64.0
        new_edge = edge_font + delta_font
        # fractional position within its pixel column (0 = exactly on boundary)
        frac = (new_edge / scale) % 1.0
        if frac < 0:
            frac += 1.0
        # coverage = how much of the pixel is inside the glyph
        return (1.0 - frac) if frac > 0 else 1.0

    delta_lo = int(round((lo - center_px) * 64))
    delta_hi = int(round((hi - center_px) * 64))

    cov_lo = edge_coverage(delta_lo)
    cov_hi = edge_coverage(delta_hi)

    lo_ok = cov_lo >= threshold_frac
    hi_ok = cov_hi >= threshold_frac

    if lo_ok and hi_ok:
        # Both fine — take the nearer one
        return delta_lo if abs(lo - center_px) <= abs(hi - center_px) else delta_hi
    elif lo_ok:
        return delta_lo
    elif hi_ok:
        return delta_hi
    else:
        # Neither keeps the edge; take whichever loses least
        return delta_lo if cov_lo >= cov_hi else delta_hi


# ---------------------------------------------------------------------------
# Rasterisation (freetype-py)
# ---------------------------------------------------------------------------

def load_face(font_path: str, grid_size: int):
    try:
        import freetype
    except ImportError:
        sys.exit("freetype-py is required.  Install with: pip install freetype-py")
    face = freetype.Face(font_path)
    face.set_pixel_sizes(0, grid_size)
    return face


def _set_ft_transform(face, dx_26dot6: int, dy_26dot6: int = 0) -> None:
    """
    Apply (or clear) a subpixel translation via FT_Set_Transform.
    face.set_transform(None, ...) is broken in freetype-py 2.5.x, so we
    call the underlying C function directly via ctypes.
    """
    import freetype
    vec = freetype.FT_Vector(dx_26dot6, dy_26dot6)
    freetype.FT_Set_Transform(face._FT_Face, None, ctypes.byref(vec))


def rasterise(face, charcode: int, threshold: int = 96, delta_26dot6: int = 0):
    """
    Render one codepoint at the face's current pixel size using grayscale
    anti-aliasing + threshold.

    If delta_26dot6 != 0, a subpixel horizontal shift is applied before
    rendering (to align the symmetry axis to a pixel boundary) and cleared
    afterwards.

    Returns
    -------
    grid        : bool ndarray [rows, cols], True = filled, row 0 = top
    advance_px  : advance width in pixels
    bm_left     : pixels from pen-origin to left edge of bitmap
    bm_top      : pixels from baseline to top edge of bitmap (positive = above)
    """
    import freetype

    if delta_26dot6 != 0:
        _set_ft_transform(face, delta_26dot6)

    face.load_char(charcode, freetype.FT_LOAD_DEFAULT | freetype.FT_LOAD_NO_HINTING)
    face.glyph.render(freetype.FT_RENDER_MODE_NORMAL)

    if delta_26dot6 != 0:
        _set_ft_transform(face, 0)   # reset

    slot = face.glyph
    bm   = slot.bitmap
    advance_px = slot.advance.x >> 6
    bm_left    = slot.bitmap_left
    bm_top     = slot.bitmap_top

    rows, cols, pitch = bm.rows, bm.width, abs(bm.pitch)
    if rows == 0 or cols == 0:
        return np.zeros((0, 0), dtype=bool), advance_px, bm_left, bm_top

    buf  = np.frombuffer(bytes(bm.buffer), dtype=np.uint8)
    gray = buf[: rows * pitch].reshape(rows, pitch)[:, :cols]
    return gray >= threshold, advance_px, bm_left, bm_top


# ---------------------------------------------------------------------------
# Bitmap → rectilinear vector outline
# ---------------------------------------------------------------------------

def _boundary_edges(grid: np.ndarray) -> defaultdict:
    """
    Build directed half-edges for the pixel boundary (y=0 at bottom).

    Winding follows TrueType convention:
      - Outer contours (filled regions)  → CCW
      - Inner contours (holes)           → CW
    This falls out naturally from the edge directions chosen below.
    Using a list per start-point survives diagonal-touch ambiguity.
    """
    rows, cols = grid.shape
    em = defaultdict(list)

    for r in range(rows):
        for c in range(cols):
            if not grid[r, c]:
                continue

            # Bottom-left and top-right corners of this pixel in y-up coords
            x0, y0 = c,     rows - r - 1
            x1, y1 = c + 1, rows - r

            t  = (r == 0)        or not grid[r - 1, c]
            b  = (r == rows - 1) or not grid[r + 1, c]
            l  = (c == 0)        or not grid[r,     c - 1]
            ri = (c == cols - 1) or not grid[r,     c + 1]

            # Each edge is directed so that the interior is on the LEFT,
            # producing CCW outer contours:
            if t:  em[(x1, y1)].append((x0, y1))   # top    → go left
            if b:  em[(x0, y0)].append((x1, y0))   # bottom → go right
            if l:  em[(x0, y1)].append((x0, y0))   # left   → go down
            if ri: em[(x1, y0)].append((x1, y1))   # right  → go up

    return em


def _trace(em: defaultdict) -> list:
    """Follow directed edges into closed loops (handles multi-edges per node)."""
    remaining = {k: list(v) for k, v in em.items()}
    contours = []

    while True:
        start = next((k for k, v in remaining.items() if v), None)
        if start is None:
            break

        path = [start]
        curr = remaining[start].pop(0)

        while curr != start:
            path.append(curr)
            nexts = remaining.get(curr, [])
            if not nexts:
                break
            curr = nexts.pop(0)

        contours.append(path)

    return contours


def _simplify(pts: list) -> list:
    """Remove collinear midpoints — keep only corner vertices."""
    n = len(pts)
    if n < 3:
        return pts
    keep = []
    for i in range(n):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if cross != 0:
            keep.append(b)
    return keep or pts


def bitmap_to_contours(grid: np.ndarray) -> list:
    """
    Full pipeline: binary pixel grid → list of simplified contours.
    Each contour is a list of (x, y) int vertices in pixel units (y=0 at bottom).
    """
    em = _boundary_edges(grid)
    if not em:
        return []
    raw = _trace(em)
    simplified = [_simplify(c) for c in raw]
    return [c for c in simplified if len(c) >= 3]


# ---------------------------------------------------------------------------
# Post-rasterisation symmetry enforcement
# ---------------------------------------------------------------------------

def enforce_lr_symmetry(grid: np.ndarray, score_threshold: float = 0.75):
    """
    If the majority of content rows are already left-right symmetric, OR-fill
    the remaining asymmetric rows to enforce full symmetry.

    The mirror axis is detected as the most common snapped row-centre (0.5
    granularity).  If fewer than `score_threshold` of content rows agree on
    that axis the grid is returned unchanged.

    Returns the (possibly modified) grid.
    """
    from collections import Counter

    rows, cols = grid.shape
    centers = []
    for r in range(rows):
        filled = np.where(grid[r])[0]
        if len(filled) == 0:
            continue
        centers.append((int(filled[0]) + int(filled[-1])) / 2.0)

    if not centers:
        return grid

    snapped = [round(c * 2) / 2 for c in centers]
    axis_counts = Counter(snapped)
    best_axis, best_count = axis_counts.most_common(1)[0]

    if best_count / len(centers) < score_threshold:
        return grid

    new_grid = grid.copy()
    for r in range(rows):
        for c in range(cols):
            if not grid[r, c]:
                continue
            mirror_c = int(round(2 * best_axis - c))
            if 0 <= mirror_c < cols:
                new_grid[r, mirror_c] = True

    return new_grid


# ---------------------------------------------------------------------------
# Font-unit helpers
# ---------------------------------------------------------------------------

def scale_contours(contours: list, scale: float, dx: float, dy: float) -> list:
    """Convert pixel-unit contours to font units."""
    return [
        [(int(round(x * scale + dx)), int(round(y * scale + dy))) for x, y in c]
        for c in contours
    ]


def draw_contours(contours: list, pen) -> None:
    """Emit contours via a SegmentPen (moveTo / lineTo / closePath)."""
    for c in contours:
        if len(c) < 3:
            continue
        pen.moveTo(c[0])
        for pt in c[1:]:
            pen.lineTo(pt)
        pen.closePath()


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(input_path: str, grid_size: int, output_path: str, threshold: int = 96) -> None:
    from fontTools.ttLib import TTFont
    from fontTools.pens.ttGlyphPen import TTGlyphPen

    print(f"Loading  {input_path}")
    font = TTFont(input_path)

    cmap = font.getBestCmap()
    if not cmap:
        sys.exit("Font has no cmap — cannot map codepoints to glyphs.")

    upm   = font["head"].unitsPerEm    # 1024
    scale = upm / grid_size            # font-units per pixel  (e.g. 64 at grid=16)

    glyf  = font["glyf"]
    hmtx  = font["hmtx"].metrics

    face  = load_face(input_path, grid_size)

    # ------------------------------------------------------------------
    # Pre-compute alignment shifts from the original vector outlines.
    # For LR-symmetric glyphs, snap the detected symmetry axis to the
    # nearest 0.5-pixel boundary.  For all other glyphs, snap the bbox
    # centre so that the rendered pixels are centred in the advance.
    # ------------------------------------------------------------------
    glyph_set = font.getGlyphSet()
    shifts = {}   # glyph_name → delta_26dot6

    for glyph_name in sorted(set(cmap.values())):
        lr, tb, cx, cy = detect_symmetry(font, glyph_name)

        # Get bounding box from the glyf table (xMin/xMax are stored there)
        try:
            g = font["glyf"][glyph_name]
            g.recalcBounds(font["glyf"])
            xMin, yMin, xMax, yMax = g.xMin, g.yMin, g.xMax, g.yMax
        except Exception:
            continue

        # For non-LR-symmetric glyphs use the bbox midpoint as the centering
        # target (same maths as alignment_shift_26dot6, just a different cx).
        if not lr:
            cx = (xMin + xMax) / 2

        dx26 = alignment_shift_26dot6(cx, xMin, scale, threshold)

        if dx26 != 0:
            shifts[glyph_name] = dx26
            delta_px = dx26 / 64
            print(f"  align  {glyph_name:30s}  lr={lr}  "
                  f"cx={cx:.1f}fu={cx/scale:.3f}px  shift={delta_px:+.3f}px")

    print()

    # ------------------------------------------------------------------
    # Rasterise, trace, write glyphs
    # ------------------------------------------------------------------
    total = len(cmap)
    done  = 0
    skipped = []

    for cp, glyph_name in sorted(cmap.items()):
        done += 1
        delta = shifts.get(glyph_name, 0)

        try:
            grid, adv_px, bm_left, bm_top = rasterise(face, cp, threshold, delta)
        except Exception as e:
            print(f"  [{done}/{total}] {glyph_name} U+{cp:04X}  SKIP (rasterise error: {e})")
            skipped.append(glyph_name)
            continue

        if grid.size == 0:
            print(f"  [{done}/{total}] {glyph_name} U+{cp:04X}  blank, keeping original")
            continue

        grid = enforce_lr_symmetry(grid)

        rows = grid.shape[0]
        contours = bitmap_to_contours(grid)

        if not contours:
            print(f"  [{done}/{total}] {glyph_name} U+{cp:04X}  no contours, keeping original")
            skipped.append(glyph_name)
            continue

        dx = bm_left * scale
        dy = (bm_top - rows) * scale

        scaled = scale_contours(contours, scale, dx, dy)

        orig_adv, _orig_lsb = hmtx.get(glyph_name, (upm, 0))
        snapped_adv = int(round(orig_adv / scale) * scale)

        pen = TTGlyphPen(None)
        draw_contours(scaled, pen)

        try:
            glyf[glyph_name] = pen.glyph()
        except Exception as e:
            print(f"  [{done}/{total}] {glyph_name} U+{cp:04X}  SKIP (glyph build error: {e})")
            skipped.append(glyph_name)
            continue

        hmtx[glyph_name] = (snapped_adv, int(round(bm_left * scale)))

        n_pts = sum(len(c) for c in scaled)
        shift_tag = f"  Δx={delta/64:+.3f}px" if delta else ""
        print(f"  [{done}/{total}] {glyph_name} U+{cp:04X}  "
              f"{grid.shape[1]}×{grid.shape[0]}px  "
              f"{len(scaled)} contour(s)  {n_pts} pts{shift_tag}")

    font.save(output_path)
    print(f"\nSaved → {output_path}")
    if skipped:
        print(f"Skipped {len(skipped)} glyph(s): {', '.join(skipped)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    inp = Path(args.input).resolve()
    if not inp.exists():
        sys.exit(f"File not found: {inp}")

    if args.output:
        out = Path(args.output)
        if out.is_dir():
            out = out / f"{inp.stem}_{args.grid}.ttf"
    else:
        out = inp.parent / f"{inp.stem}_{args.grid}.ttf"

    print(f"Input     : {inp}")
    print(f"Grid      : {args.grid}×{args.grid}  (1 pixel = {1024 / args.grid:.1f} font units)")
    print(f"Threshold : {args.threshold}/255")
    print(f"Output    : {out}")
    print()

    convert(str(inp), args.grid, str(out), args.threshold)


if __name__ == "__main__":
    main()
