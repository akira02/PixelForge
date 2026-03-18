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

Supports both TrueType (.ttf) and OpenType/CFF (.otf) input fonts.

Dependencies:
    pip install fonttools freetype-py numpy

Usage:
    python pixel_forge.py fonts/original.ttf
    python pixel_forge.py fonts/original.otf
    python pixel_forge.py --grid 16 fonts/original.ttf
    python pixel_forge.py --grid 20 --output fonts/ fonts/original.otf

Output filename: <input_stem>_<grid>.ttf/.otf  (in same dir as input by default)
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
    ap.add_argument("input", help="Input TTF or OTF file (e.g. fonts/original.ttf or fonts/original.otf)")
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
    ap.add_argument(
        "--debug-glyph", "-d", default=None, metavar="NAME",
        help="Print on-curve points and mirror-hit details for the named glyph, then exit.",
    )
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Glyph bounding-box helper (works for both TTF and OTF/CFF)
# ---------------------------------------------------------------------------

def _get_glyph_bounds(font, glyph_name: str):
    """
    Return (xMin, yMin, xMax, yMax) in font units, or None if unavailable.
    Uses BoundsPen so it works for both TrueType (glyf) and CFF outlines.
    """
    from fontTools.pens.boundsPen import BoundsPen
    glyph_set = font.getGlyphSet()
    if glyph_name not in glyph_set:
        return None
    bp = BoundsPen(glyph_set)
    try:
        glyph_set[glyph_name].draw(bp)
    except Exception:
        return None
    return bp.bounds  # (xMin, yMin, xMax, yMax) or None for empty glyph


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


def detect_symmetry(font, glyph_name: str,
                    lr_x_tolerance: int = 32, lr_y_tolerance: int = 32,
                    tb_x_tolerance: int = 32, tb_y_tolerance: int = 32,
                    score_threshold: float = 0.75):
    """
    Detect left-right and top-bottom symmetry from on-curve outline points.

    For each on-curve point P, checks whether its mirror across the bounding-
    box centre exists in the outline within the given per-axis tolerances.

    Bezier curve outlines place on-curve endpoints at positions that may differ
    by up to ~20 font units even on geometrically symmetric paths:
      - LR pairs share the same y-height (y_tol small) but x can drift (x_tol larger)
      - TB pairs share the same x-column (x_tol small) but y can drift (y_tol larger)

    Returns
    -------
    lr : bool       — glyph is left-right symmetric
    tb : bool       — glyph is top-bottom symmetric
    cx : float      — x centre of the LR symmetry axis (font units)
    cy : float      — y centre of the TB symmetry axis (font units)
    lr_score : float
    tb_score : float
    """
    pts = _on_curve_points(font, glyph_name)
    if not pts:
        return False, False, 0.0, 0.0, 0.0, 0.0

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2

    # Build integer-rounded point set for fast lookup
    pt_set = {(round(x), round(y)) for x, y in pts}

    def has_lr_mirror(mx, my):
        return any(
            (mx + dx, my + dy) in pt_set
            for dx in range(-lr_x_tolerance, lr_x_tolerance + 1)
            for dy in range(-lr_y_tolerance, lr_y_tolerance + 1)
        )

    def has_tb_mirror(mx, my):
        return any(
            (mx + dx, my + dy) in pt_set
            for dx in range(-tb_x_tolerance, tb_x_tolerance + 1)
            for dy in range(-tb_y_tolerance, tb_y_tolerance + 1)
        )

    lr_hits = sum(1 for x, y in pts if has_lr_mirror(round(2 * cx - x), round(y)))
    tb_hits = sum(1 for x, y in pts if has_tb_mirror(round(x), round(2 * cy - y)))
    n = len(pts)
    lr_score = lr_hits / n
    tb_score = tb_hits / n

    return (lr_score >= score_threshold,
            tb_score >= score_threshold,
            cx, cy,
            lr_score, tb_score)


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


def rasterise(face, charcode: int, threshold: int = 96, delta_26dot6: int = 0, dy_26dot6: int = 0):
    """
    Render one codepoint at the face's current pixel size using grayscale
    anti-aliasing + threshold.

    If delta_26dot6 or dy_26dot6 != 0, a subpixel shift is applied before
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

    if delta_26dot6 != 0 or dy_26dot6 != 0:
        _set_ft_transform(face, delta_26dot6, dy_26dot6)

    face.load_char(charcode, freetype.FT_LOAD_DEFAULT | freetype.FT_LOAD_NO_HINTING)
    face.glyph.render(freetype.FT_RENDER_MODE_NORMAL)

    if delta_26dot6 != 0 or dy_26dot6 != 0:
        _set_ft_transform(face, 0, 0)   # reset

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

def enforce_lr_symmetry(grid: np.ndarray, axis: float = None, score_threshold: float = 0.75):
    """
    OR-fill the bitmap to enforce left-right symmetry.

    If `axis` (pixel-space column, may be fractional) is provided it is used
    directly as the mirror axis.  Otherwise the axis is estimated from the
    bitmap itself (less reliable).  In the fallback case, if fewer than
    `score_threshold` of content rows agree on the detected axis the grid is
    returned unchanged.

    Returns the (possibly modified) grid.
    """
    from collections import Counter

    rows, cols = grid.shape

    if axis is None:
        # Fallback: estimate axis from bitmap row centres
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
        axis = best_axis

    # Verify the bitmap is actually close to symmetric before enforcing.
    # For each non-empty row, compute the fraction of filled pixels that already
    # have their mirror filled.  If too many rows are significantly asymmetric
    # the glyph has intentional asymmetry (e.g. a chat-bubble tail) and should
    # not be enforced.
    good_rows = 0
    total_rows = 0
    for r in range(rows):
        filled = np.where(grid[r])[0]
        if len(filled) == 0:
            continue
        total_rows += 1
        hits = sum(
            1 for c in filled
            if 0 <= int(round(2 * axis - c)) < cols
            and grid[r, int(round(2 * axis - c))]
        )
        if hits / len(filled) >= 0.85:
            good_rows += 1
    if total_rows > 0 and good_rows / total_rows < 0.90:
        return grid

    new_grid = grid.copy()
    for r in range(rows):
        for c in range(cols):
            if not grid[r, c]:
                continue
            mirror_c = int(round(2 * axis - c))
            if 0 <= mirror_c < cols:
                new_grid[r, mirror_c] = True

    return new_grid


def enforce_tb_symmetry(grid: np.ndarray, axis: float = None, score_threshold: float = 0.75):
    """
    OR-fill the bitmap to enforce top-bottom symmetry.

    If `axis` (pixel-space row, may be fractional) is provided it is used
    directly as the mirror axis.  Otherwise the axis is estimated from the
    bitmap itself (less reliable).  In the fallback case, if fewer than
    `score_threshold` of content columns agree on the detected axis the grid is
    returned unchanged.

    Returns the (possibly modified) grid.
    """
    from collections import Counter

    rows, cols = grid.shape

    if axis is None:
        # Fallback: estimate axis from bitmap column centres
        centers = []
        for c in range(cols):
            filled = np.where(grid[:, c])[0]
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
        axis = best_axis

    # Verify the bitmap is actually close to symmetric before enforcing.
    good_cols = 0
    total_cols = 0
    for c in range(cols):
        filled = np.where(grid[:, c])[0]
        if len(filled) == 0:
            continue
        total_cols += 1
        hits = sum(
            1 for r in filled
            if 0 <= int(round(2 * axis - r)) < rows
            and grid[int(round(2 * axis - r)), c]
        )
        if hits / len(filled) >= 0.85:
            good_cols += 1
    if total_cols > 0 and good_cols / total_cols < 0.90:
        return grid

    new_grid = grid.copy()
    for r in range(rows):
        for c in range(cols):
            if not grid[r, c]:
                continue
            mirror_r = int(round(2 * axis - r))
            if 0 <= mirror_r < rows:
                new_grid[mirror_r, c] = True

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

    print(f"Loading  {input_path}")
    font = TTFont(input_path)

    cmap = font.getBestCmap()
    if not cmap:
        sys.exit("Font has no cmap — cannot map codepoints to glyphs.")

    upm   = font["head"].unitsPerEm    # 1024
    scale = upm / grid_size            # font-units per pixel  (e.g. 64 at grid=16)

    is_otf = "CFF " in font or "CFF2" in font
    if is_otf:
        from fontTools.pens.t2CharStringPen import T2CharStringPen
        cff_charstrings = font["CFF "].cff.topDictIndex[0].CharStrings
    else:
        from fontTools.pens.ttGlyphPen import TTGlyphPen
        glyf = font["glyf"]

    hmtx  = font["hmtx"].metrics

    face  = load_face(input_path, grid_size)

    # ------------------------------------------------------------------
    # Pre-compute alignment shifts from the original vector outlines.
    # For LR-symmetric glyphs, snap the detected symmetry axis to the
    # nearest 0.5-pixel boundary.  For all other glyphs, snap the bbox
    # centre so that the rendered pixels are centred in the advance.
    # ------------------------------------------------------------------
    glyph_set = font.getGlyphSet()
    shifts = {}        # glyph_name → (dx_26dot6, dy_26dot6)
    lr_symmetries = {} # glyph_name → bool
    tb_symmetries = {} # glyph_name → bool
    cx_axes = {}       # glyph_name → cx in font units (LR symmetry axis)
    cy_axes = {}       # glyph_name → cy in font units (TB symmetry axis)

    for glyph_name in sorted(set(cmap.values())):
        lr, tb, cx, cy, lr_score, tb_score = detect_symmetry(font, glyph_name)
        lr_symmetries[glyph_name] = lr
        tb_symmetries[glyph_name] = tb

        # Get bounding box (works for both TTF glyf and OTF/CFF outlines)
        bounds = _get_glyph_bounds(font, glyph_name)
        if bounds is None:
            continue
        xMin, yMin, xMax, yMax = bounds

        # For non-LR-symmetric glyphs use the bbox midpoint as the centering
        # target (same maths as alignment_shift_26dot6, just a different cx).
        if not lr:
            cx = (xMin + xMax) / 2
        if not tb:
            cy = (yMin + yMax) / 2

        cx_axes[glyph_name] = cx
        cy_axes[glyph_name] = cy

        dx26 = alignment_shift_26dot6(cx, xMin, scale, threshold)
        dy26 = alignment_shift_26dot6(cy, yMin, scale, threshold)

        if dx26 != 0 or dy26 != 0:
            shifts[glyph_name] = (dx26, dy26)

    # ------------------------------------------------------------------
    # Rasterise, trace, write glyphs
    # ------------------------------------------------------------------
    total = len(cmap)
    done  = 0
    skipped = []

    for cp, glyph_name in sorted(cmap.items()):
        done += 1
        dx26, dy26 = shifts.get(glyph_name, (0, 0))

        try:
            grid, adv_px, bm_left, bm_top = rasterise(face, cp, threshold, dx26, dy26)
        except Exception as e:
            print(f"  [{done}/{total}] {glyph_name} U+{cp:04X}  SKIP (rasterise error: {e})")
            skipped.append(glyph_name)
            continue

        if grid.size == 0:
            print(f"  [{done}/{total}] {glyph_name} U+{cp:04X}  blank, keeping original")
            continue

        if lr_symmetries.get(glyph_name, False):
            grid = enforce_lr_symmetry(grid)

        if tb_symmetries.get(glyph_name, False):
            grid = enforce_tb_symmetry(grid)

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

        if is_otf:
            try:
                # Reuse the existing charstring's private dict (handles both
                # simple and CID-keyed fonts where Private is per-FD).
                existing_private = getattr(cff_charstrings.get(glyph_name), 'private', None)
                pen = T2CharStringPen(snapped_adv, None)
                draw_contours(scaled, pen)
                cff_charstrings[glyph_name] = pen.getCharString(private=existing_private)
            except Exception as e:
                print(f"  [{done}/{total}] {glyph_name} U+{cp:04X}  SKIP (glyph build error: {e})")
                skipped.append(glyph_name)
                continue
        else:
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
        sym_tag = ("LR" if lr_symmetries.get(glyph_name) else "") + \
                  ("TB" if tb_symmetries.get(glyph_name) else "")
        sym_tag = f"  [{sym_tag}]" if sym_tag else ""
        shift_tag = f"  Δ=({dx26/64:+.3f},{dy26/64:+.3f})px" if (dx26 or dy26) else ""
        print(f"  [{done}/{total}] {glyph_name} U+{cp:04X}  "
              f"{grid.shape[1]}×{grid.shape[0]}px  "
              f"{len(scaled)} contour(s)  {n_pts} pts{sym_tag}{shift_tag}")

    font.save(output_path)
    print(f"\nSaved → {output_path}")
    if skipped:
        print(f"Skipped {len(skipped)} glyph(s): {', '.join(skipped)}")


# ---------------------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------------------

def _debug_glyph(font_path: str, glyph_name: str,
                 lr_x_tolerance: int = 32, lr_y_tolerance: int = 32,
                 tb_x_tolerance: int = 32, tb_y_tolerance: int = 32) -> None:
    """
    Print on-curve points, symmetry axis, and per-point mirror results for
    a named glyph so that detection failures can be diagnosed.
    """
    from fontTools.ttLib import TTFont

    font = TTFont(font_path)
    cmap_rev = {v: k for k, v in (font.getBestCmap() or {}).items()}
    cp = cmap_rev.get(glyph_name)
    print(f"Glyph     : {glyph_name}  (U+{cp:04X})" if cp else f"Glyph     : {glyph_name}  (not in cmap)")
    print(f"Tolerances: LR x±{lr_x_tolerance} y±{lr_y_tolerance}   TB x±{tb_x_tolerance} y±{tb_y_tolerance}")

    pts = _on_curve_points(font, glyph_name)
    if not pts:
        print("No on-curve points found.")
        return

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    pt_set = {(round(x), round(y)) for x, y in pts}

    def has_lr_mirror(mx, my):
        return any(
            (mx + dx, my + dy) in pt_set
            for dx in range(-lr_x_tolerance, lr_x_tolerance + 1)
            for dy in range(-lr_y_tolerance, lr_y_tolerance + 1)
        )

    def has_tb_mirror(mx, my):
        return any(
            (mx + dx, my + dy) in pt_set
            for dx in range(-tb_x_tolerance, tb_x_tolerance + 1)
            for dy in range(-tb_y_tolerance, tb_y_tolerance + 1)
        )

    print(f"Points    : {len(pts)}")
    print(f"x range   : {min(xs):.0f} – {max(xs):.0f}   cx = {cx:.1f}")
    print(f"y range   : {min(ys):.0f} – {max(ys):.0f}   cy = {cy:.1f}")
    print()
    print(f"{'#':>3}  {'x':>6}  {'y':>6}  {'LR mirror':>10}  LR✓  {'TB mirror':>10}  TB✓")
    print("-" * 60)

    lr_hits = 0
    tb_hits = 0
    for i, (x, y) in enumerate(pts):
        lmx, lmy = round(2 * cx - x), round(y)
        tmx, tmy = round(x), round(2 * cy - y)
        lr_ok = has_lr_mirror(lmx, lmy)
        tb_ok = has_tb_mirror(tmx, tmy)
        lr_hits += lr_ok
        tb_hits += tb_ok
        print(f"{i:>3}  {x:>6.0f}  {y:>6.0f}  "
              f"({lmx:>4},{lmy:>4})  {'✓' if lr_ok else '✗':>4}  "
              f"({tmx:>4},{tmy:>4})  {'✓' if tb_ok else '✗':>4}")

    n = len(pts)
    print()
    print(f"LR score  : {lr_hits}/{n} = {lr_hits/n:.3f}")
    print(f"TB score  : {tb_hits}/{n} = {tb_hits/n:.3f}")

    bounds = _get_glyph_bounds(font, glyph_name)
    if bounds is not None:
        bxMin, byMin, bxMax, byMax = bounds
        print(f"bbox      : xMin={bxMin:.0f} xMax={bxMax:.0f} yMin={byMin:.0f} yMax={byMax:.0f}")
        print(f"bbox cx   : {(bxMin+bxMax)/2:.1f}  (vs on-curve cx={cx:.1f})")

    # ---- Bitmap visualisation -----------------------------------------------
    cmap_fwd = font.getBestCmap() or {}
    cp_code = cmap_rev.get(glyph_name)
    if cp_code is None:
        return

    face = load_face(font_path, 16)   # use grid=16 for debug
    scale = font["head"].unitsPerEm / 16

    lr, tb, cx2, cy2, _, _ = detect_symmetry(font, glyph_name)

    bounds2 = _get_glyph_bounds(font, glyph_name)
    if bounds2 is not None:
        xMin2, yMin2, xMax2, yMax2 = bounds2
    else:
        xMin2 = xMax2 = yMin2 = yMax2 = 0

    if not lr:
        cx2 = (xMin2 + xMax2) / 2
    if not tb:
        cy2 = (yMin2 + yMax2) / 2

    dx26 = alignment_shift_26dot6(cx2, xMin2, scale)
    dy26 = alignment_shift_26dot6(cy2, yMin2, scale)

    raw_grid, _, bm_left, bm_top = rasterise(face, cp_code, 96, dx26, dy26)
    if raw_grid.size == 0:
        print("\n(blank bitmap)")
        return

    lr_axis = None
    tb_axis = None

    enforced = raw_grid.copy()
    if lr:
        enforced = enforce_lr_symmetry(enforced)
    if tb:
        enforced = enforce_tb_symmetry(enforced)

    def _render(grid, axis_col=None, axis_row=None):
        rows2, cols2 = grid.shape
        lines = []
        for r in range(rows2):
            row_str = ""
            for c in range(cols2):
                row_str += "█" if grid[r, c] else "·"
            lines.append(row_str)
        # mark axis position in header
        header = " " * int(axis_col) + "|" if axis_col is not None else ""
        return header, lines

    lr_axis_str = f"{lr_axis:.2f}" if lr_axis is not None else "n/a"
    tb_axis_str = f"{tb_axis:.2f}" if tb_axis is not None else "n/a"
    print(f"\nBitmap  {raw_grid.shape[1]}×{raw_grid.shape[0]}px"
          f"  bm_left={bm_left}  bm_top={bm_top}"
          f"  lr_axis={lr_axis_str}  tb_axis={tb_axis_str}")

    rows2, cols2 = raw_grid.shape
    ax_col_i = int(round(lr_axis)) if lr_axis is not None else None
    # print side-by-side: before | after
    header_nums = "".join(str(c % 10) for c in range(cols2))
    print(f"  col:  {header_nums}    col:  {header_nums}")
    print(f"  {'before':^{cols2}}    {'after (enforced)':^{cols2}}")
    for r in range(rows2):
        b_row = "".join("█" if raw_grid[r, c]  else "·" for c in range(cols2))
        a_row = "".join("█" if enforced[r, c] else "·" for c in range(cols2))
        diff = "←" if (enforced[r] != raw_grid[r]).any() else " "
        print(f"  {b_row}    {a_row} {diff}")


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
            out = out / f"{inp.stem}_{args.grid}{inp.suffix}"
    else:
        out = inp.parent / f"{inp.stem}_{args.grid}{inp.suffix}"

    print(f"Input     : {inp}")
    print(f"Grid      : {args.grid}×{args.grid}  (1 pixel = {1024 / args.grid:.1f} font units)")
    print(f"Threshold : {args.threshold}/255")
    print(f"Output    : {out}")
    print()

    if args.debug_glyph:
        _debug_glyph(str(inp), args.debug_glyph)
        return

    convert(str(inp), args.grid, str(out), args.threshold)


if __name__ == "__main__":
    main()
