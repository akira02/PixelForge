# pixel_forge.py

A Python script that converts a smooth TrueType icon font into a pixel/bitmap style font.
Each glyph is rasterised onto a fixed pixel grid and re-vectorised as a rectilinear (all right-angle, no curves) outline.

將平滑曲線的 TrueType 圖示字型自動轉換為像素點陣風格。
每個字符會在固定格數的像素格上光柵化，再重新向量化為純直角輪廓（無曲線）。

---

## Requirements 環境需求

Python 3.8+ and the following packages:

```
pip install fonttools freetype-py numpy
```

---

## Usage 使用方式

```
python pixel_forge.py <input.ttf> [options]
```

### Options 參數

| Option          | Short     | Default           | Description                                                                                                                                              |
| --------------- | --------- | ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--grid N`      | `-g N`    | `16`              | Pixel grid height in pixels. The output filename will include this value (e.g. `PlurkIconFont_16.ttf`). 像素格高度，輸出檔名會帶入此數值。               |
| `--output PATH` | `-o PATH` | same dir as input | Output path. Can be a directory (auto-named) or a full file path. 輸出路徑，可指定目錄（自動命名）或完整檔名。                                           |
| `--threshold N` | `-t N`    | `96`              | Grayscale fill threshold (1–255). Lower values produce fatter strokes; higher values produce thinner strokes. 灰階填充閾值，數值越低筆畫越粗，越高越細。 |

### Examples 範例

```bash
# Basic conversion at 16px grid (default)
# 以預設 16 格轉換
python pixel_forge.py fonts/original.ttf

# 24px grid, thinner strokes
# 24 格，筆畫較細
python pixel_forge.py fonts/original.ttf --grid 24 --threshold 128

# Output to a specific directory
# 輸出至指定目錄
python pixel_forge.py fonts/original.ttf --grid 16 --output fonts/
```

The original font is never overwritten. Output is always written to a new file.

原始字型檔不會被覆蓋，轉換結果一律輸出為新檔案。

---

## How It Works 運作原理

1. **Rasterise** — Each glyph is rendered at the target grid size using FreeType's grayscale anti-aliasing, then thresholded to a binary pixel grid.
2. **Align** — Before rasterising, a subpixel translation is applied to snap the glyph's bounding-box centre to the nearest 0.5-pixel boundary. This ensures symmetric glyphs render symmetrically.
3. **Symmetry enforcement** — After rasterising, if at least 75% of the content rows are already symmetric around a common axis, the remaining rows are OR-filled to enforce full symmetry.
4. **Vectorise** — The binary pixel grid is traced into directed half-edge contours following TrueType winding convention (CCW outer, CW holes), then collinear points are removed.
5. **Write** — The rectilinear outlines are written back into a new TTF, with advance widths snapped to the nearest whole pixel.

---

1. **光柵化** — 使用 FreeType 灰階抗鋸齒將每個字符渲染到目標格數，再以閾值轉為二值像素格。
2. **對齊** — 光柵化前，對每個字符施加次像素平移，將字符邊界框中心對齊到最近的 0.5 像素邊界，確保對稱圖示正確呈現對稱效果。
3. **對稱強制** — 光柵化後，若超過 75% 的像素行已對稱，則對剩餘行進行 OR 填充以補全對稱。
4. **向量化** — 以有向半邊算法追蹤二值像素格的輪廓，遵循 TrueType 捲繞方向（外輪廓逆時針、孔洞順時針），並移除共線端點。
5. **輸出** — 將純直角輪廓寫入新 TTF，字距寬度取整為最近的整數像素。

---

## Notes 注意事項

- The script processes all glyphs in the font's `cmap` automatically.
  腳本會自動處理字型 `cmap` 中的所有字符。
- Glyphs that produce no contours after rasterisation (e.g. whitespace) are left unchanged from the original.
  光柵化後沒有輪廓的字符（如空白）會保留原始資料不變。
- A larger `--grid` value produces a higher-fidelity pixel font but increases file size and may require adjusting your CSS `font-size`.
  `--grid` 數值越大，像素字型細節越豐富，但檔案也越大，可能需要調整 CSS 的 `font-size`。
