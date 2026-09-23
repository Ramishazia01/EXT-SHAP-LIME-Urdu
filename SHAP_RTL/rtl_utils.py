"""
shap_rtl.rtl_utils
──────────────────
RTL text utilities with proper language-specific font handling.
"""

import os
import warnings
import numpy as np
from PIL import Image

# ── font location ──────────────────────────────────────────────────────────────
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_FONTS_DIR = os.path.join(os.path.dirname(_PKG_DIR), "fonts")

# ============================================================================
# FONT CONFIGURATION - Language-specific fonts
# ============================================================================

_LANGUAGE_FONTS = {
    "Arabic": [
        "NotoSansArabic-Regular.ttf",
        "NotoSansArabic-VariableFont_wdth,wght.ttf",
        "Amiri-Regular.ttf",
        "Amiri-Bold.ttf",
        "ScheherazadeNew-Regular.ttf",
        "Scheherazade-Regular.ttf",
        "Lateef-Regular.ttf",
        "Harmattan-Regular.ttf",
        "DroidNaskh-Regular.ttf",
    ],
    "Urdu": [
        "JameelNooriNastaliq.ttf",
        "JameelNooriNastaleeq.ttf",
        "Jameel Noori Nastaleeq Regular.ttf",
        "NotoNastaliqUrdu-Regular.ttf",
        "NotoNastaliqUrdu-VariableFont_wght.ttf",
        "UrduNastaliq.ttf",
        "NafeesNastaleeq.ttf",
    ],
    "Persian": [
        "NotoSansArabic-Regular.ttf",
        "NotoNaskhArabic-Regular.ttf",
        "Amiri-Regular.ttf",
        "ScheherazadeNew-Regular.ttf",
        "Vazir.ttf",
        "Vazirmatn-Regular.ttf",
    ],
    "Hebrew": [
        "NotoSansHebrew-Regular.ttf",
        "NotoSansHebrew-VariableFont_wdth,wght.ttf",
        "NotoSerifHebrew-Regular.ttf",
        "NotoSerifHebrew-VariableFont_wdth,wght.ttf",
        "Arial.ttf",
        "TimesNewRoman.ttf",
        "EzraSIL-Regular.ttf",
        "SBLHebrew-Regular.ttf",
    ],
    "Pashto": [
        "NotoSansArabic-Regular.ttf",
        "NotoNaskhArabic-Regular.ttf",
        "Amiri-Regular.ttf",
        "ScheherazadeNew-Regular.ttf",
        "JameelNooriNastaliq.ttf",
    ],
    "Dari": [
        "NotoSansArabic-Regular.ttf",
        "Amiri-Regular.ttf",
        "ScheherazadeNew-Regular.ttf",
        "JameelNooriNastaliq.ttf",
    ],
    "Kurdish": [
        "NotoSansArabic-Regular.ttf",
        "Amiri-Regular.ttf",
        "ScheherazadeNew-Regular.ttf",
        "NotoSansHebrew-Regular.ttf",
    ],
}

# ============================================================================
# FONT KEYWORDS FOR DETECTION (PRIORITIZED)
# ============================================================================

_LANGUAGE_KEYWORDS = {
    "Hebrew": [
        "hebrew",
        "ezra",
        "sbl",
        "noto sans hebrew",
        "noto serif hebrew",
    ],
    "Arabic": [
        "arabic",
        "naskh",
        "amiri",
        "scheherazade",
        "lateef",
        "harmattan",
        "droidnaskh",
    ],
    "Urdu": [
        "nastaliq",
        "nastaleeq",
        "jameel",
        "nafees",
        "urdu",
        "notonastaliqurdu",
    ],
    "Persian": [
        "persian",
        "farsi",
        "vazir",
        "vazirmatn",
    ],
}

_EXCLUDED_FONTS = {
    "Hebrew": ["arabic", "nastaliq", "urdu", "persian", "naskh", "amiri", "scheherazade", "lateef", "harmattan"],
    "Arabic": ["hebrew", "nastaliq", "urdu", "persian"],
    "Urdu": ["hebrew", "arabic", "persian"],
    "Persian": ["hebrew", "urdu", "arabic"],
}

# ============================================================================
# GLOBAL STATE - Track current language
# ============================================================================

_CURRENT_LANGUAGE = None
_RTL_FONT_PATH = None
_FONT_CACHE = {}
_FONT_REGISTERED = False

def set_current_language(language: str):
    """Set the current language for font detection."""
    global _CURRENT_LANGUAGE
    _CURRENT_LANGUAGE = language
    print(f"[DEBUG] Current language set to: {language}")


def get_current_language() -> str:
    """Get the current language."""
    return _CURRENT_LANGUAGE


# ============================================================================
# FONT DOWNLOADER - AUTO DOWNLOAD IF MISSING
# ============================================================================

def _download_font(language: str) -> str | None:
    """
    Auto-download font if not available.
    """
    fonts_dir = _FONTS_DIR
    os.makedirs(fonts_dir, exist_ok=True)
    
    font_map = {
        "Hebrew": {
            "url": "https://github.com/google/fonts/raw/main/ofl/notosanshebrew/NotoSansHebrew%5Bwdth,wght%5D.ttf",
            "filename": "NotoSansHebrew-Regular.ttf"
        },
        "Arabic": {
            "url": "https://github.com/google/fonts/raw/main/ofl/notosansarabic/NotoSansArabic%5Bwdth,wght%5D.ttf",
            "filename": "NotoSansArabic-Regular.ttf"
        },
        "Urdu": {
            "url": "https://github.com/google/fonts/raw/main/ofl/notonastaliqurdu/NotoNastaliqUrdu%5Bwght%5D.ttf",
            "filename": "NotoNastaliqUrdu-Regular.ttf"
        },
        "Persian": {
            "url": "https://github.com/google/fonts/raw/main/ofl/notosansarabic/NotoSansArabic%5Bwdth,wght%5D.ttf",
            "filename": "NotoSansArabic-Regular.ttf"
        },
    }
    
    if language not in font_map:
        return None
    
    font_info = font_map[language]
    font_path = os.path.join(fonts_dir, font_info["filename"])
    
    if os.path.exists(font_path):
        return font_path
    
    try:
        import urllib.request
        print(f"[INFO] Downloading {language} font...")
        urllib.request.urlretrieve(font_info["url"], font_path)
        print(f"[INFO] Font downloaded: {font_path}")
        return font_path
    except Exception as e:
        print(f"[WARNING] Could not download font: {e}")
        return None


# ============================================================================
# FONT MANAGEMENT
# ============================================================================

def _find_rtl_font(language: str = None) -> tuple[str | None, str]:
    """
    Find the best font for the specified language.
    """
    global _RTL_FONT_PATH, _CURRENT_LANGUAGE
    
    if language is None:
        language = _CURRENT_LANGUAGE
        print(f"[DEBUG] Using current language: {language}")
    
    if language is None:
        print("[WARNING] No language specified!")
        language = "Arabic"
    
    cache_key = language
    if cache_key in _FONT_CACHE:
        cached_path, cached_name = _FONT_CACHE[cache_key]
        if language == "Hebrew" and cached_name and ("arabic" in cached_name.lower() or "nastaliq" in cached_name.lower()):
            print(f"[DEBUG] Clearing cache - wrong font cached for Hebrew")
            del _FONT_CACHE[cache_key]
        else:
            return _FONT_CACHE[cache_key]
    
    font_path = None
    font_name = None
    
    print(f"[DEBUG] Looking for font for language: {language}")
    
    # 1) Try language-specific fonts FIRST
    if language in _LANGUAGE_FONTS:
        fonts_to_try = _LANGUAGE_FONTS[language]
        print(f"[DEBUG] Trying language-specific fonts: {fonts_to_try}")
        
        if os.path.isdir(_FONTS_DIR):
            for fname in fonts_to_try:
                full = os.path.join(_FONTS_DIR, fname)
                if os.path.isfile(full):
                    font_path = full
                    font_name = fname
                    print(f"[INFO] Found bundled {language} font: {fname}")
                    break
    
    # 2) If not found, try to download
    if font_path is None:
        print(f"[INFO] {language} font not found in bundled fonts.")
        font_path = _download_font(language)
        if font_path:
            font_name = os.path.basename(font_path)
            print(f"[INFO] Downloaded {language} font: {font_name}")
    
    # 3) Try system fonts
    if font_path is None:
        try:
            import matplotlib.font_manager as fm
            keywords = _LANGUAGE_KEYWORDS.get(language, [])
            excluded = _EXCLUDED_FONTS.get(language, [])
            
            for f in fm.fontManager.ttflist:
                f_lower = f.name.lower()
                if any(excl in f_lower for excl in excluded):
                    continue
                if any(kw in f_lower for kw in keywords):
                    font_path = f.fname
                    font_name = f.name
                    print(f"[INFO] Found system font for {language}: {f.name}")
                    break
        except Exception:
            pass
    
    # 4) For Hebrew specifically
    if font_path is None and language == "Hebrew":
        try:
            import matplotlib.font_manager as fm
            for f in fm.fontManager.ttflist:
                if "hebrew" in f.name.lower():
                    font_path = f.fname
                    font_name = f.name
                    print(f"[INFO] Found Hebrew system font: {f.name}")
                    break
        except Exception:
            pass
    
    # 5) Try any font in fonts directory
    if font_path is None and os.path.isdir(_FONTS_DIR):
        excluded = _EXCLUDED_FONTS.get(language, [])
        for fname in os.listdir(_FONTS_DIR):
            if not fname.lower().endswith((".ttf", ".otf")):
                continue
            fname_lower = fname.lower()
            if any(excl in fname_lower for excl in excluded):
                continue
            font_path = os.path.join(_FONTS_DIR, fname)
            font_name = fname
            print(f"[INFO] Using fallback font: {fname}")
            break
    
    _FONT_CACHE[cache_key] = (font_path, font_name)
    if font_path:
        _RTL_FONT_PATH = font_path
        _register_font_with_matplotlib(font_path)
    
    return font_path, font_name


def _register_font_with_matplotlib(font_path: str) -> bool:
    try:
        import matplotlib as mpl
        from matplotlib import font_manager
        
        font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_path)
        font_name = prop.get_name()
        
        mpl.rcParams['font.family'] = font_name
        mpl.rcParams['axes.unicode_minus'] = False
        print(f"[INFO] Font registered with matplotlib: {font_name}")
        return True
    except Exception as e:
        print(f"[WARNING] Could not register font: {e}")
        return False


def set_rtl_font_path(font_path: str) -> bool:
    global _RTL_FONT_PATH, _FONT_REGISTERED
    
    if not os.path.isfile(font_path):
        print(f"[ERROR] Font not found: {font_path}")
        return False
    
    _RTL_FONT_PATH = font_path
    
    try:
        import matplotlib as mpl
        from matplotlib import font_manager
        
        font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_path)
        font_name = prop.get_name()
        
        mpl.rcParams['font.family'] = font_name
        mpl.rcParams['axes.unicode_minus'] = False
        mpl.rcParams['text.color'] = 'black'
        
        _FONT_REGISTERED = True
        print(f"[INFO] Font registered with matplotlib: {font_name}")
        return True
    except Exception as e:
        print(f"[WARNING] Could not register font: {e}")
        return False


def get_rtl_font_path(language: str = None) -> str | None:
    font_path, _ = _find_rtl_font(language)
    return font_path


# ============================================================================
# RTL TEXT DETECTION
# ============================================================================

HEBREW_RANGE = (0x0590, 0x05FF)
ARABIC_RANGES = [
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
]

def is_rtl_text(text: str) -> bool:
    if not isinstance(text, str):
        return False
    for ch in text:
        cp = ord(ch)
        if HEBREW_RANGE[0] <= cp <= HEBREW_RANGE[1]:
            return True
        for start, end in ARABIC_RANGES:
            if start <= cp <= end:
                return True
    return False


def detect_language(text: str) -> str:
    if not is_rtl_text(text):
        return "unknown"
    for ch in text:
        cp = ord(ch)
        if HEBREW_RANGE[0] <= cp <= HEBREW_RANGE[1]:
            return "Hebrew"
    return "Arabic"


def is_hebrew(text: str) -> bool:
    if not isinstance(text, str):
        return False
    for ch in text:
        if HEBREW_RANGE[0] <= ord(ch) <= HEBREW_RANGE[1]:
            return True
    return False


# ============================================================================
# RENDER FUNCTION
# ============================================================================

def render_rtl_label(
    text: str,
    font_path: str,
    font_size_pt: float = 13,
    dpi: int = 150,
    color: str = "black"
) -> Image.Image:
    try:
        import uharfbuzz as hb
        import freetype as ft
        
        px_per_pt = dpi / 72.0
        px_size = font_size_pt * px_per_pt
        ft_size = int(px_size * 64)
        
        blob = hb.Blob.from_file_path(font_path)
        face_hb = hb.Face(blob)
        hb_font = hb.Font(face_hb)
        hb_font.scale = (int(px_size * 64), int(px_size * 64))
        
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(hb_font, buf)
        
        infos = buf.glyph_infos
        positions = buf.glyph_positions
        
        face_ft = ft.Face(font_path)
        face_ft.set_char_size(ft_size)
        
        total_adv = sum(abs(p.x_advance) for p in positions) // 64 + 10
        h = int(px_size * 3.2)
        w = max(int(total_adv) + 10, 10)
        
        canvas = np.zeros((h, w, 4), dtype=np.uint8)
        baseline = int(h * 0.6)
        x = 4
        
        import matplotlib.colors as mcolors
        try:
            rgb = mcolors.to_rgb(color)
            text_rgb = (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
        except:
            text_rgb = (0, 0, 0)
        
        for info, pos in zip(infos, positions):
            gid = info.codepoint
            try:
                face_ft.load_glyph(gid, ft.FT_LOAD_RENDER)
            except Exception:
                x += abs(pos.x_advance) // 64
                continue
            
            bm = face_ft.glyph.bitmap
            if bm.width > 0 and bm.rows > 0:
                arr = np.frombuffer(bytes(bm.buffer), dtype=np.uint8).reshape(bm.rows, bm.width)
                bx = x + pos.x_offset // 64 + face_ft.glyph.bitmap_left
                by = baseline - face_ft.glyph.bitmap_top - pos.y_offset // 64
                x1, y1 = max(bx, 0), max(by, 0)
                x2, y2 = min(bx + bm.width, w), min(by + bm.rows, h)
                ax1 = x1 - bx
                ay1 = y1 - by
                ax2 = ax1 + (x2 - x1)
                ay2 = ay1 + (y2 - y1)
                if ax2 > ax1 and ay2 > ay1:
                    for i in range(ay1, ay2):
                        for j in range(ax1, ax2):
                            if arr[i, j] > 0:
                                alpha_val = arr[i, j]
                                canvas[y1 + (i - ay1), x1 + (j - ax1), 0] = text_rgb[0]
                                canvas[y1 + (i - ay1), x1 + (j - ax1), 1] = text_rgb[1]
                                canvas[y1 + (i - ay1), x1 + (j - ax1), 2] = text_rgb[2]
                                canvas[y1 + (i - ay1), x1 + (j - ax1), 3] = alpha_val
            
            x += abs(pos.x_advance) // 64
        
        alpha = canvas[:, :, 3]
        rows = np.any(alpha > 0, axis=1)
        cols = np.any(alpha > 0, axis=0)
        if rows.any() and cols.any():
            r0, r1 = np.where(rows)[0][[0, -1]]
            c0, c1 = np.where(cols)[0][[0, -1]]
            pad = 3
            canvas = canvas[max(r0 - pad, 0):r1 + pad + 1,
                           max(c0 - pad, 0):c1 + pad + 1]
        
        return Image.fromarray(canvas, "RGBA")
        
    except (ImportError, Exception):
        try:
            from PIL import ImageDraw, ImageFont
            img = Image.new("RGBA", (200, 50), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype(font_path, int(font_size_pt))
                draw.text((5, 5), text, font=font, fill=(0, 0, 0, 255))
                bbox = img.getbbox()
                if bbox:
                    img = img.crop(bbox)
                return img
            except Exception:
                pass
        except Exception:
            pass
        
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))


# ============================================================================
# RENDER MIXED LABEL - ADD THIS FUNCTION
# ============================================================================

def render_mixed_label(
    text: str,
    font_path: str,
    font_size_pt: float = 12,
    dpi: int = 150,
) -> Image.Image:
    """
    Render a label that contains both LTR and RTL parts.
    Used for waterfall plots where feature values and names are combined.
    """
    try:
        if " = " in text:
            ltr_part, rtl_part = text.split(" = ", 1)
            if is_rtl_text(rtl_part):
                # Render LTR part
                ltr_img = _render_latin_with_freetype(ltr_part + " = ", font_size_pt, dpi)
                # Render RTL part
                rtl_img = render_rtl_label(rtl_part, font_path, font_size_pt, dpi)
                
                if ltr_img.width < 2 and rtl_img.width < 2:
                    return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
                
                h = max(ltr_img.height, rtl_img.height, 1)
                w = ltr_img.width + rtl_img.width
                combined = Image.new("RGBA", (max(w, 1), h), (0, 0, 0, 0))
                combined.paste(ltr_img, (0, (h - ltr_img.height) // 2))
                if rtl_img.width >= 2:
                    combined.paste(
                        rtl_img,
                        (ltr_img.width, (h - rtl_img.height) // 2),
                        rtl_img,
                    )
                return combined
        
        return render_rtl_label(text, font_path, font_size_pt, dpi)
        
    except Exception:
        return render_rtl_label(text, font_path, font_size_pt, dpi)


def _render_latin_with_freetype(
    text: str,
    font_size_pt: float = 12,
    dpi: int = 150,
) -> Image.Image:
    """Render plain LTR/ASCII text using FreeType with a Latin system font."""
    try:
        import freetype as ft
        import matplotlib.font_manager as fm
        
        # Find a Latin font
        latin_font = None
        for f in fm.fontManager.ttflist[:20]:
            if "arial" in f.name.lower() or "dejavu" in f.name.lower() or "liberation" in f.name.lower():
                latin_font = f.fname
                break
        
        if latin_font is None:
            return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        
        px_per_pt = dpi / 72.0
        px_size = font_size_pt * px_per_pt
        ft_size = int(px_size * 64)
        
        face = ft.Face(latin_font)
        face.set_char_size(ft_size)
        
        total_adv = 0
        for ch in text:
            try:
                face.load_char(ch, ft.FT_LOAD_DEFAULT)
                total_adv += face.glyph.advance.x >> 6
            except Exception:
                total_adv += int(px_size * 0.5)
        
        h = int(px_size * 2.2)
        w = max(int(total_adv) + 10, 10)
        canvas = np.zeros((h, w, 4), dtype=np.uint8)
        baseline = int(h * 0.68)
        x = 2
        
        for ch in text:
            try:
                face.load_char(ch, ft.FT_LOAD_RENDER)
                bm = face.glyph.bitmap
                if bm.width > 0 and bm.rows > 0:
                    arr = np.frombuffer(bytes(bm.buffer), dtype=np.uint8).reshape(bm.rows, bm.width)
                    bx = x + face.glyph.bitmap_left
                    by = baseline - face.glyph.bitmap_top
                    x1, y1 = max(bx, 0), max(by, 0)
                    x2, y2 = min(bx + bm.width, w), min(by + bm.rows, h)
                    ax1 = x1 - bx
                    ay1 = y1 - by
                    ax2 = ax1 + (x2 - x1)
                    ay2 = ay1 + (y2 - y1)
                    if ax2 > ax1 and ay2 > ay1:
                        canvas[y1:y2, x1:x2, 3] = np.maximum(
                            canvas[y1:y2, x1:x2, 3], arr[ay1:ay2, ax1:ax2]
                        )
                x += face.glyph.advance.x >> 6
            except Exception:
                x += int(px_size * 0.5)
        
        alpha = canvas[:, :, 3]
        rows = np.any(alpha > 0, axis=1)
        cols = np.any(alpha > 0, axis=0)
        if rows.any() and cols.any():
            r0, r1 = np.where(rows)[0][[0, -1]]
            c0, c1 = np.where(cols)[0][[0, -1]]
            pad = 2
            canvas = canvas[max(r0 - pad, 0):r1 + pad + 1,
                           max(c0 - pad, 0):c1 + pad + 1]
        
        return Image.fromarray(canvas, "RGBA")
        
    except Exception:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))


# ============================================================================
# RTL PLOT UTILITIES
# ============================================================================

def apply_rtl_ytick_images(
    ax,
    tick_labels,
    font_path,
    font_size_pt=13,
    dpi=150,
    ha="right",
    color="black",
    x_offset_pts=-6
):
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    
    fig = ax.figure
    fig.canvas.draw()
    
    yticks = ax.get_yticks()
    ylim = ax.get_ylim()
    ylim_lo = min(ylim) - 0.5
    ylim_hi = max(ylim) + 0.5
    
    for ytick_pos, label_text in zip(yticks, tick_labels):
        if not is_rtl_text(label_text):
            continue
        if not (ylim_lo <= ytick_pos <= ylim_hi):
            continue
        
        img_pil = render_rtl_label(
            label_text,
            font_path,
            font_size_pt=font_size_pt,
            dpi=dpi,
            color=color
        )
        if img_pil.size == (1, 1):
            continue
        
        arr = np.array(img_pil, dtype=np.uint8).copy()
        
        if color != "black" and color != "#000000":
            import matplotlib.colors as mcolors
            try:
                rgb = mcolors.to_rgb(color)
                for i in range(arr.shape[0]):
                    for j in range(arr.shape[1]):
                        if arr[i, j, 3] > 0:
                            arr[i, j, 0] = int(rgb[0] * 255)
                            arr[i, j, 1] = int(rgb[1] * 255)
                            arr[i, j, 2] = int(rgb[2] * 255)
            except Exception:
                pass
        
        oi = OffsetImage(arr, zoom=72.0 / dpi)
        oi.image.axes = ax
        
        ab = AnnotationBbox(
            oi,
            xy=(0, ytick_pos),
            xycoords=("axes fraction", "data"),
            xybox=(x_offset_pts, 0),
            boxcoords="offset points",
            frameon=False,
            box_alignment=(1.0, 0.5),
            pad=0,
        )
        ax.add_artist(ab)


def hide_rtl_text_ticks(ax, tick_labels):
    text_ticks = ax.get_yticklabels()
    for txt, label_text in zip(text_ticks, tick_labels):
        if is_rtl_text(label_text):
            txt.set_visible(False)


def _adjust_left_margin_for_rtl_labels(
    ax,
    tick_labels,
    font_path,
    font_size_pt=12,
    dpi=150,
    extra_pad_in=0.20
):
    rtl_labels = [lbl for lbl in tick_labels if is_rtl_text(lbl)]
    if not rtl_labels:
        return
    
    max_px = max(
        render_rtl_label(lbl, font_path, font_size_pt=font_size_pt, dpi=dpi).size[0]
        for lbl in rtl_labels
    )
    if max_px < 2:
        return
    
    fig = ax.figure
    fig_dpi = fig.dpi
    fig_w_in = fig.get_size_inches()[0]
    label_w_in = (max_px * 72.0 / dpi) / fig_dpi
    needed_left = (label_w_in + extra_pad_in) / fig_w_in
    
    if needed_left > fig.subplotpars.left:
        fig.subplots_adjust(left=min(needed_left, 0.55))


def add_rtl_title(
    text: str,
    ax=None,
    font_size_pt: float = 14,
    dpi: int = 150,
    pad: float = 15,
    color: str = "black"
):
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    
    if ax is None:
        ax = plt.gca()
    
    font_path, _ = _find_rtl_font()
    if font_path is None:
        ax.set_title(text, fontsize=font_size_pt, color=color)
        return
    
    img_pil = render_rtl_label(
        text,
        font_path,
        font_size_pt=font_size_pt,
        dpi=dpi,
        color=color
    )
    if img_pil.width < 2:
        ax.set_title(text, fontsize=font_size_pt, color=color)
        return
    
    arr = np.array(img_pil, dtype=np.uint8)
    oi = OffsetImage(arr, zoom=72.0 / dpi)
    oi.image.axes = ax
    
    ab = AnnotationBbox(
        oi,
        xy=(0.5, 1.0),
        xycoords="axes fraction",
        xybox=(0, pad),
        boxcoords="offset points",
        frameon=False,
        box_alignment=(0.5, 0.0),
        clip_on=False,
        pad=0,
    )
    ax.add_artist(ab)


# ============================================================================
# AUTO-INITIALIZATION
# ============================================================================

def _auto_initialize():
    """Auto-initialize RTL font support."""
    try:
        import matplotlib as mpl
        mpl.rcParams['text.color'] = 'black'
        mpl.rcParams['axes.labelcolor'] = 'black'
        mpl.rcParams['xtick.color'] = 'black'
        mpl.rcParams['ytick.color'] = 'black'
    except:
        pass
    
    print("[INFO] RTL font support ready. Font will be loaded based on language.")
    print("[INFO] Supported languages: Arabic, Urdu, Persian, Hebrew")


# Run auto-initialization
_auto_initialize()


# ============================================================================
# EXPORTED FUNCTIONS
# ============================================================================

__all__ = [
    '_find_rtl_font',
    'set_rtl_font_path',
    'get_rtl_font_path',
    'set_current_language',
    'get_current_language',
    'is_rtl_text',
    'detect_language',
    'is_hebrew',
    'render_rtl_label',
    'render_mixed_label',
    '_render_latin_with_freetype',
    'apply_rtl_ytick_images',
    'hide_rtl_text_ticks',
    '_adjust_left_margin_for_rtl_labels',
    'add_rtl_title',
    '_LANGUAGE_FONTS',
]