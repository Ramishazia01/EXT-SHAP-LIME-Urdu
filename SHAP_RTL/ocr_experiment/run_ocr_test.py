"""
===========================================================================
 OCR TEST FOR RTL SHAP PLOTS
===========================================================================

WHAT THIS DOES
--------------
It draws each of your feature words on a chart several different ways, then
uses text-reading software (OCR) to try to read them back. If the software
reads the word correctly, the chart drew it correctly. The score is called
CER (Character Error Rate):

    0.00 = read perfectly          1.00 = complete garbage

The output is a table of numbers that goes into Section 7 of your paper.


HOW TO USE IT - THREE COMMANDS
------------------------------
Open a terminal in the folder containing this file, then:

    python run_ocr_test.py --setup     (once - downloads fonts and OCR models)
    python run_ocr_test.py --demo      (once - proves everything works)
    python run_ocr_test.py             (the real run, on your words)

Before the real run, put your words in "words.csv" and edit the section
marked EDIT HERE further down.

If something is missing, the script tells you exactly what to install.
===========================================================================
"""

import argparse
import csv
import io
import os
import shutil
import subprocess
import sys
import unicodedata
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
FONT_DIR = HERE / "fonts"
TESS_DIR = HERE / "tessdata"

# Which font is used for each language. Urdu needs a Nastaliq font - that is
# the whole point of your paper, so do not substitute a plain Arabic font.
FONTS = {
    "urd": "NotoNastaliqUrdu.ttf",
    "ara": "NotoNaskhArabic.ttf",
    "fas": "NotoNaskhArabic.ttf",
    "heb": "NotoSansHebrew.ttf",
}

FONT_URLS = {
    "NotoNastaliqUrdu.ttf":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/notonastaliqurdu/NotoNastaliqUrdu%5Bwght%5D.ttf",
    "NotoNaskhArabic.ttf":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/notonaskharabic/NotoNaskhArabic%5Bwght%5D.ttf",
    "NotoSansHebrew.ttf":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/notosanshebrew/NotoSansHebrew%5Bwdth,wght%5D.ttf",
    "NotoSansArabic.ttf":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/notosansarabic/NotoSansArabic%5Bwdth,wght%5D.ttf",
}

TESS_URL = "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/{}.traineddata"


# =========================================================================
#  STEP 1 OF 3 - SETUP.  Run: python run_ocr_test.py --setup
# =========================================================================


def find_tesseract():
    """The Windows installer often does not add Tesseract to PATH, so look in
    the usual places before giving up."""
    if shutil.which("tesseract"):
        return shutil.which("tesseract")
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        str(Path.home() / "AppData/Local/Programs/Tesseract-OCR/tesseract.exe"),
        str(Path.home() / "AppData/Local/Tesseract-OCR/tesseract.exe"),
    ]
    for c in candidates:
        if Path(c).exists():
            try:
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = c
            except ImportError:
                pass
            return c
    return None


def setup():
    print("\nDownloading fonts and OCR models. This takes a few minutes.\n")
    FONT_DIR.mkdir(exist_ok=True)
    TESS_DIR.mkdir(exist_ok=True)

    for name, url in FONT_URLS.items():
        target = FONT_DIR / name
        if target.exists():
            print(f"  font   {name:24s} already here")
            continue
        print(f"  font   {name:24s} downloading...")
        urllib.request.urlretrieve(url, target)

    for lang in ("urd", "ara", "fas", "heb"):
        target = TESS_DIR / f"{lang}.traineddata"
        if target.exists():
            print(f"  OCR    {lang:24s} already here")
            continue
        print(f"  OCR    {lang:24s} downloading (this one is large)...")
        urllib.request.urlretrieve(TESS_URL.format(lang), target)

    print("\nSetup finished.\n")
    check_install()


def check_install():
    """Tells you in plain words what is missing and how to get it."""
    ok = True
    print("Checking your installation:\n")

    for module, pipname in [
        ("matplotlib", "matplotlib"), ("PIL", "pillow"),
        ("pytesseract", "pytesseract"), ("jiwer", "jiwer"),
        ("numpy", "numpy"), ("bidi", "python-bidi"),
        ("arabic_reshaper", "arabic-reshaper"),
        ("uharfbuzz", "uharfbuzz"), ("freetype", "freetype-py"),
    ]:
        try:
            __import__(module)
            print(f"  [ok]      {pipname}")
        except ImportError:
            ok = False
            print(f"  [MISSING] {pipname}   ->  pip install {pipname}")

    found = find_tesseract()
    if found:
        print(f"  [ok]      tesseract program  ({found})")
    else:
        ok = False
        print("  [MISSING] tesseract program")
        print("            Windows: download the installer from")
        print("              https://github.com/UB-Mannheim/tesseract/wiki")
        print("            Mac:     brew install tesseract")
        print("            Linux:   sudo apt install tesseract-ocr")

    # Raqm is the part that does the actual complex text shaping. Without it
    # the "correct" condition silently draws wrong text and every number in
    # your results is meaningless. This check matters more than the others.
    # Complex text layout. Either route works; you need one of them.
    if _hb_available():
        print("  [ok]      complex text layout  (uharfbuzz + freetype-py)")
    else:
        try:
            from PIL import features
            if features.check("raqm"):
                print(f"  [ok]      complex text layout  (Pillow raqm "
                      f"{features.version('raqm')})")
            else:
                raise RuntimeError
        except Exception:
            ok = False
            print("  [MISSING] complex text layout - THIS ONE IS IMPORTANT")
            print("            Without it your results will be wrong but look fine.")
            print("            Fix:  pip install uharfbuzz freetype-py")
            print("            (Pillow's raqm needs fribidi.dll on Windows;")
            print("             uharfbuzz installs from pip and does the same job.)")

    missing_fonts = [f for f in FONT_URLS if not (FONT_DIR / f).exists()]
    missing_tess = [l for l in ("urd", "ara", "fas", "heb")
                    if not (TESS_DIR / f"{l}.traineddata").exists()]
    if missing_fonts or missing_tess:
        ok = False
        print(f"  [MISSING] fonts/models - run: python {Path(__file__).name} --setup")
    else:
        print("  [ok]      fonts and OCR models")

    print("\n" + ("Everything is ready.\n" if ok else "Fix the items above first.\n"))
    return ok


# =========================================================================
#  The four ways of drawing a word. You do not need to change these.
# =========================================================================


# ===== HarfBuzz + FreeType renderer (inlined) =====


import numpy as np
from PIL import Image


def _load(font_path):
    import freetype
    import uharfbuzz as hb

    with open(font_path, "rb") as fh:
        data = fh.read()
    face_hb = hb.Face(data)
    font_hb = hb.Font(face_hb)
    face_ft = freetype.Face(str(font_path))
    return font_hb, face_ft, face_hb.upem


def shape_and_render(text, font_path, px=120, pad=40, direction="rtl",
                     script=None, language=None):
    """
    Returns a PIL image of the text, correctly shaped and ordered.

    HarfBuzz is given the string in logical order with direction='rtl'. It
    returns glyphs already in visual order, left to right, so no separate bidi
    reordering step is needed for a single-direction run. Mixed-direction
    strings would need splitting into runs first - noted as a limitation.
    """
    import freetype
    import uharfbuzz as hb

    font_hb, face_ft, upem = _load(font_path)
    face_ft.set_pixel_sizes(0, px)

    buf = hb.Buffer()
    buf.add_str(text)
    buf.direction = direction
    if script:
        buf.script = script
    if language:
        buf.language = language
    buf.guess_segment_properties()
    if direction:
        buf.direction = direction

    hb.shape(font_hb, buf, None)
    scale = px / upem

    # First pass: work out how big the canvas needs to be. Nastaliq stacks
    # glyphs well above and below the baseline, so measuring rather than
    # guessing matters here.
    pen_x = 0.0
    boxes = []
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        face_ft.load_glyph(info.codepoint,
                           freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_NORMAL)
        bmp = face_ft.glyph.bitmap
        left, top = face_ft.glyph.bitmap_left, face_ft.glyph.bitmap_top
        x = pen_x + pos.x_offset * scale + left
        y = -(pos.y_offset * scale) - top
        boxes.append((x, y, bmp.width, bmp.rows,
                      bytes(bmp.buffer), bmp.pitch, info.codepoint,
                      pos.x_offset, pos.y_offset))
        pen_x += pos.x_advance * scale

    if not boxes:
        return Image.new("L", (px, px), 255)

    min_x = min(b[0] for b in boxes)
    min_y = min(b[1] for b in boxes)
    max_x = max(b[0] + b[2] for b in boxes)
    max_y = max(b[1] + b[3] for b in boxes)

    width = int(max_x - min_x) + 2 * pad
    height = int(max_y - min_y) + 2 * pad
    canvas = np.zeros((height, width), dtype=np.uint8)

    # Second pass: draw. Re-render each glyph because FreeType reuses one
    # internal bitmap slot.
    for x, y, w, h, _, _, gid, xo, yo in boxes:
        if w == 0 or h == 0:
            continue
        face_ft.load_glyph(gid, freetype.FT_LOAD_RENDER |
                           freetype.FT_LOAD_TARGET_NORMAL)
        bmp = face_ft.glyph.bitmap
        arr = np.array(bmp.buffer, dtype=np.uint8)
        if bmp.pitch != bmp.width:
            arr = arr.reshape(bmp.rows, bmp.pitch)[:, :bmp.width]
        else:
            arr = arr.reshape(bmp.rows, bmp.width)

        px0 = int(x - min_x) + pad
        py0 = int(y - min_y) + pad
        px1, py1 = px0 + bmp.width, py0 + bmp.rows
        if px1 > width or py1 > height or px0 < 0 or py0 < 0:
            continue
        # Maximum, not addition - overlapping Nastaliq glyphs would otherwise
        # saturate to black blobs at the join.
        canvas[py0:py1, px0:px1] = np.maximum(canvas[py0:py1, px0:px1], arr)

    return Image.fromarray(255 - canvas, mode="L")


def _hb_available():
    try:
        import freetype  # noqa: F401
        import uharfbuzz  # noqa: F401
        return True
    except ImportError:
        return False


# Page segmentation mode 8 = "one word". Mode 7 ("one text line") makes
# Tesseract expect line context it does not have, and it returns empty
# strings on short Arabic tokens even when they are rendered perfectly.
# On a clean reference set this single flag moved mean CER from 0.25 to 0.03.
# Report the flag in your paper - it is the difference between a working
# metric and a broken one.
TESS_CONFIG = "--psm 8 --oem 1"

# arabic_reshaper emits Unicode Presentation Forms; modern OpenType fonts do
# shaping through font tables and do not contain those codepoints, so the
# glyphs go missing. Counting them is itself a finding.
MISSING_GLYPHS = set()


def _trim(img, pad=20):
    from PIL import Image
    grey = img.convert("L")
    box = Image.eval(grey, lambda p: 255 - p).getbbox()
    if box is None:
        return grey
    crop = grey.crop(box)
    out = Image.new("L", (crop.width + 2 * pad, crop.height + 2 * pad), 255)
    out.paste(crop, (pad, pad))
    return out


def _matplotlib_draw(text, font_path, size=30, dpi=300):
    """Draws text the way SHAP does - through matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from PIL import Image

    prop = font_manager.FontProperties(fname=str(font_path), size=size)
    fig = plt.figure(figsize=(8, 1.5), dpi=dpi)
    fig.patch.set_facecolor("white")
    fig.text(0.98, 0.5, text, fontproperties=prop, ha="right", va="center")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return _trim(Image.open(buf))


def draw_broken_font(word, lang):
    """A0: what a SHAP user sees by default - empty boxes, wrong font."""
    from matplotlib import font_manager
    return _matplotlib_draw(word, Path(font_manager.findfont("DejaVu Sans")))


def draw_no_fix(word, lang):
    """A: right font, but no direction or joining fix. The reported bug."""
    return _matplotlib_draw(word, FONT_DIR / FONTS[lang])


def draw_simple_fix(word, lang):
    """B: the two-line workaround people currently use."""
    import warnings as _w
    import arabic_reshaper
    from bidi.algorithm import get_display
    staged = arabic_reshaper.reshape(word) if lang != "heb" else word
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        img = _matplotlib_draw(get_display(staged), FONT_DIR / FONTS[lang])
    for c in caught:
        m = str(c.message)
        if "missing from font" in m:
            MISSING_GLYPHS.add(m.split("(")[-1].split(")")[0])
    return img


def draw_full_shaping(word, lang, size=30, dpi=300):
    """C: proper shaping. Uses whichever complex-text-layout path is available.

    Two routes to the same result. Pillow's raqm is simpler but needs
    fribidi.dll on Windows; uharfbuzz + freetype-py installs from pip
    everywhere. Report which one you used in the paper - rendering results are
    stack-dependent and a reviewer reproducing your work needs to know."""
    px = int(size * dpi / 72)
    font_path = FONT_DIR / FONTS[lang]

    if _hb_available():
        return _trim(shape_and_render(word, font_path, px=px, pad=10))

    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(str(font_path), px,
                              layout_engine=ImageFont.Layout.RAQM)
    canvas = Image.new("L", (px * len(word) + 600, px * 4), 255)
    ImageDraw.Draw(canvas).text(
        (canvas.width - 150, canvas.height // 2), word, font=font, fill=0,
        anchor="rm", direction="rtl", language=lang)
    return _trim(canvas)


def draw_ocr_floor(word, lang):
    """Not a real condition. Same word drawn huge and clean, to find out how
    many mistakes the OCR software makes even on perfect text. You must
    report this number, or nobody can interpret your other numbers."""
    return draw_full_shaping(word, lang, size=60, dpi=300)


# =========================================================================
#  EDIT HERE  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# =========================================================================
#
#  Replace the body of this function with a call into YOUR package, so that
#  it draws the word the way SHAP_RTL draws it and returns the picture.
#
#  It must return a PIL image (black text on white). If your package saves
#  a PNG file, open it with:  from PIL import Image; Image.open(path)
#
#  Leave it as "return None" for now if your package is not ready - the
#  script will just skip this column.

def draw_your_package(word, lang):
    import sys
    pkg = r"C:\Users\USER\OneDrive - Higher Education Commission\Dr. shahid project\SHAP_package"
    if pkg not in sys.path:
        sys.path.insert(0, pkg)

    from PIL import Image
    from shap_rtl.rtl_utils import set_current_language, render_rtl_label

    names = {"urd": "Urdu", "ara": "Arabic", "fas": "Persian", "heb": "Hebrew"}
    set_current_language(names[lang])

    img = render_rtl_label(word, str(FONT_DIR / FONTS[lang]),
                           font_size_pt=30, dpi=300)

    # The package returns a transparent-background image for compositing onto
    # plots. Converting that straight to greyscale turns transparent pixels
    # black, which buries the text. Flatten onto white first.
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white, rgba)

    img = img.convert("L")

    # If it still came out mostly dark, the glyphs are light-on-dark - invert
    import numpy as np
    if np.asarray(img).mean() < 128:
        img = Image.eval(img, lambda p: 255 - p)

    return _trim(img)
    # Example of what it will look like:
    #
    # import shap_rtl
    # image_path = shap_rtl.render_label(word, language=lang)
    # from PIL import Image
    # return Image.open(image_path)

# =========================================================================
#  END OF EDIT SECTION
# =========================================================================


# A0 removed. Matplotlib's default font (DejaVu Sans) turns out to contain
# 165 Arabic and 54 Hebrew glyphs, so it is not a missing-glyph case at all -
# it rendered the same unshaped text as A. Useful finding for the paper: for
# these scripts the default failure is shaping and direction, not tofu.
CONDITIONS = [
    ("A_no_fix",        draw_no_fix),
    ("B_simple_fix",    draw_simple_fix),
    ("C_full_shaping",  draw_full_shaping),
    ("YOUR_PACKAGE",    draw_your_package),
    ("OCR_FLOOR",       draw_ocr_floor),
]


# =========================================================================
#  Reading the word back and scoring it
# =========================================================================

def clean(text):
    """Remove invisible marks before comparing. These are not visible on the
    page, so counting them as errors would measure the wrong thing. Mention
    this step in your paper."""
    text = unicodedata.normalize("NFC", text)
    invisible = {"\u200e", "\u200f", "\u202a", "\u202b", "\u202c", "\u202d",
                 "\u202e", "\u2066", "\u2067", "\u2068", "\u2069",
                 "\u200c", "\u200d", "\u0640"}
    return "".join(c for c in text if c not in invisible).strip()


def score_one(word, lang, draw_fn):
    import pytesseract
    from jiwer import cer
    os.environ["TESSDATA_PREFIX"] = str(TESS_DIR)
    find_tesseract()
    image = draw_fn(word, lang)
    if image is None:
        return None
    read_back = pytesseract.image_to_string(
        image, lang=lang, config=TESS_CONFIG)
    truth = clean(word)
    if not truth:
        return None
    return min(float(cer(truth, clean(read_back))), 1.0)


def bootstrap(values, n=5000):
    """Gives the '95% confidence interval' - the range the true average is
    likely to sit in. Report it. A single average with no interval will be
    challenged by any reviewer who knows statistics."""
    import numpy as np
    v = np.asarray(values, float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(0)
    means = rng.choice(v, size=(n, v.size), replace=True).mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_test(a, b, n=5000):
    """Tests whether two conditions really differ, on the same words.
    p below 0.05 means the difference is unlikely to be chance."""
    import numpy as np
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    rng = np.random.default_rng(0)
    draws = rng.choice(diff, size=(n, diff.size), replace=True).mean(axis=1)
    p = float((np.abs(draws - diff.mean()) >= abs(diff.mean())).mean())
    return float(diff.mean()), p


# =========================================================================
#  Putting it together
# =========================================================================

DEMO_WORDS = {
    "urd": ["حکومت", "معاشرہ", "لوگوں", "سیاست", "تعلیم"],
    "ara": ["الحكومة", "المجتمع", "الناس", "السياسة", "التعليم"],
    "fas": ["حکومت", "جامعه", "مردم", "سیاست", "آموزش"],
    "heb": ["ממשלה", "חברה", "אנשים", "פוליטיקה", "חינוך"],
}


def load_words(path):
    """Reads words.csv - two columns: lang,word. lang is urd/ara/fas/heb."""
    if not Path(path).exists():
        print(f"\nCannot find {path}.")
        print("Make a file called words.csv that looks like this:\n")
        print("  lang,word")
        print("  urd,حکومت")
        print("  ara,الحكومة\n")
        print("Use the actual feature words your SHAP plots display, not")
        print("dictionary words. At least 200 per language.\n")
        sys.exit(1)
    words = {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            lang = row["lang"].strip()
            word = row["word"].strip()
            if lang and word:
                words.setdefault(lang, []).append(word)
    return words


def env_stamp():
    """Every row records the environment that produced it. Rendering results
    are version-dependent, so a results table without version numbers cannot
    be reproduced or trusted."""
    import matplotlib, platform, sys
    import pytesseract
    try:
        tess = str(pytesseract.get_tesseract_version()).split()[0]
    except Exception:
        tess = "?"
    return {"matplotlib": matplotlib.__version__,
            "python": sys.version.split()[0],
            "platform": platform.system() + "-" + platform.release(),
            "tesseract": tess}


def run(words, out_csv="ocr_results.csv"):
    results, rows = {}, []
    stamp = env_stamp()
    print(f"\nmatplotlib {stamp['matplotlib']}  |  python {stamp['python']}  "
          f"|  tesseract {stamp['tesseract']}  |  {stamp['platform']}")
    for lang, wordlist in words.items():
        if lang not in FONTS:
            print(f"Skipping unknown language code '{lang}'")
            continue
        print(f"\n{lang}  ({len(wordlist)} words)")
        results[lang] = {}
        for name, fn in CONDITIONS:
            scores = []
            for w in wordlist:
                s = score_one(w, lang, fn)
                if s is not None:
                    scores.append(s)
            if not scores:
                print(f"  {name:18s} skipped")
                continue
            results[lang][name] = scores
            mean, lo, hi = bootstrap(scores)
            print(f"  {name:18s} CER = {mean:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
            rows.append({"language": lang, "condition": name, "n": len(scores),
                         "CER": round(mean, 4), "ci_low": round(lo, 4),
                         "ci_high": round(hi, 4), **stamp})

            # Short tokens carry less signal for any OCR engine, so a mean
            # over mixed lengths can hide the effect. Split it out.
            short = [s for s, w in zip(scores, wordlist) if len(w.replace(" ", "")) <= 4]
            longw = [s for s, w in zip(scores, wordlist) if len(w.replace(" ", "")) >= 7]
            if short and longw:
                print(f"  {'':18s}   short(<=4): {sum(short)/len(short):.3f}  "
                      f"long(>=7): {sum(longw)/len(longw):.3f}")

        got = results[lang]
        if "B_simple_fix" in got and "C_full_shaping" in got:
            d, p = paired_test(got["B_simple_fix"], got["C_full_shaping"])
            verdict = "SIGNIFICANT" if p < 0.05 else "not significant"
            print(f"  {'B vs C':18s} difference = {d:+.3f}  p = {p:.3f}  ({verdict})")
            rows.append({"language": lang, "condition": "B_minus_C",
                         "n": len(got["B_simple_fix"]), "CER": round(d, 4),
                         "ci_low": "", "ci_high": f"p={p:.3f}", **stamp})

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["language", "condition", "n",
                                                "CER", "ci_low", "ci_high",
                                                "matplotlib", "python",
                                                "platform", "tesseract"])
        writer.writeheader()
        writer.writerows(rows)

    if MISSING_GLYPHS:
        print(f"\nB_simple_fix triggered {len(MISSING_GLYPHS)} distinct "
              f"missing-glyph errors.")
        print("Those are Unicode Presentation Forms that arabic_reshaper emits")
        print("and modern OpenType fonts do not contain. Report this - it")
        print("explains why the standard workaround fails on current fonts.")

    print(f"\nSaved to {out_csv}. Open it in Excel and paste into your paper.")
    print("\nHOW TO READ THIS:")
    print("  CER 0.00 = drawn perfectly.  CER 1.00 = unreadable.")
    print("  Compare every condition against OCR_FLOOR, not against zero -")
    print("  the floor is how many mistakes the OCR makes on perfect text.")
    print("  If your package matches C_full_shaping, say so honestly and")
    print("  argue your contribution on packaging, fonts and layout.\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="OCR test for RTL SHAP plots")
    ap.add_argument("--setup", action="store_true", help="download fonts and OCR models")
    ap.add_argument("--check", action="store_true", help="check the installation only")
    ap.add_argument("--demo", action="store_true", help="quick run on 5 built-in words")
    ap.add_argument("--words", default="words.csv", help="your word list")
    args = ap.parse_args()

    if args.setup:
        setup()
    elif args.check:
        check_install()
    elif args.demo:
        if check_install():
            import matplotlib
            run(DEMO_WORDS,
                out_csv=f"results_mpl{matplotlib.__version__}_demo.csv")
    else:
        if check_install():
            # Name the output after the input file and the matplotlib version,
            # so runs can never overwrite each other and every result file
            # says on its face which environment produced it.
            import matplotlib
            stem = Path(args.words).stem.replace("words_", "").replace("words", "all")
            out = f"results_mpl{matplotlib.__version__}_{stem}.csv"
            run(load_words(args.words), out_csv=out)
