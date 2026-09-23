"""
===========================================================================
 DIAGNOSE  -  see what is actually being rendered and read
===========================================================================

 Run it in the same folder as run_ocr_test.py:

     python diagnose.py

 It takes the first 6 words from words_ara.csv, renders each one under every
 condition, saves the pictures into a folder called "diagnose", and prints
 what Tesseract read back.

 Then open the diagnose folder and look at the images. Whatever is wrong will
 be visible immediately - blank pages, boxes, reversed text, or text that
 looks fine but reads back wrong.
===========================================================================
"""

import csv
import importlib.util
import sys
import unicodedata
from pathlib import Path

OUT = Path("diagnose")
OUT.mkdir(exist_ok=True)

# Load the main script so we reuse exactly the same rendering code
spec = importlib.util.spec_from_file_location("rot", "run_ocr_test.py")
rot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rot)


def clean(t):
    t = unicodedata.normalize("NFC", t)
    bad = {"\u200e", "\u200f", "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",
           "\u2066", "\u2067", "\u2068", "\u2069", "\u200c", "\u200d", "\u0640"}
    return "".join(c for c in t if c not in bad).strip()


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "words_ara.csv"
    if not Path(src).exists():
        print(f"Cannot find {src}")
        sys.exit(1)

    rows = []
    with open(src, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append((r["lang"].strip(), r["word"].strip()))
    rows = rows[:6]

    print(f"\nFont files the script will use:")
    for lang in {l for l, _ in rows}:
        fp = rot.FONT_DIR / rot.FONTS[lang]
        print(f"   {lang}: {fp}  exists={fp.exists()}  "
              f"size={fp.stat().st_size if fp.exists() else 0}")

    print(f"\nComplex text layout available: {rot._hb_available()}")

    conditions = [c for c in rot.CONDITIONS if c[0] != "OCR_FLOOR"]
    conditions += [("OCR_FLOOR", rot.draw_ocr_floor)]

    for lang, word in rows:
        print("\n" + "=" * 70)
        print(f"WORD: {word!r}   ({len(word)} characters, lang={lang})")
        print("=" * 70)

        for name, fn in conditions:
            try:
                img = fn(word, lang)
            except Exception as exc:
                print(f"  {name:18s} RENDER FAILED: {exc}")
                continue
            if img is None:
                print(f"  {name:18s} skipped")
                continue

            safe = "".join(ch if ch.isalnum() else "_" for ch in word)[:20]
            path = OUT / f"{lang}_{safe}_{name}.png"
            img.save(path)

            import os
            import pytesseract
            os.environ["TESSDATA_PREFIX"] = str(rot.TESS_DIR)
            rot.find_tesseract()
            read = clean(pytesseract.image_to_string(
                img, lang=lang, config=rot.TESS_CONFIG))

            from jiwer import cer
            score = min(float(cer(clean(word), read)), 1.0) if clean(word) else 1.0

            flag = ""
            if img.size[0] < 20 or img.size[1] < 20:
                flag = "  <-- image is basically empty"
            print(f"  {name:18s} size={str(img.size):14s} CER={score:.3f}  "
                  f"read={read!r}{flag}")

    print(f"\nImages saved in: {OUT.resolve()}")
    print("\nMOST IMPORTANT CHECK: open that folder and put the")
    print("*_YOUR_PACKAGE.png and *_C_full_shaping.png files side by side.")
    print("They should look identical. If YOUR_PACKAGE looks different -")
    print("letters detached, wrong order, squashed, cut off at an edge -")
    print("that is a real bug in the package, and this is how you find it.")
    print("\nOpen that folder and look at them. Check in this order:")
    print("  1. Is A0_default_font blank or full of boxes?  (it should be)")
    print("  2. Is A_no_fix showing disconnected letters in the wrong order?")
    print("  3. Is C_full_shaping showing correct, joined, right-to-left text?")
    print("  4. Is OCR_FLOOR the same as C but bigger?")
    print("\nIf the images look right but the CER numbers are wrong, the")
    print("problem is in the OCR step. If the images look wrong, the problem")
    print("is in rendering.")


if __name__ == "__main__":
    main()
