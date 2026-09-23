"""
===========================================================================
 EXTRACT FEATURE WORDS  ->  words_<lang>.csv
===========================================================================

 Run once per language:

     python extract_words_arabic.py ara
     python extract_words_arabic.py urd
     python extract_words_arabic.py fas
     python extract_words_arabic.py heb

 Each run expects two files saved by Cell 6 in that language's notebook:

     data/<lang>_feature_names.npy
     data/<lang>_shap_values.npy

 When all four are done, combine them:

     python extract_words_arabic.py --merge

 That writes one words.csv, which is what the OCR test reads.
===========================================================================
"""

import csv
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# --------------------------------------------------------------- settings
DATA_DIR = Path("data")
VALID = ("ara", "urd", "fas", "heb")
LANG = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "ara"
TOP_K = 5        # features shown per plot - 
N_INSTANCES = 300
TARGET = 200     # how many words to keep


VERSION = "v2 - multi-language"


def banner():
    print(f"[extract_words {VERSION}]  language = {LANG}")
    if LANG not in VALID and "--merge" not in sys.argv:
        print(f"Unknown language '{LANG}'. Use one of: {', '.join(VALID)}")
        sys.exit(1)


def load():
    names_file = DATA_DIR / f"{LANG}_feature_names.npy"
    values_file = DATA_DIR / f"{LANG}_shap_values.npy"

    missing = [f for f in (names_file, values_file) if not f.exists()]
    if missing:
        print("\nERROR: cannot find these files:\n")
        for f in missing:
            print(f"   {f.resolve()}")
        print("\nRun Cell 6 in your Arabic notebook first, and make sure it")
        print("saved into a folder called 'data' next to this script.")
        print(f"\nI am looking in: {Path.cwd().resolve()}")
        if DATA_DIR.exists():
            print("Files I can see in data/:")
            for f in sorted(DATA_DIR.iterdir()):
                print(f"   {f.name}")
        sys.exit(1)

    names = np.load(names_file, allow_pickle=True)
    values = np.load(values_file)
    print(f"Loaded {len(names)} feature names and SHAP values of shape {values.shape}")
    return names, values


def is_rtl(word):
    return any("\u0590" <= c <= "\u05FF" or "\u0600" <= c <= "\u06FF"
               or "\u0750" <= c <= "\u077F" or "\uFB50" <= c <= "\uFDFF"
               for c in word)


def is_mixed(word):
    """RTL letters sitting next to Latin letters or digits. These go in a
    separate file - they are the mixed-direction evidence your paper claims
    and does not currently show."""
    return is_rtl(word) and any(c.isascii() and c.isalnum() for c in word)


def main():
    banner()
    names, values = load()

    if values.ndim == 3:
        print("Note: 3D SHAP array, taking class 1.")
        values = values[:, :, 1]

    if values.shape[1] != len(names):
        print(f"\nERROR: {values.shape[1]} columns of SHAP values but "
              f"{len(names)} feature names. These do not match - they were "
              f"probably saved from different runs. Re-run Cell 6.")
        sys.exit(1)

    # Pool the top features from each instance. This is the set that actually
    # gets drawn on a plot.
    counter = Counter()
    n = min(N_INSTANCES, values.shape[0])
    for row in values[:n]:
        for i in np.argsort(-np.abs(row))[:TOP_K]:
            if row[i] != 0:
                counter[str(names[i])] += 1

    ranked = [w for w, _ in counter.most_common()]
    rtl = [w for w in ranked if is_rtl(w) and not is_mixed(w)]
    mixed = [w for w in ranked if is_mixed(w)]
    other = [w for w in ranked if not is_rtl(w)]

    print(f"\nAcross {n} instances, {len(counter)} distinct features appeared:")
    print(f"   RTL letters only : {len(rtl)}")
    print(f"   mixed-direction  : {len(mixed)}")
    print(f"   no Arabic letters: {len(other)}  (skipped)")

    if not rtl:
        print(f"\nERROR: no {LANG} words found. Check that feature_names really")
        print("contains the language's text and not indices or English tokens.")
        sys.exit(1)

    # Sample across word lengths. Longer words are harder to render - more
    # joining contexts - so an all-short sample understates your own problem.
    rng = random.Random(0)
    bands = ([w for w in rtl if len(w.replace(" ", "")) <= 4],
             [w for w in rtl if 5 <= len(w.replace(" ", "")) <= 7],
             [w for w in rtl if len(w.replace(" ", "")) >= 8])
    chosen = []
    for band in bands:
        rng.shuffle(band)
        chosen.extend(band[:max(1, TARGET // 3)])
    leftover = [w for w in rtl if w not in set(chosen)]
    rng.shuffle(leftover)
    chosen.extend(leftover[:max(0, TARGET - len(chosen))])

    lengths = [len(w) for w in chosen]
    ngrams = sum(1 for w in chosen if " " in w)
    print(f"\nSampled {len(chosen)} words")
    print(f"   length  min {min(lengths)} / median {int(np.median(lengths))} / max {max(lengths)}")
    print(f"   two-word (bigram) features: {ngrams}")

    if int(np.median(lengths)) <= 4:
        print("\n   WARNING: median length is very short. Raise TOP_K to 15")
        print("   at the top of this file and run again.")
    if len(chosen) < 100:
        print(f"\n   WARNING: only {len(chosen)} words. Your confidence intervals")
        print("   will be wide. Raise TOP_K, or report the small n honestly.")

    out_name = f"words_{LANG}.csv"
    with open(out_name, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["lang", "word"])
        w.writerows([(LANG, word) for word in chosen])
    print(f"\nWrote {out_name}  ({len(chosen)} rows)")

    if mixed:
        with open(f"words_mixed_{LANG}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["lang", "word"])
            w.writerows([(LANG, word) for word in mixed])
        print(f"Wrote words_mixed_{LANG}.csv  ({len(mixed)} rows)")
        print("\nRun the OCR test on that file separately - it is your")
        print("mixed-direction evidence.")
    else:
        print("\nNo mixed-direction features found. Your preprocessing probably")
        print("strips hashtags, mentions and digits. That is fine, but say so")
        print("in the paper - it means the mixed-direction claim is untested.")

    done = sorted(Path(".").glob("words_*.csv"))
    done = [f for f in done if "mixed" not in f.name]
    print(f"\nLanguages extracted so far: {[f.stem.replace('words_','') for f in done]}")
    if len(done) < 4:
        print("Run this again for the remaining languages.")
    else:
        print("All four done. Next:  python extract_words_arabic.py --merge")




def merge():
    """Combine words_<lang>.csv files into one words.csv for the OCR test."""
    files = [f for f in sorted(Path(".").glob("words_*.csv"))
             if "mixed" not in f.name]
    if not files:
        print("No words_<lang>.csv files found here.")
        sys.exit(1)
    rows = []
    for f in files:
        with open(f, encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            n = 0
            for row in reader:
                rows.append((row["lang"], row["word"]))
                n += 1
        print(f"  {f.name:24s} {n} words")
    with open("words.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["lang", "word"])
        w.writerows(rows)
    print(f"\nWrote words.csv  ({len(rows)} rows total)")
    print("Next:  python run_ocr_test.py")


if __name__ == "__main__":
    if "--merge" in sys.argv:
        merge()
    elif LANG not in VALID:
        print(f"Unknown language '{LANG}'. Use one of: {VALID}")
        sys.exit(1)
    else:
        main()
