"""
===========================================================================
 EXTRACT ARABIC FEATURE WORDS  ->  words.csv
===========================================================================

 Nothing to edit. Just run:

     python extract_words_arabic.py

 It expects these two files, saved by Cell 6 in your notebook:

     data/ara_feature_names.npy
     data/ara_shap_values.npy

 If they are somewhere else, change DATA_DIR on the next line.
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
LANG = "ara"
TOP_K = 15        # features shown per plot - matches max_features=5 in your code
N_INSTANCES = 300
TARGET = 200     # how many words to keep


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
    print(f"   Arabic only      : {len(rtl)}")
    print(f"   mixed-direction  : {len(mixed)}")
    print(f"   no Arabic letters: {len(other)}  (skipped)")

    if not rtl:
        print("\nERROR: no Arabic words found. Check that feature_names really")
        print("contains Arabic text and not indices or English tokens.")
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

    with open("words.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["lang", "word"])
        w.writerows([(LANG, word) for word in chosen])
    print(f"\nWrote words.csv  ({len(chosen)} rows)")

    if mixed:
        with open("words_mixed.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["lang", "word"])
            w.writerows([(LANG, word) for word in mixed])
        print(f"Wrote words_mixed.csv  ({len(mixed)} rows)")
        print("\nRun the OCR test on that file separately - it is your")
        print("mixed-direction evidence.")
    else:
        print("\nNo mixed-direction features found. Your preprocessing probably")
        print("strips hashtags, mentions and digits. That is fine, but say so")
        print("in the paper - it means the mixed-direction claim is untested.")

    print("\nNext:  python run_ocr_test.py")


if __name__ == "__main__":
    main()
