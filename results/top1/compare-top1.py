#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare x86 vs RV output tensors and compute argmax agreement."""
import os
import struct
import sys

PAIRS = [
    ("mobilenet_v1_1.0_224", "mobilenet_v1_fp32", "float32"),
    ("mobilenet_v1_1.0_224_quant", "mobilenet_v1_int8", "bytes"),
    ("mobilenet_v2_1.0_224", "mobilenet_v2_fp32", "float32"),
    ("mobilenet_v2_1.0_224_quant", "mobilenet_v2_int8", "bytes"),
]


def load(path, suffix):
    with open(path, "rb") as f:
        data = f.read()
    if suffix == "float32":
        return struct.unpack("<%df" % (len(data) // 4), data)
    return data


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(__file__)
    rows = []
    for x86_name, rv_name, suffix in PAIRS:
        match = total = 0
        diffs = []
        for i in range(1, 201):
            img = f"img{i:02d}"
            x86_path = os.path.join(root, "x86-ref2-raw", f"{x86_name}-{img}.raw")
            rv_path = os.path.join(root, "rv-raw", f"{rv_name}-{img}.raw")
            x = load(x86_path, suffix)
            r = load(rv_path, suffix)
            if len(x) != len(r):
                diffs.append(f"{img}:size {len(x)}!={len(r)}")
                continue
            total += 1
            if max(range(len(x)), key=x.__getitem__) == max(range(len(r)), key=r.__getitem__):
                match += 1
            else:
                diffs.append(img)
        rows.append((x86_name, match, total, diffs))
        print(f"{x86_name}\tmatch={match}/{total}\t{match / total * 100:.1f}%\tdiffs={diffs[:5]}")
    with open(os.path.join(root, "top1-summary.tsv"), "w", encoding="utf-8") as f:
        for name, match, total, diffs in rows:
            f.write(f"{name}\tmatch={match}/{total}\t{match / total * 100:.1f}%\tdiffs={diffs}\n")


if __name__ == "__main__":
    main()
