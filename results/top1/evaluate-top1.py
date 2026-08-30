#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Evaluate true Top-1 on the frozen 200-image Imagenette subset."""
import os
import struct

CLASSES = [
    # The MobileNet TFLite outputs contain the ImageNet background slot at
    # index 0, so the label-file line numbers are shifted by one.
    ("n01440764", "tench", 1),
    ("n02102040", "English springer", 218),
    ("n02979186", "cassette player", 483),
    ("n03000684", "chain saw", 492),
    ("n03028079", "church", 498),
    ("n03394916", "French horn", 567),
    ("n03417042", "garbage truck", 570),
    ("n03425413", "gas pump", 572),
    ("n03445777", "golf ball", 575),
    ("n03888257", "parachute", 702),
]
PAIRS = [
    ("mobilenet_v1_fp32", "float32"),
    ("mobilenet_v1_int8", "bytes"),
    ("mobilenet_v2_fp32", "float32"),
    ("mobilenet_v2_int8", "bytes"),
]


def load(path, kind):
    with open(path, "rb") as f:
        data = f.read()
    if kind == "float32":
        return struct.unpack("<%df" % (len(data) // 4), data)
    return data


def argmax(path, kind):
    values = load(path, kind)
    return max(range(len(values)), key=values.__getitem__)


def main():
    root = os.path.dirname(__file__)
    with open(os.path.join(root, "ground-truth.tsv"), "w", encoding="utf-8") as labels:
        labels.write("image\tclass_dir\tclass_name\timagenet_id\n")
        for i in range(1, 201):
            cls_dir, cls_name, target = CLASSES[(i - 1) // 20]
            labels.write(f"img{i:02d}\t{cls_dir}\t{cls_name}\t{target}\n")

    rows = []
    for model, kind in PAIRS:
        x86_correct = rv_correct = 0
        x86_rv_agree = 0
        for i in range(1, 201):
            target = CLASSES[(i - 1) // 20][2]
            if model == "mobilenet_v1_fp32":
                x86_name = "mobilenet_v1_1.0_224"
            elif model == "mobilenet_v1_int8":
                x86_name = "mobilenet_v1_1.0_224_quant"
            elif model == "mobilenet_v2_fp32":
                x86_name = "mobilenet_v2_1.0_224"
            else:
                x86_name = "mobilenet_v2_1.0_224_quant"
            x86_index = argmax(os.path.join(root, "x86-ref2-raw", f"{x86_name}-img{i:02d}.raw"), kind)
            rv_index = argmax(os.path.join(root, "rv-raw", f"{model}-img{i:02d}.raw"), kind)
            x86_correct += x86_index == target
            rv_correct += rv_index == target
            x86_rv_agree += x86_index == rv_index
        rows.append((model, x86_correct, rv_correct, x86_rv_agree, kind))

    with open(os.path.join(root, "top1-accuracy.tsv"), "w", encoding="utf-8") as out:
        out.write("model\tx86_correct\trv_correct\ttotal\tx86_accuracy\trv_accuracy\tdelta_pp\targmax_agree\trequirement\n")
        for model, x86_correct, rv_correct, agree, kind in rows:
            delta = (rv_correct - x86_correct) / 2.0
            limit = 0.1 if kind == "float32" else 1.0
            verdict = "PASS" if abs(delta) <= limit else "FAIL"
            out.write(f"{model}\t{x86_correct}\t{rv_correct}\t200\t{x86_correct / 2:.2f}%\t{rv_correct / 2:.2f}%\t{delta:+.2f}\t{agree}/200\t{verdict}\n")
            print(f"{model}\tx86={x86_correct}/200\trv={rv_correct}/200\tdelta_pp={delta:+.2f}\tagree={agree}/200\t{verdict}")


if __name__ == "__main__":
    main()
