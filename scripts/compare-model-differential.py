#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare GCV outputs with the same-source GC scalar reference."""

from pathlib import Path
import struct
import sys


ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/model-differential")
MODELS = (
    ("v1-fp32", "fp32"),
    ("v1-int8", "int8"),
    ("v2-fp32", "fp32"),
    ("v2-int8", "int8"),
    ("efficientdet-fp32", "fp32"),
    ("efficientdet-int8", "int8"),
)


def max_int8_diff(candidate: bytes, reference: bytes) -> int:
    actual = struct.unpack(f"<{len(candidate)}b", candidate)
    expected = struct.unpack(f"<{len(reference)}b", reference)
    return max((abs(a - b) for a, b in zip(actual, expected)), default=0)


def main() -> int:
    for name, dtype in MODELS:
        reference = (ROOT / f"{name}-gc.raw").read_bytes()
        print(f"[{name}] reference_bytes={len(reference)}")
        for vlen in (128, 256, 512):
            candidate = (ROOT / f"{name}-vlen{vlen}-gcv.raw").read_bytes()
            if len(candidate) != len(reference):
                raise AssertionError(f"{name} VLEN={vlen}: output length differs")
            if dtype == "int8":
                diffs = sum(a != b for a, b in zip(candidate, reference))
                maximum = max_int8_diff(candidate, reference)
                print(f"  VLEN={vlen} byte_diffs={diffs} max_int8_diff={maximum}")
                if maximum > 1:
                    raise AssertionError(f"{name} VLEN={vlen}: INT8 error > 1 LSB")
            else:
                actual = struct.unpack(f"<{len(candidate) // 4}f", candidate)
                expected = struct.unpack(f"<{len(reference) // 4}f", reference)
                maximum = max((abs(a - b) for a, b in zip(actual, expected)), default=0.0)
                print(f"  VLEN={vlen} max_abs={maximum:.3e}")
                if maximum > 1e-5:
                    raise AssertionError(f"{name} VLEN={vlen}: FP32 error > 1e-5")
    print("PASS: 6 models x 3 VLEN values match the scalar reference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
