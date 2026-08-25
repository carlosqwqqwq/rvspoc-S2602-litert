#!/usr/bin/env python3
"""Audit the 61 ARM optimized entry mappings against a patched source tree."""

from pathlib import Path
import argparse
import re
import sys


TEST_MAP = (
    ("Shuffled", "fully_connected_test"),
    ("Add", "add_test"),
    ("Mul", "mul_test"),
    ("Lstm", "lstm_test"),
    ("AveragePool", "internal-averagepool_quantized_test"),
    ("MaxPool", "internal-maxpool_quantized_test"),
    ("Softmax", "internal-softmax_quantized_test"),
    ("Logistic", "activations_test"),
    ("Tanh", "activations_test"),
    ("HardSwish", "activations_test"),
    ("Quantize", "quantize_test"),
    ("Requantize", "quantize_test"),
    ("AffineQuantize", "quantize_test"),
    ("Dequantize", "dequantize_test"),
    ("Maximum", "maximum_minimum_test"),
    ("Minimum", "maximum_minimum_test"),
    ("PRelu", "activations_test"),
    ("Arg", "arg_min_max_test"),
)


def read_rows(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not re.match(r"^\|\s*\d+\s*\|", line):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 8:
            raise ValueError(f"bad matrix row: {line}")
        rows.append({"id": int(cells[0]), "entry": cells[1], "route": cells[2],
                     "kind": cells[3], "vlen": cells[4], "gc": cells[5],
                     "diff": cells[6], "status": cells[7]})
    return rows


def clean(value):
    return re.sub(r"[`*]", "", value)


def anchor(entry):
    value = clean(entry).split("（")[0].split("(")[0].strip()
    value = re.sub(r"<.*?>", "", value)
    return value.split()[0]


def route_tokens(route):
    return re.findall(r"(?:rvv_optimized_ops|tensor_utils)::[A-Za-z_][A-Za-z_0-9]*", clean(route))


def target_for(entry):
    name = clean(entry)
    for token, target in TEST_MAP:
        if token in name:
            return target
    return "internal-tensor_utils_test"


def suite_state(suite: Path):
    states = []
    for vlen in (128, 256, 512):
        summary = suite / f"vlen{vlen}-summary"
        if not summary.is_file():
            states.append("MISSING")
            continue
        text = summary.read_text(encoding="utf-8", errors="replace")
        states.append("PASS" if "FAIL " not in text and "PASS " in text else "FAIL")
    return states


def depthwise_inventory_state(source_root: Path, inventory: Path):
    rows = []
    for line in inventory.read_text(encoding="utf-8").splitlines():
        match = re.match(
            r"^\|\s*([FIU])-\d+\s*\|.*\|\s*(RVV-GENERIC(?:-M>1)?)\s*\|",
            line,
        )
        if not match:
            continue
        rows.append((match.group(1), match.group(2)))
    expected = {"F": 14, "I": 22, "U": 22}
    counts = {family: sum(row[0] == family for row in rows) for family in expected}
    files = {
        "F": source_root / "kernels/internal/optimized/depthwiseconv_float.h",
        "I": source_root / "kernels/internal/optimized/integer_ops/depthwise_conv.h",
        "U": source_root / "kernels/internal/optimized/depthwiseconv_uint8.h",
    }
    missing = [str(files[family]) for family in expected if not files[family].is_file()]
    bad_status = [status for _, status in rows if not status.startswith("RVV-GENERIC")]
    if len(rows) != 58 or counts != expected or missing or bad_status:
        return "FAIL", f"rows={len(rows)} counts={counts} missing={missing[:1]}"
    return "PASS", "58 templates (14 FP32 + 22 UINT8 + 22 INT8)"


def test_state(source_root: Path, target: str):
    test_source = target.replace("internal-", "internal/") + ".cc"
    path = source_root / "kernels" / test_source
    cmake = source_root / "kernels" / "CMakeLists.txt"
    registered = cmake.is_file() and test_source in cmake.read_text(
        encoding="utf-8", errors="replace"
    )
    return test_source, "PASS" if path.is_file() and registered else "FAIL"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True,
                        help="patched checkout's tflite directory")
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--suite-root", type=Path)
    parser.add_argument("--depthwise-inventory", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--require-suite", action="store_true")
    args = parser.parse_args()

    rows = read_rows(args.matrix)
    if [row["id"] for row in rows] != list(range(1, 62)):
        raise SystemExit("matrix must contain exactly sequential rows 1..61")

    source = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in (args.source_root / "kernels/internal/optimized").rglob("*")
        if path.is_file() and path.suffix in {".h", ".cc"}
    )
    suite = suite_state(args.suite_root) if args.suite_root else ["N/A"] * 3
    if args.require_suite and suite != ["PASS", "PASS", "PASS"]:
        raise SystemExit(f"official suite is not complete: {suite}")

    output = [
        "id\tentry\troute\tstatic\ttest_target\ttest_source\ttest_registered\t"
        "suite_vlen128\tsuite_vlen256\tsuite_vlen512\tstatus"
    ]
    failures = []
    for row in rows:
        entry_anchor = anchor(row["entry"])
        entry_ok = entry_anchor in source
        tokens = route_tokens(row["route"])
        route_ok = (bool(tokens) and
                    all(token.split("::", 1)[1] in source for token in tokens))
        if clean(row["kind"]) == "RVV-SHARED" and not tokens:
            route_ok = "__riscv_vector" in source
        target = target_for(row["entry"])
        test_source, test_registered = test_state(args.source_root, target)
        static = "PASS" if entry_ok and route_ok and test_registered == "PASS" else "FAIL"
        status = "STATIC-PASS" if static == "PASS" else "STATIC-FAIL"
        if static != "PASS":
            failures.append(f"{row['id']}: {entry_anchor} -> {row['route']}")
        output.append("\t".join(str(value) for value in (
            row["id"], clean(row["entry"]), clean(row["route"]), static, target,
            test_source, test_registered, suite[0], suite[1], suite[2], status)))

    text = "\n".join(output) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    if failures:
        raise SystemExit("static coverage failures:\n" + "\n".join(failures))
    print("PASS: 61 sequential ARM entry mappings and RVV route symbols audited",
          file=sys.stderr)
    if args.depthwise_inventory:
        status, detail = depthwise_inventory_state(args.source_root,
                                                    args.depthwise_inventory)
        if status != "PASS":
            raise SystemExit(f"depthwise inventory failure: {detail}")
        print(f"PASS: depthwise inventory {detail}", file=sys.stderr)


if __name__ == "__main__":
    main()
