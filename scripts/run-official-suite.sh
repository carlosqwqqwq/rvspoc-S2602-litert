#!/usr/bin/env bash
set -eu

build_root=${1:?usage: run-official-suite.sh BUILD_ROOT VLEN [OUT_DIR]}
vlen=${2:?usage: run-official-suite.sh BUILD_ROOT VLEN [OUT_DIR]}
out=${3:-"$build_root/kernel-suite-vlen$vlen"}
mkdir -p "$out"
find "$build_root/kernels" -maxdepth 1 -type f -name '*test' -printf '%p\n' | sort >"$out/list"

run_one() {
  exe=$1
  name=$(basename "$exe")
  log="$out/$name.log"
  if timeout 600 qemu-riscv64 -L /usr/riscv64-linux-gnu \
      -cpu "rv64,v=true,vlen=$vlen,vext_spec=v1.0" \
      "$exe" --gtest_color=no >"$log" 2>&1; then
    printf 'PASS %s\n' "$name"
  else
    printf 'FAIL %s\n' "$name"
  fi
}

export out vlen
export -f run_one
xargs -P4 -n1 bash -c 'run_one "$1"' bash <"$out/list" >"$out/summary"
printf 'TOTAL '; wc -l <"$out/list"
printf 'PASS '; grep -c '^PASS ' "$out/summary" || true
printf 'FAIL '; grep -c '^FAIL ' "$out/summary" || true
cat "$out/summary"
! grep -q '^FAIL ' "$out/summary"
