#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -eu

root=${1:-"$(cd "$(dirname "$0")/.." && pwd)"}
gcv=${2:-"$root/bin/benchmark_model-rv64gcv-20260830"}
gc=${3:-"$root/bin/benchmark_model-rv64gc-20260830"}
out=${4:-"$root/results/model-differential"}
qemu=${QEMU_RISCV64:-qemu-riscv64}
sysroot=${QEMU_LD_PREFIX:-/usr/riscv64-linux-gnu}

run() {
  local exe=$1 vlen=$2 model=$3 shape=$4 input=$5 output=$6
  local args=(
    "--graph=$root/models/$model"
    --input_layer=input
    "--input_layer_shape=$shape"
    "--input_layer_value_files=input:$root/inputs/$input"
    --num_threads=1 --warmup_runs=0 --num_runs=1 --min_secs=0 --max_secs=900
    "--output_filepath=$output"
  )
  if [[ $vlen == gc ]]; then
    "$qemu" -L "$sysroot" "$exe" "${args[@]}"
  else
    "$qemu" -L "$sysroot" -cpu "rv64,v=true,vlen=$vlen,vext_spec=v1.0" \
      "$exe" "${args[@]}"
  fi
}

run_model() {
  local name=$1 model=$2 shape=$3 input=$4
  for vlen in 128 256 512; do
    run "$gcv" "$vlen" "$model" "$shape" "$input" \
      "$out/$name-vlen${vlen}-gcv.raw"
  done
  run "$gc" gc "$model" "$shape" "$input" "$out/$name-gc.raw"
}

mkdir -p "$out"
run_model v1-fp32 mobilenet_v1_fp32.tflite 1,224,224,3 mobilenet-fp32.input
run_model v1-int8 mobilenet_v1_int8.tflite 1,224,224,3 mobilenet-u8.input
run_model v2-fp32 mobilenet_v2_fp32.tflite 1,224,224,3 mobilenet-fp32.input
run_model v2-int8 mobilenet_v2_int8.tflite 1,224,224,3 mobilenet-u8.input
run_model efficientdet-fp32 efficientdet_lite0_fp32.tflite 1,320,320,3 efficientdet-fp32.input
run_model efficientdet-int8 efficientdet_lite0_int8.tflite 1,320,320,3 efficientdet-u8.input
