<!-- SPDX-License-Identifier: Apache-2.0 -->

# 题目模型

本目录包含 S2602 要求的六个冻结模型：

| 文件 | 模型变体 | 输入 / 输出类型 | 官方来源 |
| --- | --- | --- | --- |
| `mobilenet_v1_fp32.tflite` | MobileNetV1 FP32 | FP32 / FP32 | [TensorFlow](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz) |
| `mobilenet_v1_int8.tflite` | MobileNetV1 量化 | UINT8 / UINT8 | [TensorFlow](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz) |
| `mobilenet_v2_fp32.tflite` | MobileNetV2 FP32 | FP32 / FP32 | [TensorFlow](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz) |
| `mobilenet_v2_int8.tflite` | MobileNetV2 量化 | UINT8 / UINT8 | [TensorFlow](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz) |
| `efficientdet_lite0_fp32.tflite` | EfficientDet-Lite0 FP32 | FP32 / FP32 | [MediaPipe](https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite) |
| `efficientdet_lite0_int8.tflite` | EfficientDet-Lite0 量化 | UINT8 / FP32 | [MediaPipe](https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite) |

模型文件名与 `SUBMISSION.md`、`results/` 中的记录一致。

EfficientDet-Lite0 使用 MediaPipe Object Detector 发布的官方 LiteRT FlatBuffer；
该 FP32/INT8 文件随提交固定，便于评审直接复现同一模型载荷；模型使用条件以
官方来源页面为准。
