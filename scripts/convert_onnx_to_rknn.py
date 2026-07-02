#!/usr/bin/env python3
# 功能：将 ONNX 模型转换为 RKNN 模型。


import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Convert a YOLOv5 ONNX model to RKNN.")
    parser.add_argument("--onnx", required=True, help="Input ONNX model path")
    parser.add_argument("--output", required=True, help="Output RKNN model path")
    parser.add_argument("--target-platform", default="rk3588", help="RKNN target platform")
    parser.add_argument("--quantize", action="store_true", help="Enable int8 quantization")
    parser.add_argument("--dataset", default="", help="Calibration image list for int8 quantization")
    args = parser.parse_args()

    try:
        from rknn.api import RKNN
    except Exception as exc:
        raise SystemExit(
            "Failed to import rknn.api.RKNN. Install/activate rknn-toolkit2, "
            "not only rknn-toolkit-lite2. Original error: {}".format(exc)
        )

    onnx_path = Path(args.onnx).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    dataset_path = Path(args.dataset).expanduser().resolve() if args.dataset else None

    if not onnx_path.is_file():
        raise SystemExit("ONNX model not found: {}".format(onnx_path))
    if args.quantize and (dataset_path is None or not dataset_path.is_file()):
        raise SystemExit("int8 quantization needs a valid --dataset image list")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rknn = RKNN(verbose=True)
    try:
        print("RKNN config: target_platform={}".format(args.target_platform))
        ret = rknn.config(
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
            target_platform=args.target_platform,
        )
        if ret != 0:
            raise SystemExit("rknn.config failed: {}".format(ret))

        print("Loading ONNX:", onnx_path)
        ret = rknn.load_onnx(model=str(onnx_path))
        if ret != 0:
            raise SystemExit("rknn.load_onnx failed: {}".format(ret))

        print("Building RKNN: quantize={}".format(args.quantize))
        ret = rknn.build(
            do_quantization=bool(args.quantize),
            dataset=str(dataset_path) if dataset_path else None,
        )
        if ret != 0:
            raise SystemExit("rknn.build failed: {}".format(ret))

        print("Exporting RKNN:", output_path)
        ret = rknn.export_rknn(str(output_path))
        if ret != 0:
            raise SystemExit("rknn.export_rknn failed: {}".format(ret))

        print("convert_onnx_to_rknn: success")
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
