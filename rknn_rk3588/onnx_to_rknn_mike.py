#!/usr/bin/env python3
import argparse
from pathlib import Path
from rknn.api import RKNN


def build_one(onnx_path: str, T: int, out_path: str):
    rknn = RKNN(verbose=True)
    rknn.config(target_platform="rk3588", optimization_level=3)

    ret = rknn.load_onnx(
        model=onnx_path,
        inputs=["x", "x_lengths", "scales"],
        input_size_list=[[1, T], [1], [3]],
    )
    if ret != 0:
        rknn.release()
        raise SystemExit("load_onnx failed")

    ret = rknn.build(do_quantization=False)
    if ret != 0:
        rknn.release()
        raise SystemExit("build failed")

    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    ret = rknn.export_rknn(out_path)
    if ret != 0:
        rknn.release()
        raise SystemExit("export_rknn failed")

    rknn.release()
    print("Saved:", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="Input ONNX path")
    ap.add_argument("--T", type=int, nargs="+", required=True, help="One or more T values, e.g. --T 64 128 256")
    ap.add_argument(
        "--out",
        default="mike_T{T}.rknn",
        help="Output path or pattern. You can use {T} placeholder. Default: mike_T{T}.rknn",
    )
    args = ap.parse_args()

    onnx_path = args.onnx
    for T in args.T:
        out_path = args.out.format(T=T)
        build_one(onnx_path, T, out_path)


if __name__ == "__main__":
    main()
