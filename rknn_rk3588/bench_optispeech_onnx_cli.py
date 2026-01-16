#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf


def _percentile(xs, p):
    xs = np.asarray(xs, dtype=np.float64)
    return float(np.percentile(xs, p))


def _guess_T_from_filename(onnx_path: str, default: int = 256) -> int:
    b = os.path.basename(onnx_path)
    m = re.search(r"(?:^|[_-])T(\d+)(?:[_-]|\.onnx$)", b)
    if m:
        return int(m.group(1))
    return default


def _make_session(onnx_path: str, use_cuda: bool):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.enable_mem_pattern = False

    if use_cuda:
        providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    return ort.InferenceSession(onnx_path, sess_options=so, providers=providers)


def _load_model_from_session(session: ort.InferenceSession):
    from optispeech.onnx.infer import OptiSpeechONNXModel
    return OptiSpeechONNXModel.from_onnx_session(session)


def _pad_to_fixed_T(x: np.ndarray, x_lengths: np.ndarray, T: int, pad_value: int = 0):
    x = np.asarray(x, dtype=np.int64)
    x_lengths = np.asarray(x_lengths, dtype=np.int64)

    if x.ndim != 2:
        raise RuntimeError(f"Expected x rank 2 [B,L], got {x.shape}")
    if x_lengths.ndim != 1 or x_lengths.shape[0] != x.shape[0]:
        raise RuntimeError(f"Expected x_lengths shape [B], got {x_lengths.shape} for x {x.shape}")

    B, L = x.shape
    out = np.full((B, T), pad_value, dtype=np.int64)
    out_lengths = x_lengths.copy()

    for i in range(B):
        li = int(out_lengths[i])
        if li > L:
            li = L
        if li > T:
            li = T
        out[i, :li] = x[i, :li]
        out_lengths[i] = li

    return out, out_lengths


def _unbatch_and_concat(wav: np.ndarray, wav_lengths: np.ndarray):
    wav = np.asarray(wav, dtype=np.float32)
    wav_lengths = np.asarray(wav_lengths, dtype=np.int64).reshape(-1)

    if wav.ndim == 1:
        wav = wav[None, :]
    if wav.ndim != 2:
        raise RuntimeError(f"Expected wav rank 2 [B,N], got {wav.shape}")

    outs = []
    for i in range(wav.shape[0]):
        n = int(wav_lengths[i]) if i < wav_lengths.shape[0] else wav.shape[1]
        n = max(0, min(n, wav.shape[1]))
        outs.append(wav[i, :n].copy())

    if outs:
        cat = np.concatenate(outs, axis=0)
    else:
        cat = np.zeros((0,), dtype=np.float32)
    return outs, cat


def _fmt_ms(x_s: float) -> str:
    return f"{x_s * 1000.0:.2f} ms"


def _fmt_rtf(x_s: float, audio_s: float) -> str:
    if audio_s <= 0:
        return "inf"
    return f"{(x_s / audio_s):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--no-split", action="store_true", help="Do not split text into sentences.")
    ap.add_argument("--T", type=int, default=0, help="Fixed token length for patched ONNX (0 = infer from filename or use 256).")
    ap.add_argument("--pad-id", type=int, default=0, help="Padding ID for x (usually 0).")
    ap.add_argument("--d-factor", type=float, default=1.0)
    ap.add_argument("--p-factor", type=float, default=1.0)
    ap.add_argument("--e-factor", type=float, default=1.0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    T = args.T if args.T > 0 else _guess_T_from_filename(args.onnx, default=256)

    print("ONNX:", args.onnx)
    print("T:", T)
    print("Warmup:", args.warmup, "Runs:", args.runs)
    print("Scales: d=", args.d_factor, "p=", args.p_factor, "e=", args.e_factor)

    session = _make_session(args.onnx, use_cuda=args.cuda)
    model = _load_model_from_session(session)

    try:
        meta = session.get_modelmeta()
        infer_params = json.loads(meta.custom_metadata_map["inference"])
        tp = infer_params.get("text_processor", {})
        print("ONNX meta name:", infer_params.get("name"))
        print("Sample rate:", infer_params.get("sample_rate"))
        print("TextProcessor:", tp)
    except Exception:
        pass

    sr = int(model.sample_rate)

    def run_once(save_audio: bool, tag: str):
        t_all0 = time.perf_counter()

        t0 = time.perf_counter()
        try:
            inputs = model.prepare_input(
                args.text,
                d_factor=args.d_factor,
                p_factor=args.p_factor,
                e_factor=args.e_factor,
                split_sentences=not args.no_split,
            )
        except ImportError as e:
            raise SystemExit(
                f"{e}\n\n"
                "This repo snapshot uses tokenizer=ipa, which depends on piper-phonemize.\n"
                "Install:\n"
                "  pip install piper-phonemize\n"
            )
        t_prepare = time.perf_counter() - t0

        t0 = time.perf_counter()
        inputs = inputs.as_numpy()
        x = inputs.x
        x_lengths = inputs.x_lengths
        x_pad, x_lengths_cap = _pad_to_fixed_T(x, x_lengths, T=T, pad_value=args.pad_id)
        scales = np.array([inputs.d_factor, inputs.p_factor, inputs.e_factor], dtype=np.float32)
        t_numpy_pad = time.perf_counter() - t0

        t0 = time.perf_counter()
        feed = {
            "x": x_pad,
            "x_lengths": x_lengths_cap,
            "scales": scales,
        }
        if getattr(model, "is_multispeaker", False):
            feed["sids"] = np.asarray(inputs.sids, dtype=np.int64)
        if getattr(model, "is_multilanguage", False):
            feed["lids"] = np.asarray(inputs.lids, dtype=np.int64)

        wav, wav_lengths, durations = session.run(None, feed)
        t_ort = time.perf_counter() - t0

        t0 = time.perf_counter()
        wav = np.asarray(wav, dtype=np.float32)
        wav_lengths = np.asarray(wav_lengths, dtype=np.int64).reshape(-1)
        audio_s = float(wav_lengths.sum()) / float(sr) if wav_lengths.size else 0.0
        parts, cat = _unbatch_and_concat(wav, wav_lengths)
        t_post = time.perf_counter() - t0

        t_write = 0.0
        if save_audio:
            t0 = time.perf_counter()
            for i, w in enumerate(parts):
                sf.write(str(outdir / f"{tag}-gen-{i+1}.wav"), w, sr)
            sf.write(str(outdir / "output.wav"), cat, sr)
            t_write = time.perf_counter() - t0

        t_all = time.perf_counter() - t_all0
        rtf_total = (t_all / audio_s) if audio_s > 0 else float("inf")

        max_tok = int(x_lengths_cap.max()) if x_lengths_cap.size else 0

        breakdown = {
            "prepare_s": t_prepare,
            "numpy_pad_s": t_numpy_pad,
            "ort_run_s": t_ort,
            "post_s": t_post,
            "write_s": t_write,
            "total_s": t_all,
        }
        return breakdown, audio_s, rtf_total, max_tok

    def print_breakdown(prefix: str, b: dict, audio_s: float, max_tok: int):
        print(
            f"{prefix} max_tok={max_tok} audio={audio_s:.3f}s "
            f"total={_fmt_ms(b['total_s'])} RTF={_fmt_rtf(b['total_s'], audio_s)} | "
            f"prep={_fmt_ms(b['prepare_s'])}({ _fmt_rtf(b['prepare_s'], audio_s) }) "
            f"np+pad={_fmt_ms(b['numpy_pad_s'])}({ _fmt_rtf(b['numpy_pad_s'], audio_s) }) "
            f"ort={_fmt_ms(b['ort_run_s'])}({ _fmt_rtf(b['ort_run_s'], audio_s) }) "
            f"post={_fmt_ms(b['post_s'])}({ _fmt_rtf(b['post_s'], audio_s) }) "
            f"write={_fmt_ms(b['write_s'])}({ _fmt_rtf(b['write_s'], audio_s) })"
        )

    # Warmup
    for i in range(max(0, args.warmup)):
        b, audio_s, rtf, max_tok = run_once(save_audio=False, tag="warmup")
        print_breakdown(f"[warmup {i+1}/{args.warmup}]", b, audio_s, max_tok)

    # Stats collectors
    totals_ms = []
    rtfs = []
    prep_ms = []
    np_ms = []
    ort_ms = []
    post_ms = []
    write_ms = []
    last_audio_s = 0.0

    for i in range(max(1, args.runs)):
        b, audio_s, rtf_total, max_tok = run_once(save_audio=(i == args.runs - 1), tag=f"run{i:02d}")
        last_audio_s = audio_s

        totals_ms.append(b["total_s"] * 1000.0)
        rtfs.append(rtf_total)
        prep_ms.append(b["prepare_s"] * 1000.0)
        np_ms.append(b["numpy_pad_s"] * 1000.0)
        ort_ms.append(b["ort_run_s"] * 1000.0)
        post_ms.append(b["post_s"] * 1000.0)
        write_ms.append(b["write_s"] * 1000.0)

        print_breakdown(f"[run {i+1}/{args.runs}]", b, audio_s, max_tok)

    def summary_line(name: str, arr_ms):
        return f"{name}: mean {np.mean(arr_ms):.2f} ms  p50 {_percentile(arr_ms,50):.2f} ms  p90 {_percentile(arr_ms,90):.2f} ms"

    print("\n==== Summary ====")
    print(f"Sample rate: {sr} Hz")
    print(f"Audio seconds (last): {last_audio_s:.3f} s")
    print(summary_line("TOTAL", totals_ms))
    print(summary_line("prepare_input", prep_ms))
    print(summary_line("as_numpy+pad", np_ms))
    print(summary_line("ORT session.run", ort_ms))
    print(summary_line("postprocess", post_ms))
    print(summary_line("write_wav", write_ms))
    print(f"RTF: mean {np.mean(rtfs):.3f} p50 {_percentile(rtfs,50):.3f} p90 {_percentile(rtfs,90):.3f}")
    print("Wrote:", str(outdir / "output.wav"))


if __name__ == "__main__":
    main()
