#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf


def _percentile(xs, p):
    xs = np.asarray(xs, dtype=np.float64)
    return float(np.percentile(xs, p))


def _ms(s: float) -> float:
    return float(s) * 1000.0


def _pct(part_ms: float, total_ms: float) -> float:
    return 0.0 if total_ms <= 0 else (100.0 * part_ms / total_ms)


def read_inference_meta_from_onnx(onnx_path: str):
    import onnx

    m = onnx.load(onnx_path)
    meta = {}
    for p in m.metadata_props:
        meta[p.key] = p.value
    if "inference" in meta:
        try:
            return json.loads(meta["inference"])
        except Exception:
            return None
    return None


def build_text_processor_from_meta(infer_meta: dict, lang_override: str | None):
    from optispeech.onnx.infer import TextProcessor

    tp_cfg = (infer_meta or {}).get("text_processor", {}) or {}
    langs = tp_cfg.get("languages") or ["en-us"]
    lang = (lang_override or langs[0]).strip().lower()

    tp = TextProcessor(
        tokenizer=tp_cfg.get("tokenizer", "ipa"),
        add_blank=tp_cfg.get("add_blank", False),
        add_bos_eos=tp_cfg.get("add_bos_eos", False),
        normalize_text=bool(tp_cfg.get("normalize_text", True)),
        languages=[l.strip().lower() for l in langs],
    )
    return tp, lang, tp_cfg


def phoneme_ids_to_x(phoneme_ids, split_sentences: bool):
    if split_sentences:
        if not isinstance(phoneme_ids, (list, tuple)) or (
            phoneme_ids and not isinstance(phoneme_ids[0], (list, tuple))
        ):
            raise RuntimeError(
                f"Expected list[list[int]] when split_sentences=True, got {type(phoneme_ids)}"
            )
        ids = list(phoneme_ids[0]) if phoneme_ids else []
    else:
        if not isinstance(phoneme_ids, (list, tuple)) or (
            phoneme_ids and isinstance(phoneme_ids[0], (list, tuple))
        ):
            raise RuntimeError(
                f"Expected list[int] when split_sentences=False, got {type(phoneme_ids)}"
            )
        ids = list(phoneme_ids)

    x = np.asarray(ids, dtype=np.int64).reshape(1, -1)
    x_lengths = np.asarray([x.shape[1]], dtype=np.int64)
    return x, x_lengths


def pad_to_T(x: np.ndarray, x_lengths: np.ndarray, T: int, pad_id: int):
    x = np.asarray(x, dtype=np.int64)
    x_lengths = np.asarray(x_lengths, dtype=np.int64).reshape(-1)

    if x.ndim != 2:
        raise RuntimeError(f"Expected x [B,L], got {x.shape}")
    if x.shape[0] != 1:
        raise RuntimeError("This script expects batch=1")

    L = x.shape[1]
    real_len = int(x_lengths[0]) if x_lengths.size else L
    real_len = min(real_len, L)
    real_len = min(real_len, T)

    out = np.full((1, T), pad_id, dtype=np.int64)
    out[0, :real_len] = x[0, :real_len]
    out_len = np.array([real_len], dtype=np.int64)
    return out, out_len


def load_rknn(rknn_path: str, core_mask: str):
    from rknnlite.api import RKNNLite

    rknn = RKNNLite(verbose=False)
    ret = rknn.load_rknn(rknn_path)
    if ret != 0:
        raise SystemExit(f"load_rknn failed: {ret}")

    if core_mask == "0":
        mask = RKNNLite.NPU_CORE_0
    elif core_mask == "1":
        mask = RKNNLite.NPU_CORE_1
    elif core_mask == "2":
        mask = RKNNLite.NPU_CORE_2
    else:
        mask = RKNNLite.NPU_CORE_0_1_2

    try:
        rknn.set_core_mask(mask)
    except Exception:
        pass

    ret = rknn.init_runtime()
    if ret != 0:
        raise SystemExit(f"init_runtime failed: {ret}")

    return rknn


def crop_wav(wav: np.ndarray, wav_lengths: np.ndarray):
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    wav_lengths = np.asarray(wav_lengths, dtype=np.int64).reshape(-1)
    if wav_lengths.size >= 1:
        n = int(wav_lengths[0])
        if 0 < n <= wav.size:
            wav = wav[:n]
    return wav


def fmt_run_line(tag: str, m: dict):
    a = m["audio_s"]
    total_ms = _ms(m["total_s"])
    rtf = m["rtf"]
    x = a / m["total_s"] if m["total_s"] > 0 else 0.0
    return (
        f"{tag:<12} tok={m['tok_len']:<3d}  audio={a:>6.3f}s  "
        f"total={total_ms:>8.2f}ms  RTF={rtf:>6.3f}  ({x:>5.2f}x)"
        f" | prep={_ms(m['prep_s']):>7.2f}  pad={_ms(m['pad_s']):>6.2f}"
        f"  npu={_ms(m['npu_s']):>8.2f}  post={_ms(m['post_s']):>6.2f}"
        f"  write={_ms(m['write_s']):>7.2f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rknn", required=True, help="e.g. mike_T64.rknn")
    ap.add_argument("--text", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--onnx", default=None, help="Optional: mike_T64_patched.onnx (for metadata)")
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--pad-id", type=int, default=0)
    ap.add_argument("--d-factor", type=float, default=1.0)
    ap.add_argument("--p-factor", type=float, default=1.0)
    ap.add_argument("--e-factor", type=float, default=1.0)
    ap.add_argument("--lang", default=None, help="Override language (e.g. en-us)")
    ap.add_argument("--sr", type=int, default=24000, help="Sample rate if no ONNX metadata is provided")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--core-mask", default="012", choices=["0", "1", "2", "012"], help="NPU core mask")
    ap.add_argument("--split-sentences", action="store_true", help="Tokenizer splits; we synth only the first sentence.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    infer_meta = None
    if args.onnx:
        infer_meta = read_inference_meta_from_onnx(args.onnx)

    sr = args.sr
    name = None
    tp_cfg = None
    if infer_meta:
        name = infer_meta.get("name")
        if "sample_rate" in infer_meta:
            try:
                sr = int(infer_meta["sample_rate"])
            except Exception:
                pass

    tp, lang, tp_cfg = build_text_processor_from_meta(infer_meta or {}, args.lang)

    print("RKNN:", args.rknn)
    if args.onnx:
        print("ONNX(meta):", args.onnx)
    print("T:", args.T)
    print("Warmup:", args.warmup, "Runs:", args.runs)
    print("Scales: d=", args.d_factor, "p=", args.p_factor, "e=", args.e_factor)
    if name:
        print("Model name:", name)
    print("Lang:", lang, "TextProcessor:", tp_cfg if tp_cfg else "(default ipa)")
    print("Sample rate:", sr)

    rknn = load_rknn(args.rknn, args.core_mask)

    def run_once(write_audio: bool):
        t_all0 = time.perf_counter()

        t0 = time.perf_counter()
        phoneme_ids, _ = tp(args.text, lang, split_sentences=args.split_sentences)
        x, x_lengths = phoneme_ids_to_x(phoneme_ids, split_sentences=args.split_sentences)
        t_prep = time.perf_counter() - t0

        t0 = time.perf_counter()
        tok_len = int(x_lengths[0]) if x_lengths.size else 0
        if tok_len > args.T:
            print(f"WARN: token length {tok_len} > T={args.T}. Truncating (speech may be cut).")
        x_pad, x_len_cap = pad_to_T(x, x_lengths, args.T, args.pad_id)
        scales = np.array([args.d_factor, args.p_factor, args.e_factor], dtype=np.float32)
        t_pad = time.perf_counter() - t0

        t0 = time.perf_counter()
        outs = rknn.inference(inputs=[x_pad, x_len_cap, scales])
        t_npu = time.perf_counter() - t0

        t0 = time.perf_counter()
        wav = crop_wav(outs[0], outs[1])
        audio_s = float(len(wav)) / float(sr) if len(wav) else 0.0
        t_post = time.perf_counter() - t0

        t_write = 0.0
        if write_audio:
            t0 = time.perf_counter()
            sf.write(str(outdir / "output.wav"), wav.astype(np.float32, copy=False), sr)
            t_write = time.perf_counter() - t0

        t_all = time.perf_counter() - t_all0
        rtf_total = (t_all / audio_s) if audio_s > 0 else float("inf")

        return {
            "prep_s": t_prep,
            "pad_s": t_pad,
            "npu_s": t_npu,
            "post_s": t_post,
            "write_s": t_write,
            "total_s": t_all,
            "audio_s": audio_s,
            "rtf": rtf_total,
            "tok_len": int(x_len_cap[0]) if x_len_cap.size else 0,
        }

    # Warmup
    for i in range(max(0, args.warmup)):
        m = run_once(write_audio=False)
        print(fmt_run_line(f"[warmup {i+1}]", m))

    # Runs
    metrics = []
    for i in range(max(1, args.runs)):
        m = run_once(write_audio=(i == args.runs - 1))
        metrics.append(m)
        print(fmt_run_line(f"[run {i+1}]", m))

    # Summary
    totals = [m["total_s"] for m in metrics]
    preps = [m["prep_s"] for m in metrics]
    pads = [m["pad_s"] for m in metrics]
    npus = [m["npu_s"] for m in metrics]
    posts = [m["post_s"] for m in metrics]
    writes = [m["write_s"] for m in metrics]
    audios = [m["audio_s"] for m in metrics]
    rtfs = [m["rtf"] for m in metrics]

    mean_audio = float(np.mean(audios)) if audios else 0.0
    mean_total_s = float(np.mean(totals)) if totals else 0.0
    mean_total_ms = mean_total_s * 1000.0

    def stage_row(name, stage_s_mean):
        ms = stage_s_mean * 1000.0
        rtf = (stage_s_mean / mean_audio) if mean_audio > 0 else float("inf")
        return name, ms, _pct(ms, mean_total_ms), rtf

    rows = [
        stage_row("text->ids", float(np.mean(preps)) if preps else 0.0),
        stage_row("pad/scales", float(np.mean(pads)) if pads else 0.0),
        stage_row("RKNN infer", float(np.mean(npus)) if npus else 0.0),
        stage_row("postprocess", float(np.mean(posts)) if posts else 0.0),
        stage_row("write_wav", float(np.mean(writes)) if writes else 0.0),
    ]

    print("\n==== Summary (per-run, excludes warmup) ====")
    print(f"Sample rate: {sr} Hz")
    print(f"Audio: {mean_audio:.3f} s  (p50 {_percentile(audios,50):.3f}  p90 {_percentile(audios,90):.3f})")
    if mean_audio > 0 and mean_total_s > 0:
        print(f"TOTAL: {mean_total_ms:.2f} ms  |  RTF {mean_total_s/mean_audio:.3f}  |  {mean_audio/mean_total_s:.2f}x realtime")
    else:
        print(f"TOTAL: {mean_total_ms:.2f} ms")

    print("\n" + f"{'Stage':<12} {'ms':>10} {'%total':>8} {'RTF':>8}")
    print("-" * 42)
    for name, ms, pct, rtf in rows:
        rtf_str = "inf" if not np.isfinite(rtf) else f"{rtf:.3f}"
        print(f"{name:<12} {ms:>10.2f} {pct:>7.1f}% {rtf_str:>8}")
    print("-" * 42)
    print(f"{'TOTAL':<12} {mean_total_ms:>10.2f} {100.0:>7.1f}% {np.mean(rtfs):>8.3f}")
    print("\nTop takeaway:")
    if mean_total_ms > 0:
        print(f"- RKNN inference is {rows[2][2]:.1f}% of total time.")
    if mean_audio > 0 and mean_total_s > 0:
        print(f"- End-to-end speed is {mean_audio/mean_total_s:.2f}x realtime (RTF {mean_total_s/mean_audio:.3f}).")
    print("Wrote:", str(outdir / "output.wav"))

    rknn.release()


if __name__ == "__main__":
    main()
