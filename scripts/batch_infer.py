#!/usr/bin/env python3
"""Model-resident batch inference tool for Irodori-TTS.

Repeatedly invoking ``infer.py`` (one subprocess per line) reloads the codec
and the model checkpoint from disk every single time, which costs roughly
130s/line even though the actual synthesis itself is fast. This tool runs as
a single long-lived process instead:

- The DACVAE codec is loaded exactly once and shared across every line.
- Each distinct ``checkpoint`` is loaded into an ``InferenceRuntime`` at most
  once and cached (keyed by the checkpoint path); lines that share a
  checkpoint reuse the cached runtime instead of reloading the model.

The per-line synthesis procedure and all default parameter values mirror
``infer.py`` exactly (``num-steps=40``, CFG scales, tail-trim settings, etc.)
so that output audio is equivalent to running ``infer.py`` line-by-line.

Manifest format (JSONL, one JSON object per line):

    {"text": "...", "caption": "... (optional)",
     "checkpoint": "<path>", "ref_wav": "<path>", "out_path": "<path>",
     "seed": 42, "tail_fade_ms": 120, "tail_pad_out_ms": 250}

``checkpoint`` / ``ref_wav`` / ``out_path`` may be absolute or relative to the
Irodori-TTS project root (cwd); existence of ``checkpoint``/``ref_wav`` is
validated up front. ``out_path``'s parent directory is created automatically.
``seed`` may be omitted, in which case a deterministic seed is derived from
the line's own content (no wall-clock time or global RNG involved, so the
same manifest always reproduces the same seeds). ``tail_fade_ms`` /
``tail_pad_out_ms`` default to 120 / 250 (the values recommended in
``CLAUDE.md`` / the LoRA test scripts) when omitted. ``caption`` is optional.

Usage:

    uv run python scripts/batch_infer.py --manifest <path.jsonl> [--device cuda] \\
        [--precision fp32|bf16]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

# irodori_tts is not installed into site-packages (no editable install); scripts/ tools that
# import it must add the project root to sys.path first (same pattern as scripts/encode_latents.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from irodori_tts.codec import DACVAECodec
from irodori_tts.config import ModelConfig
from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    _load_checkpoint_for_inference,
    _maybe_compile_inference_model,
    _move_inference_module,
    default_runtime_device,
    resolve_cfg_scales,
    resolve_runtime_device,
    resolve_runtime_dtype,
    save_wav,
)
from irodori_tts.model import TextToLatentRFDiT
from irodori_tts.tail import postprocess_tail
from irodori_tts.tokenizer import PretrainedTextTokenizer

# infer.py の argparse 既定値と同一。dramadeus/irodori_batch_worker.py と同じ方針で、
# infer.py が明示指定しないパラメータはすべてここで infer.py のデフォルトを再現する。
CODEC_REPO_DEFAULT = "Aratako/Semantic-DACVAE-Japanese-32dim"
CODEC_PRECISION_DEFAULT = "fp32"
CODEC_DETERMINISTIC_ENCODE_DEFAULT = True
CODEC_DETERMINISTIC_DECODE_DEFAULT = True
MAX_REF_SECONDS_DEFAULT = 30.0
REF_NORMALIZE_DB_DEFAULT = -16.0
REF_ENSURE_MAX_DEFAULT = True
NUM_STEPS_DEFAULT = 40
T_SCHEDULE_MODE_DEFAULT = "linear"
SWAY_COEFF_DEFAULT = -1.0
CFG_SCALE_TEXT_DEFAULT = 3.0
CFG_SCALE_CAPTION_DEFAULT = 3.0
CFG_SCALE_SPEAKER_DEFAULT = 5.0
CFG_GUIDANCE_MODE_DEFAULT = "independent"
CFG_MIN_T_DEFAULT = 0.5
CFG_MAX_T_DEFAULT = 1.0
SPEAKER_UNCOND_MODE_DEFAULT = "mask"
TAIL_WINDOW_SIZE_DEFAULT = 20
TAIL_STD_THRESHOLD_DEFAULT = 0.05
TAIL_MEAN_THRESHOLD_DEFAULT = 0.1
TAIL_MARGIN_MS_DEFAULT = 100.0

# CLAUDE.md で推奨されている既定値（LoRA テストスクリプトの TAIL_ARGS と同一）。
TAIL_FADE_MS_DEFAULT = 120.0
TAIL_PAD_OUT_MS_DEFAULT = 250.0


class ManifestError(ValueError):
    """Raised for malformed manifest lines."""


def _resolve_existing_path(raw: str, *, root: Path, field: str, line_no: int) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise ManifestError(f"line {line_no}: {field} not found: {path}")
    return path


def _resolve_out_path(raw: str, *, root: Path) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _derive_seed(record: dict) -> int:
    """Derive a deterministic seed from a manifest line's own content.

    Used only when ``seed`` is omitted from the manifest. Avoids wall-clock
    time or global RNG state so that re-running the same manifest always
    produces the same seeds.
    """
    payload = json.dumps(
        {
            "text": record.get("text"),
            "caption": record.get("caption"),
            "checkpoint": record.get("checkpoint"),
            "ref_wav": record.get("ref_wav"),
            "out_path": record.get("out_path"),
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    # 63-bit non-negative int, matching infer.py's random-seed range (secrets.randbits(63)).
    return int.from_bytes(digest[:8], byteorder="big") & ((1 << 63) - 1)


def _load_manifest(manifest_path: Path, *, root: Path) -> list[dict]:
    lines: list[dict] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if stripped == "":
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"line {line_no}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ManifestError(f"line {line_no}: expected a JSON object.")

            for required_field in ("text", "checkpoint", "ref_wav", "out_path"):
                if required_field not in record:
                    raise ManifestError(
                        f"line {line_no}: missing required field '{required_field}'."
                    )

            checkpoint_path = _resolve_existing_path(
                record["checkpoint"], root=root, field="checkpoint", line_no=line_no
            )
            ref_wav_path = _resolve_existing_path(
                record["ref_wav"], root=root, field="ref_wav", line_no=line_no
            )
            out_path = _resolve_out_path(record["out_path"], root=root)

            seed = record.get("seed")
            resolved_seed = int(seed) if seed is not None else _derive_seed(record)

            lines.append(
                {
                    "line_no": line_no,
                    "text": str(record["text"]),
                    "caption": record.get("caption"),
                    "checkpoint": str(checkpoint_path),
                    "ref_wav": str(ref_wav_path),
                    "out_path": out_path,
                    "seed": resolved_seed,
                    "tail_fade_ms": float(record.get("tail_fade_ms", TAIL_FADE_MS_DEFAULT)),
                    "tail_pad_out_ms": float(
                        record.get("tail_pad_out_ms", TAIL_PAD_OUT_MS_DEFAULT)
                    ),
                }
            )
    return lines


def _load_runtime_for_checkpoint(
    checkpoint_path: str,
    shared_codec: DACVAECodec,
    *,
    model_device_str: str,
    model_precision: str,
    codec_key_fields: dict,
) -> InferenceRuntime:
    """Load only the model for ``checkpoint_path``, reusing ``shared_codec``.

    Mirrors ``InferenceRuntime.from_key`` step-by-step (see
    ``irodori_tts/inference_runtime.py``), except the codec is not reloaded.
    """
    model_device = resolve_runtime_device(model_device_str)
    model_dtype = resolve_runtime_dtype(precision=model_precision, device=model_device)

    model_state, model_cfg_dict, train_cfg = _load_checkpoint_for_inference(Path(checkpoint_path))
    model_cfg = ModelConfig(**model_cfg_dict)

    model = TextToLatentRFDiT(model_cfg).to(model_device)
    model.load_state_dict(model_state)
    model = _move_inference_module(model, device=model_device, dtype=model_dtype)
    model.eval()
    model = _maybe_compile_inference_model(model, enabled=False, dynamic=False)

    tokenizer = PretrainedTextTokenizer.from_pretrained(
        repo_id=model_cfg.text_tokenizer_repo,
        add_bos=bool(model_cfg.text_add_bos),
        local_files_only=False,
    )
    if tokenizer.vocab_size != model_cfg.text_vocab_size:
        raise ValueError(
            f"text_vocab_size mismatch: checkpoint text_vocab_size={model_cfg.text_vocab_size} "
            f"but tokenizer ({model_cfg.text_tokenizer_repo}) vocab_size={tokenizer.vocab_size}."
        )

    caption_tokenizer = None
    if model_cfg.use_caption_condition:
        caption_tokenizer = PretrainedTextTokenizer.from_pretrained(
            repo_id=model_cfg.caption_tokenizer_repo_resolved,
            add_bos=model_cfg.caption_add_bos_resolved,
            local_files_only=False,
        )
        if caption_tokenizer.vocab_size != model_cfg.caption_vocab_size_resolved:
            raise ValueError(
                f"caption_vocab_size mismatch: checkpoint caption_vocab_size="
                f"{model_cfg.caption_vocab_size_resolved} but tokenizer "
                f"({model_cfg.caption_tokenizer_repo_resolved}) "
                f"vocab_size={caption_tokenizer.vocab_size}."
            )

    default_text_max_len = 256
    default_caption_max_len = default_text_max_len
    if isinstance(train_cfg, dict):
        ckpt_text_max_len = train_cfg.get("max_text_len")
        if isinstance(ckpt_text_max_len, int) and ckpt_text_max_len > 0:
            default_text_max_len = int(ckpt_text_max_len)
        ckpt_caption_max_len = train_cfg.get("max_caption_len")
        if isinstance(ckpt_caption_max_len, int) and ckpt_caption_max_len > 0:
            default_caption_max_len = int(ckpt_caption_max_len)
        else:
            default_caption_max_len = default_text_max_len

    if model_cfg.latent_dim != shared_codec.latent_dim:
        raise ValueError(
            f"Latent dimension mismatch: checkpoint latent_dim={model_cfg.latent_dim} "
            f"but codec latent_dim={shared_codec.latent_dim}."
        )

    key = RuntimeKey(
        checkpoint=str(checkpoint_path),
        model_device=str(model_device),
        codec_repo=codec_key_fields["codec_repo"],
        model_precision=model_precision,
        codec_device=codec_key_fields["codec_device"],
        codec_precision=codec_key_fields["codec_precision"],
        codec_deterministic_encode=codec_key_fields["codec_deterministic_encode"],
        codec_deterministic_decode=codec_key_fields["codec_deterministic_decode"],
        compile_model=False,
        compile_dynamic=False,
    )

    return InferenceRuntime(
        key=key,
        model_cfg=model_cfg,
        train_cfg=train_cfg if isinstance(train_cfg, dict) else None,
        model=model,
        tokenizer=tokenizer,
        caption_tokenizer=caption_tokenizer,
        codec=shared_codec,
        default_text_max_len=default_text_max_len,
        default_caption_max_len=default_caption_max_len,
    )


def _synthesize_one(
    runtime: InferenceRuntime,
    *,
    text: str,
    caption: str | None,
    ref_wav: str,
    seed: int,
    tail_fade_ms: float,
    tail_pad_out_ms: float,
    out_path: Path,
) -> None:
    use_speaker_for_request = bool(runtime.model_cfg.use_speaker_condition_resolved)
    use_caption_condition = bool(
        runtime.model_cfg.use_caption_condition
        and caption is not None
        and str(caption).strip() != ""
    )
    cfg_scale_text, cfg_scale_caption, cfg_scale_speaker, messages = resolve_cfg_scales(
        cfg_guidance_mode=CFG_GUIDANCE_MODE_DEFAULT,
        cfg_scale_text=CFG_SCALE_TEXT_DEFAULT,
        cfg_scale_caption=CFG_SCALE_CAPTION_DEFAULT,
        cfg_scale_speaker=CFG_SCALE_SPEAKER_DEFAULT,
        cfg_scale=None,
        use_caption_condition=use_caption_condition,
        use_speaker_condition=use_speaker_for_request,
    )
    for msg in messages:
        print(f"[batch] {msg}", flush=True)

    result = runtime.synthesize(
        SamplingRequest(
            text=str(text),
            caption=None if caption is None else str(caption),
            ref_wav=str(ref_wav),
            ref_latent=None,
            ref_embed=None,
            no_ref=False,
            ref_normalize_db=REF_NORMALIZE_DB_DEFAULT,
            ref_ensure_max=REF_ENSURE_MAX_DEFAULT,
            num_candidates=1,
            decode_mode="sequential",
            seconds=None,
            duration_scale=1.0,
            max_ref_seconds=MAX_REF_SECONDS_DEFAULT,
            max_text_len=None,
            max_caption_len=None,
            num_steps=NUM_STEPS_DEFAULT,
            cfg_scale_text=cfg_scale_text,
            cfg_scale_caption=cfg_scale_caption,
            cfg_scale_speaker=cfg_scale_speaker,
            cfg_guidance_mode=CFG_GUIDANCE_MODE_DEFAULT,
            cfg_scale=None,
            cfg_min_t=CFG_MIN_T_DEFAULT,
            cfg_max_t=CFG_MAX_T_DEFAULT,
            truncation_factor=None,
            rescale_k=None,
            rescale_sigma=None,
            context_kv_cache=True,
            speaker_kv_scale=None,
            speaker_kv_min_t=None,
            speaker_kv_max_layers=None,
            speaker_uncond_mode=SPEAKER_UNCOND_MODE_DEFAULT,
            seed=int(seed),
            t_schedule_mode=T_SCHEDULE_MODE_DEFAULT,
            sway_coeff=SWAY_COEFF_DEFAULT,
            trim_tail=True,
            tail_window_size=TAIL_WINDOW_SIZE_DEFAULT,
            tail_std_threshold=TAIL_STD_THRESHOLD_DEFAULT,
            tail_mean_threshold=TAIL_MEAN_THRESHOLD_DEFAULT,
            tail_margin_ms=TAIL_MARGIN_MS_DEFAULT,
            lora_adapter=None,
        ),
        log_fn=None,
    )

    audio = postprocess_tail(
        result.audio,
        result.sample_rate,
        fade_ms=float(tail_fade_ms),
        pad_out_ms=float(tail_pad_out_ms),
    )
    save_wav(out_path, audio, result.sample_rate)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Model-resident batch inference for Irodori-TTS: loads the codec once and caches "
            "one InferenceRuntime per checkpoint, replacing the slow per-line infer.py subprocess "
            "loop (~130s/line)."
        )
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to a JSONL manifest (one synthesis request per line).",
    )
    parser.add_argument(
        "--device",
        default=default_runtime_device(),
        help="Device used for both the model and the codec (e.g. cuda, mps, cpu).",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "bf16"],
        default="fp32",
        help="Model precision for weights/compute (codec precision is always fp32, matching "
        "infer.py's defaults).",
    )
    args = parser.parse_args()

    root = Path.cwd()
    manifest_path = Path(str(args.manifest)).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        print(f"[batch] manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    try:
        records = _load_manifest(manifest_path, root=root)
    except ManifestError as exc:
        print(f"[batch] manifest error: {exc}", file=sys.stderr)
        return 2

    model_device = str(args.device)
    codec_device = str(args.device)
    model_precision = str(args.precision)

    codec_key_fields = {
        "codec_repo": CODEC_REPO_DEFAULT,
        "codec_device": codec_device,
        "codec_precision": CODEC_PRECISION_DEFAULT,
        "codec_deterministic_encode": CODEC_DETERMINISTIC_ENCODE_DEFAULT,
        "codec_deterministic_decode": CODEC_DETERMINISTIC_DECODE_DEFAULT,
    }

    print(
        f"[batch] loading codec repo={CODEC_REPO_DEFAULT} device={codec_device} "
        "(loaded once, shared across all lines)",
        flush=True,
    )
    t0 = time.monotonic()
    codec_dtype = resolve_runtime_dtype(
        precision=CODEC_PRECISION_DEFAULT, device=resolve_runtime_device(codec_device)
    )
    shared_codec = DACVAECodec.load(
        repo_id=CODEC_REPO_DEFAULT,
        device=codec_device,
        dtype=codec_dtype,
        deterministic_encode=CODEC_DETERMINISTIC_ENCODE_DEFAULT,
        deterministic_decode=CODEC_DETERMINISTIC_DECODE_DEFAULT,
    )
    print(f"[batch] codec loaded ({time.monotonic() - t0:.1f}s)", flush=True)

    # checkpoint パス -> InferenceRuntime。同一checkpointの行はここでヒットしモデル再ロードを
    # スキップする。
    runtime_cache: dict[str, InferenceRuntime] = {}

    total = len(records)
    ok = 0
    fail = 0
    elapsed_times: list[float] = []

    for record in records:
        line_no = record["line_no"]
        checkpoint = record["checkpoint"]
        t_line = time.monotonic()
        try:
            runtime = runtime_cache.get(checkpoint)
            if runtime is None:
                print(f"[batch] loading checkpoint {checkpoint}", flush=True)
                t_load = time.monotonic()
                runtime = _load_runtime_for_checkpoint(
                    checkpoint,
                    shared_codec,
                    model_device_str=model_device,
                    model_precision=model_precision,
                    codec_key_fields=codec_key_fields,
                )
                runtime_cache[checkpoint] = runtime
                print(f"[batch] checkpoint loaded ({time.monotonic() - t_load:.1f}s)", flush=True)

            _synthesize_one(
                runtime,
                text=record["text"],
                caption=record["caption"],
                ref_wav=record["ref_wav"],
                seed=record["seed"],
                tail_fade_ms=record["tail_fade_ms"],
                tail_pad_out_ms=record["tail_pad_out_ms"],
                out_path=record["out_path"],
            )
            elapsed = time.monotonic() - t_line
            elapsed_times.append(elapsed)
            print(f"[batch] 行{line_no} OK ({elapsed:.1f}s)", flush=True)
            ok += 1
        except Exception as exc:  # noqa: BLE001 - 1行失敗しても続行する
            elapsed = time.monotonic() - t_line
            print(f"[batch] 行{line_no} FAIL ({elapsed:.1f}s): {exc}", flush=True)
            traceback.print_exc()
            fail += 1

    avg_sec = (sum(elapsed_times) / len(elapsed_times)) if elapsed_times else 0.0
    print(f"[batch] summary: total={total} ok={ok} fail={fail}", flush=True)
    print(f"[batch] 平均 {avg_sec:.1f}秒/行", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
