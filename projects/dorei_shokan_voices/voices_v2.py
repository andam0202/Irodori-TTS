"""台本 v2（部屋＝プレイの種類ごと 7 行、lines_v2.toml）の生成・品質ゲート・後処理・書き出し。

``build_voices.py`` の v2-* サブコマンドの実体。方式A（Irodori が参照音声で直接生成）:
喘ぎ・吐息の行はセルの喘ぎ参照、台詞の行はセルの台詞参照（採用時の ref_wav）を使う。

品質ゲート（1 行 1 テイク → 測って取り直し）:
- resemblyzer でテイクと参照音声の話者類似度を測る（Seed-VC の venv で実行）
- セル内の 1 回目テイクの平均から ``--sim-margin``（0.08）より下、無音、短すぎ・長すぎのテイクは
  seed を +1000, +2000 して最大 ``--max-retakes`` 回取り直し、合格テイク優先で類似度最大を採る
後処理: 前後の無音トリム（-45 dBFS、前 50ms・後 150ms 残し）→ -16 LUFS → true peak -1 dBTP 以下。

出力:
    outputs/dorei_shokan_voices/v2/<race>_<body>/raw/<line>_s<seed>.wav   （生テイク）
    outputs/dorei_shokan_voices/v2/<race>_<body>/final/<line>.wav         （後処理済み 48kHz）
    outputs/dorei_shokan_voices/v2/<race>_<body>/takes.json / index.html  （記録・試聴ページ）
    <Godot>/game/assets/voices/<race>/<body>/<play>/<line>.ogg + voices_index.json（"plays" キー）
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import build_voices as bv
import numpy as np
import soundfile as sf
import voice_matrix as vm

LINES_V2 = bv.HERE / "lines_v2.toml"
V2_OUT = bv.OUT / "v2"
SEEDVC_SH = bv.REPO / "scripts/seedvc.sh"
SEEDVC_BATCH = bv.HERE / "seedvc_batch.py"
SERVE_URL = "http://localhost:8766/audio/v2/{dir}/index.html"

MIN_SEC, MAX_SEC = 0.5, 12.0  # 生テイクの長さの許容（末尾パディング込み）
SILENT_PEAK = 0.03  # これ未満のピークは無音扱い
TRIM_DB, PRE_MS, POST_MS = -45.0, 50, 150
TARGET_LUFS, PEAK_DBTP = -16.0, -1.0
# ラウドネス基準（LOUDNESS.md）。行の種類ごとの Integrated の目標、許容 ±1 LU、True Peak -1 dBTP 以下
LOUDNESS_TARGETS = {"breath": -20.0, "soft": -18.0, "mid": -16.0, "hard": -14.0, "climax": -14.0}
LOUDNESS_TOL = 1.0
WAV_CEILING_DBTP = -1.5  # ogg 化での inter-sample peak の増えを見込んで wav は -1.5 に抑える
WAV_CEILINGS = (-1.5, -2.5, -3.5, -5.0, -7.0, -9.0, -12.0)  # ogg の True Peak が超えたら順に下げて作り直す（キス音・水音は -5 でも超えることがある）
OGG_Q = "3"  # vorbis q3（約 112kbps 相当、声には十分でサイズを抑える）
# 所要時間の見積もり（inu:standard パイロットの実測: 読み込み 135〜190 秒、合成 2.9 秒/本、
# 取り直し 1 巡目 5/56・2 巡目 1/56、類似度は起動 15 秒 + 0.6 秒/本、後処理・ogg 0.3 秒/本）
EST_LOAD_SEC, EST_SYNTH_SEC = 170.0, 2.9
EST_RETAKE = (5 / 56, 1 / 56)
EST_SCORE_START_SEC, EST_SCORE_SEC, EST_POST_SEC = 15.0, 0.6, 0.3


# --------------------------------------------------------------------------- データ


def load_v2() -> tuple[list[dict], dict]:
    with LINES_V2.open("rb") as f:
        d = bv.tomllib.load(f)
    return d["line"], d["plays"]


def cell_refs(m: dict, key: str) -> dict:
    """セルの参照音声（talk = 採用時の ref_wav、moan = 同じ seed のオーディションの喘ぎ）。"""
    c = m["cells"][key]
    seed = int(c.get("seed") or 0)
    if not (seed and c.get("ref_wav")):
        raise SystemExit(f"{key}: 未採用（seed / ref_wav が無い）")
    talk = bv.REPO / c["ref_wav"]
    if c.get("ref_moan_wav"):
        moan = bv.REPO / c["ref_moan_wav"]
    elif c["mode"] == "preset":
        moan = bv.OUT / "audition" / c["preset"] / f"seed{seed}_mid_m01.wav"
    else:
        moan = vm.MATRIX_OUT / vm.cell_dir(key) / f"seed{seed}_mid_m01.wav"
    for p in (talk, moan):
        if not p.is_file():
            raise SystemExit(f"{key}: 参照音声が無い: {p}")
    # 台本 v2 は台詞なし（全行喘ぎ参照）。speech は旧台本・将来用に残す
    return {"speech": talk, "moan": moan, "breath": moan}


# --------------------------------------------------------------------------- 音の検査・後処理


def inspect(path: Path) -> dict:
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    dur = len(y) / sr
    peak = float(np.abs(y).max()) if len(y) else 0.0
    reason = ""
    if peak < SILENT_PEAK:
        reason = "無音"
    elif dur < MIN_SEC:
        reason = "短すぎ"
    elif dur > MAX_SEC:
        reason = "長すぎ"
    return {"dur": round(dur, 2), "peak": round(peak, 3), "reason": reason}


def _true_peak(y: np.ndarray) -> float:
    from scipy.signal import resample_poly  # noqa: PLC0415

    return float(np.abs(resample_poly(y, 4, 1)).max()) if len(y) else 0.0


def loudness_class(ln: dict) -> str:
    if ln.get("climax"):
        return "climax"
    if ln["kind"] == "breath":
        return "breath"
    return ln.get("level") or "mid"


def _limit(y: np.ndarray, ceiling: float) -> tuple[np.ndarray, bool]:
    """頭だけ tanh で丸め、残る inter-sample peak は全体をわずかに下げる。"""
    knee = ceiling * 0.7
    over = np.abs(y) > knee
    limited = bool(over.any())
    if limited:
        mag = np.abs(y[over])
        y[over] = np.sign(y[over]) * (
            knee + (ceiling - knee) * np.tanh((mag - knee) / (ceiling - knee))
        )
    tp = _true_peak(y)
    if tp > ceiling:
        y = y * (ceiling / tp)
    return y, limited


def postprocess(
    src: Path, dst: Path, target: float = TARGET_LUFS, ceiling_db: float = WAV_CEILING_DBTP
) -> dict:
    import pyloudnorm as pyln  # noqa: PLC0415

    y, sr = sf.read(src, dtype="float64", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    hop = int(sr * 0.01)
    frames = max(1, len(y) // hop)
    rms = np.array(
        [np.sqrt(np.mean(y[i * hop : (i + 1) * hop] ** 2) + 1e-12) for i in range(frames)]
    )
    loud = np.where(20 * np.log10(rms) > TRIM_DB)[0]
    if len(loud):
        a = max(0, loud[0] * hop - int(sr * PRE_MS / 1000))
        b = min(len(y), (loud[-1] + 1) * hop + int(sr * POST_MS / 1000))
        y = y[a:b]
    fi, fo = int(sr * 0.005), int(sr * 0.03)
    y[:fi] *= np.linspace(0, 1, fi)[: len(y[:fi])]
    y[-fo:] *= np.linspace(1, 0, fo)[-len(y[-fo:]) :]
    meter = pyln.Meter(sr)

    def measure(x: np.ndarray) -> float:
        if len(x) / sr >= 0.4:
            v = meter.integrated_loudness(x)
            if np.isfinite(v):
                return float(v)
        return float(20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-12))  # 短すぎるものは RMS

    base = y.copy()
    ceiling = 10 ** (ceiling_db / 20)
    gain_db, limited = 0.0, False
    # 丸めでラウドネスが下がる分を見込んで、目標に入るまでゲインを詰める（最大 6 回）
    for _ in range(6):
        y, limited = _limit(base * 10 ** (gain_db / 20), ceiling)
        err = target - measure(y)
        if abs(err) <= 0.2:
            break
        gain_db += err
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, y.astype(np.float32), sr, subtype="PCM_16")
    return {
        "final_sec": round(len(y) / sr, 2),
        "gain_db": round(float(gain_db), 1),
        "target": target,
        "peak_limited": limited,
    }


# --------------------------------------------------------------------------- 生成


def score(jobs: list[dict], work: Path) -> dict:
    if not jobs:
        return {}
    jp, op = work / "score_jobs.json", work / "scores_tmp.json"
    jp.write_text(json.dumps(jobs, ensure_ascii=False), encoding="utf-8")
    cmd = ["bash", str(SEEDVC_SH), "python", str(SEEDVC_BATCH), "score", "--jobs", str(jp)]
    rc = subprocess.call([*cmd, "--out", str(op)], cwd=bv.REPO)
    if rc != 0:
        raise SystemExit(f"類似度の計算に失敗（rc={rc}）")
    return json.loads(op.read_text(encoding="utf-8"))


class ResidentTTS:
    """1 プロセスでモデルを 1 回だけ読み、全セル・全ラウンド（1 巡目＋取り直し）を合成する。

    ``scripts/batch_infer.py`` の関数（マニフェスト読み込み・ランタイム構築・1 行合成）を import して
    使い、モデルを取り直しのラウンドをまたいで保持する（``run_batch`` はラウンドごとに別プロセスで
    読み直すため 1 回約 150〜190 秒かかっていた）。import できないときは ``run_batch`` に戻る。
    """

    def __init__(self) -> None:
        self.runtime = None
        self.bi = None
        self.load_count = 0
        try:
            import importlib.util  # noqa: PLC0415

            spec = importlib.util.spec_from_file_location("batch_infer", bv.BATCH_INFER)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            for name in ("_load_manifest", "_load_runtime_for_checkpoint", "_synthesize_one"):
                getattr(mod, name)
            self.bi = mod
        except Exception as e:  # noqa: BLE001 - 内部 API が変わったら従来方式に戻る
            print(f"[v2] batch_infer の関数を使えないので別プロセス方式にする: {e}", flush=True)

    def run(self, rows: list[dict], name: str) -> None:
        manifest = bv.write_manifest(rows, name)
        if self.bi is None:
            bv.run_batch(manifest, V2_OUT / "timings.jsonl")
            self.load_count += 1
            return
        bi = self.bi
        records = bi._load_manifest(manifest, root=bv.REPO)
        t0 = time.monotonic()
        ok = fail = 0
        for rec in records:
            try:
                if self.runtime is None:
                    t = time.monotonic()
                    self.runtime = bi._load_runtime_for_checkpoint(
                        rec["checkpoint"],
                        model_device_str=bi.default_runtime_device(),
                        model_precision="fp32",
                        codec_device_str=bi.default_runtime_device(),
                    )
                    self.ckpt = rec["checkpoint"]
                    self.load_count += 1
                    print(f"[v2] モデル読み込み {time.monotonic() - t:.0f} 秒", flush=True)
                if rec["checkpoint"] != self.ckpt:
                    raise RuntimeError("checkpoint が行ごとに違う（v2 は 1 本前提）")
                bi._synthesize_one(
                    self.runtime,
                    text=rec["text"],
                    caption=rec["caption"],
                    ref_wav=rec["ref_wav"],
                    seed=rec["seed"],
                    tail_fade_ms=rec["tail_fade_ms"],
                    tail_pad_out_ms=rec["tail_pad_out_ms"],
                    duration_scale=rec["duration_scale"],
                    num_steps=rec["num_steps"],
                    out_path=rec["out_path"],
                )
                ok += 1
            except Exception as e:  # noqa: BLE001 - 1 行の失敗で止めない（判定で取り直しになる）
                fail += 1
                print(f"[v2] 合成失敗 {rec['out_path']}: {e}", flush=True)
            if (ok + fail) % 50 == 0:
                print(f"[v2] {name}: {ok + fail}/{len(records)}", flush=True)
        print(f"[v2] {name}: ok={ok} fail={fail}（{time.monotonic() - t0:.0f} 秒）", flush=True)


def filter_plays(lines: list[dict], plays: dict, wanted: list[str] | None) -> list[dict]:
    if not wanted:
        return lines
    bad = [p for p in wanted if p not in plays]
    if bad:
        raise SystemExit(f"未定義のプレイ: {bad}（{list(plays)}）")
    return [ln for ln in lines if ln["play"] in wanted]


def cmd_v2_build(args: argparse.Namespace) -> None:
    m = vm.load_matrix()
    cfg = bv.load_presets(args.presets_file)
    presets = vm.presets_by_id(cfg)
    all_lines, plays = load_v2()
    lines = filter_plays(all_lines, plays, args.plays)
    if args.lines:
        lines = bv.pick(lines, args.lines, "行")
    ckpt = bv.checkpoint_path(cfg)
    keys = [
        k
        for k in vm.select_cells(m, args.cells)
        if int(m["cells"][k].get("seed") or 0) and m["cells"][k].get("ref_wav")
    ]
    skipped = [k for k in vm.select_cells(m, args.cells) if k not in keys]
    if skipped:
        print(f"[v2] 未採用のため対象外: {', '.join(skipped)}")
    if not keys:
        raise SystemExit("対象セルが無い")

    # セルごとの状態（takes.json）
    state, info = {}, {}
    for k in keys:
        d = V2_OUT / vm.cell_dir(k)
        tj = d / "takes.json"
        state[k] = json.loads(tj.read_text(encoding="utf-8")) if tj.is_file() else {}
        info[k] = {
            "dir": d,
            "refs": cell_refs(m, k),
            "caption": vm.cell_caption(m, k, presets),
            "seed": int(m["cells"][k]["seed"]),
        }

    def todo_first(k: str) -> list[dict]:
        return [ln for ln in lines if args.redo or ln["id"] not in state[k]]

    def make_row(k: str, ln: dict, seed: int) -> dict:
        i = info[k]
        out = i["dir"] / "raw" / f"{ln['id']}_s{seed}.wav"
        return bv.row(
            text=ln["text"],
            caption=bv.caption_for({"caption": i["caption"]}, ln),
            ckpt=ckpt,
            out=out,
            seed=seed,
            line=ln,
            ref_wav=i["refs"][ln["kind"]],
            meta={"cell": k, "line": ln["id"]},
        )

    rows = []
    for k in keys:
        for ln in todo_first(k):
            state[k][ln["id"]] = {"takes": []}
            rows.append(make_row(k, ln, info[k]["seed"]))
    manifest = bv.write_manifest(rows, "v2_build")
    if args.dry_run:
        bv.show_dry_run(manifest, rows)
        per = {k: sum(1 for r in rows if r["cell"] == k) for k in keys}
        print(f"[dry-run] セルごとの行数: {per}")
        n = len(rows)
        r1, r2 = n * EST_RETAKE[0], n * EST_RETAKE[1]
        rounds = 1 + (r1 > 0) + (r2 > 0)
        est = (
            EST_LOAD_SEC
            + (n + r1 + r2) * EST_SYNTH_SEC
            + rounds * EST_SCORE_START_SEC
            + (n + r1 + r2) * EST_SCORE_SEC
            + n * EST_POST_SEC
        )
        print(
            f"[dry-run] モデル読み込み 1 回（取り直しも同じプロセスで続けて合成）／"
            f" 1 巡目 {n} 本 + 取り直し見込み {r1:.0f} + {r2:.0f} 本"
            f"（パイロットの率 {EST_RETAKE[0]:.0%}・{EST_RETAKE[1]:.0%}）"
        )
        print(f"[dry-run] 推定所要時間 約 {est / 60:.0f} 分（{est / 3600:.1f} 時間）")
        return

    t0 = time.monotonic()
    tts = ResidentTTS()
    by_id = {ln["id"]: ln for ln in lines}
    work = V2_OUT / "_work"
    work.mkdir(parents=True, exist_ok=True)
    counts = {"generated": 0, "retaken": 0}
    for rnd in range(args.max_retakes + 1):
        if not rows:
            break
        tts.run(rows, f"v2_build_r{rnd}")
        counts["generated"] += len(rows)
        if rnd:
            counts["retaken"] += len(rows)
        jobs = [
            {"wav": r["out_path"], "ref": r["ref_wav"], "key": r["out_path"]}
            for r in rows
            if Path(r["out_path"]).is_file()
        ]
        sims = score(jobs, work)
        for r in rows:
            p = Path(r["out_path"])
            take = {"seed": r["seed"], "path": p.relative_to(bv.OUT).as_posix()}
            take.update(inspect(p) if p.is_file() else {"reason": "生成失敗"})
            take["sim"] = sims.get(r["out_path"])
            state[r["cell"]][r["line"]]["takes"].append(take)
        # 合否（セル平均は各行の 1 回目テイク）。今回取った行のうち合格テイクが無いものを取り直す
        this_round = [(r["cell"], r["line"]) for r in rows]
        rows = []
        for k in keys:
            firsts = [
                v["takes"][0].get("sim")
                for lid, v in state[k].items()
                if not lid.startswith("_") and v["takes"]
            ]
            firsts = [x for x in firsts if x is not None]
            state[k]["_mean"] = round(sum(firsts) / len(firsts), 4) if firsts else 0.0
        for k, lid in this_round:
            v, mean = state[k][lid], state[k]["_mean"]
            last = v["takes"][-1]
            if not last.get("reason") and last.get("sim") is not None:
                if last["sim"] < mean - args.sim_margin:
                    last["reason"] = "類似度低"
            if rnd < args.max_retakes and not any(not t.get("reason") for t in v["takes"]):
                rows.append(make_row(k, by_id[lid], info[k]["seed"] + 1000 * (rnd + 1)))

    # 採用テイクの決定と後処理
    for k in keys:
        for lid, v in state[k].items():
            if lid.startswith("_") or not v.get("takes"):
                continue
            if lid not in by_id:  # --plays / --lines で絞ったときは今回の対象行だけ後処理する
                continue
            ok = [t for t in v["takes"] if not t.get("reason")] or [
                t for t in v["takes"] if t.get("reason") not in ("無音", "生成失敗")
            ]
            if not ok:
                v["chosen"] = None
                continue
            best = max(ok, key=lambda t: t.get("sim") or 0)
            v["chosen"] = best["seed"]
            v["post"] = postprocess(
                bv.OUT / best["path"],
                info[k]["dir"] / "final" / f"{lid}.wav",
                LOUDNESS_TARGETS[loudness_class(by_id[lid])],
            )
        state[k]["_meta"] = {"caption": info[k]["caption"], "seed": info[k]["seed"]}
        (info[k]["dir"] / "takes.json").write_text(
            json.dumps(state[k], ensure_ascii=False, indent=1), encoding="utf-8"
        )
        write_page(k, state[k], all_lines, plays)
    print(
        f"[v2] 生成 {counts['generated']} 本（うち取り直し {counts['retaken']}）"
        f" {time.monotonic() - t0:.0f} 秒"
    )
    for k in keys:
        s = state[k]
        n_retake = sum(1 for lid, v in s.items() if not lid.startswith("_") and len(v["takes"]) > 1)
        n_fail = sum(
            1 for lid, v in s.items() if not lid.startswith("_") and v.get("chosen") is None
        )
        print(
            f"[v2] {k}: 類似度平均 {s['_mean']:.3f} 取り直し行 {n_retake} 採用なし {n_fail}"
            f" → {SERVE_URL.format(dir=vm.cell_dir(k))}"
        )
    if not args.no_godot:
        export(keys, state, lines, Path(args.godot_dir), args.force, index_lines=all_lines)
    else:
        print("[v2] Godot へは書き出していない（v2-export で基準適用・実測・書き出し）")


def measure_file(path: Path) -> tuple[float, float]:
    """ffmpeg ebur128 で (Integrated LUFS, True Peak dBTP) を測る。"""
    r = subprocess.run(
        [
            "ffmpeg",
            "-nostats",
            "-hide_banner",
            "-i",
            str(path),
            "-af",
            "ebur128=peak=true",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    tail = r.stderr[r.stderr.rfind("Summary:") :]
    i = re.search(r"I:\s+(-?[\d.]+|-inf) LUFS", tail)
    pk = re.search(r"Peak:\s+(-?[\d.]+|-inf) dBFS", tail)
    f = lambda m: float("-inf") if m is None or m.group(1) == "-inf" else float(m.group(1))  # noqa: E731
    return f(i), f(pk)


def encode_ogg(src: Path, dst: Path, gain_db: float = 0.0) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    af = ["-af", f"volume={gain_db:.2f}dB"] if abs(gain_db) > 1e-3 else []
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(src),
            *af,
            "-ac",
            "1",
            "-ar",
            "44100",
            "-c:a",
            "libvorbis",
            "-q:a",
            OGG_Q,
            str(dst),
        ],
        check=True,
    )


def export(
    keys: list[str],
    state: dict,
    lines: list[dict],
    godot_dir: Path | None,
    force: bool,
    repost: bool = True,
    index_lines: list[dict] | None = None,
) -> dict:
    """final wav（必要なら基準で後処理し直し）→ ogg（出力側に仮置き）→ 実測・外れを直す → Godot へ。"""
    by_id = {ln["id"]: ln for ln in lines}
    report = []
    t0 = time.monotonic()
    for k in keys:
        race, body = k.split(":")
        d = V2_OUT / vm.cell_dir(k)
        for lid, v in state[k].items():
            if lid.startswith("_") or lid not in by_id or not v.get("chosen"):
                continue
            ln = by_id[lid]
            cls = loudness_class(ln)
            target = LOUDNESS_TARGETS[cls]
            wav = d / "final" / f"{lid}.wav"
            take = next(t for t in v["takes"] if t["seed"] == v["chosen"])
            ogg = d / "ogg" / ln["play"] / f"{lid}.ogg"
            fixed = 0
            # vorbis 化でピークが増えることがあるので、外れたら wav の上限を下げて作り直す
            for i, ceil in enumerate(WAV_CEILINGS):
                if repost or i or not wav.is_file():
                    v["post"] = postprocess(bv.OUT / take["path"], wav, target, ceil)
                encode_ogg(wav, ogg)
                lufs, tp = measure_file(ogg)
                if abs(target - lufs) > LOUDNESS_TOL and tp - (target - lufs) <= PEAK_DBTP:
                    encode_ogg(wav, ogg, target - lufs)  # ラウドネスだけのずれは ogg 側で補正
                    lufs, tp = measure_file(ogg)
                if abs(target - lufs) <= LOUDNESS_TOL and tp <= PEAK_DBTP:
                    break
                fixed += 1
            ok = abs(target - lufs) <= LOUDNESS_TOL and tp <= PEAK_DBTP
            report.append(
                {
                    "cell": k,
                    "line": lid,
                    "class": cls,
                    "target": target,
                    "lufs": lufs,
                    "tp": tp,
                    "ok": ok,
                    "refixed": fixed,
                    "bytes": ogg.stat().st_size,
                    "ogg": str(ogg),
                }
            )
        (d / "takes.json").write_text(
            json.dumps(state[k], ensure_ascii=False, indent=1), encoding="utf-8"
        )
    print(f"[v2] ogg 化と実測 {len(report)} 本（{time.monotonic() - t0:.0f} 秒）")
    n = kept = 0
    if godot_dir is not None:
        for r in report:
            race, body = r["cell"].split(":")
            dst = godot_dir / race / body / by_id[r["line"]]["play"] / f"{r['line']}.ogg"
            if dst.is_file() and not force:  # 手修正を守る
                kept += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(r["ogg"], dst)
            n += 1
        print(f"[v2] Godot へ ogg {n} 本（既存のため残した {kept} 本）")
        idx = vm.write_godot_index(
            godot_dir, bv.load_lines(bv.LINES_TOML) + (index_lines or lines)
        )
        print(f"[v2] 索引: {idx}")
    # 既存のレポートに、今回の（セル, 行）ぶんを差し替えて合わせる
    # （一部セル・一部プレイだけ書き出しても全体表になる）
    prev_p = V2_OUT / "loudness_report.json"
    if prev_p.is_file():
        done = {(k, lid) for k in keys for lid in by_id}
        prev = json.loads(prev_p.read_text(encoding="utf-8"))
        report = [r for r in prev if (r["cell"], r["line"]) not in done] + report
    write_loudness_report(report)
    return {"report": report, "copied": n, "kept": kept}


def write_loudness_report(report: list[dict]) -> None:
    out = V2_OUT / "loudness_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    bad = [r for r in report if not r["ok"]]
    classes = list(LOUDNESS_TARGETS)
    cells = sorted({r["cell"] for r in report})
    lines = [
        "| セル | " + " | ".join(classes) + " | 本数 | 合計 KB |",
        "|---|" + "---|" * (len(classes) + 2),
    ]
    for c in cells:
        rs = [r for r in report if r["cell"] == c]
        avg = []
        for cl in classes:
            v = [r["lufs"] for r in rs if r["class"] == cl]
            avg.append(f"{sum(v) / len(v):.1f}" if v else "-")
        lines.append(
            f"| {c} | "
            + " | ".join(avg)
            + f" | {len(rs)} | {sum(r['bytes'] for r in rs) // 1024} |"
        )
    allavg = []
    for cl in classes:
        v = [r["lufs"] for r in report if r["class"] == cl]
        allavg.append(f"{sum(v) / len(v):.1f}" if v else "-")
    lines.append(
        "| **全体** | "
        + " | ".join(allavg)
        + f" | {len(report)} | {sum(r['bytes'] for r in report) // 1024} |"
    )
    lines.append("| 目標 | " + " | ".join(f"{LOUDNESS_TARGETS[c]:.0f}" for c in classes) + " | | |")
    tp_max = max((r["tp"] for r in report), default=float("-inf"))
    lines += ["", f"True Peak の最大: {tp_max:.1f} dBTP ／ 基準外: {len(bad)} 本"]
    lines += [
        f"- {r['cell']} {r['line']}: {r['lufs']:.1f} LUFS / {r['tp']:.1f} dBTP"
        f"（目標 {r['target']:.0f}）"
        for r in bad
    ]
    md = V2_OUT / "loudness_report.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def cmd_v2_export(args: argparse.Namespace) -> None:
    m = vm.load_matrix()
    all_lines, plays = load_v2()
    lines = filter_plays(all_lines, plays, args.plays)
    keys, state = [], {}
    for k in vm.select_cells(m, args.cells):
        tj = V2_OUT / vm.cell_dir(k) / "takes.json"
        if tj.is_file():
            keys.append(k)
            state[k] = json.loads(tj.read_text(encoding="utf-8"))
    if not keys:
        raise SystemExit("書き出すセルが無い（v2-build 前）")
    print(f"[v2] 書き出し対象 {len(keys)} セル")
    godot = None if args.no_godot else Path(args.godot_dir)
    export(keys, state, lines, godot, args.force, repost=not args.no_repost, index_lines=all_lines)


# --------------------------------------------------------------------------- 試聴ページ

PAGE_CSS = """
:root{--bg:#141418;--fg:#e8e6e3;--mut:#8f8b85;--card:#1f1f25;--line:#33333c;--acc:#e07a9b;--warn:#c9a23a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);padding:14px;
font:13px/1.4 "Yu Gothic UI","Hiragino Sans",sans-serif}
h1{font-size:18px;margin:0 0 4px}.note{color:var(--mut);margin:0 0 10px}
.bar{position:sticky;top:0;background:var(--bg);padding:6px 0;display:flex;gap:12px;align-items:center;
flex-wrap:wrap;z-index:2;border-bottom:1px solid var(--line)}
button{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:4px;padding:4px 10px;
cursor:pointer;font:inherit}button:hover{border-color:var(--acc)}button.on{background:var(--acc);color:#1a1016}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:10px;margin-top:10px}
.play{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px}
.play h2{font-size:15px;margin:0}.play .dir{color:var(--mut);font-size:11px;margin-bottom:6px}
.ctl{display:flex;gap:6px;margin-bottom:6px}
.ln{display:flex;gap:6px;align-items:center;padding:3px 4px;border-radius:4px}
.ln.now{background:#3a2330}.ln .id{font-family:monospace;color:var(--acc);width:62px;flex:none}
.ln .t{flex:1}.ln .s{font-family:monospace;color:var(--mut);font-size:11px;width:92px;text-align:right}
.ln .s.retake{color:var(--warn)}.ln.cl .id::after{content:" ★";color:var(--warn)}
.pb{padding:1px 7px}
"""

PAGE_JS = """
const ctx=new (window.AudioContext||window.webkitAudioContext)();
const lp=ctx.createBiquadFilter();lp.type='lowpass';lp.frequency.value=700;lp.Q.value=0.7;
const wallGain=ctx.createGain();const dry=ctx.createGain();
lp.connect(wallGain);wallGain.connect(ctx.destination);dry.connect(ctx.destination);
let muffled=false,cur=null,loop=null,lastId=null,token=0;
function route(){wallGain.gain.value=muffled?0.8:0;dry.gain.value=muffled?0:1;}
route();
const cache={};
async function buf(url){if(!cache[url]){const r=await fetch(url);cache[url]=await ctx.decodeAudioData(await r.arrayBuffer());}return cache[url];}
function stop(){token++;if(cur){try{cur.stop()}catch(e){}cur=null;}document.querySelectorAll('.ln.now').forEach(e=>e.classList.remove('now'));
 document.querySelectorAll('button.rnd').forEach(b=>b.classList.remove('on'));}
async function play(el,tk){
 await ctx.resume();if(tk!==token)return;
 const b=await buf(el.dataset.src);if(tk!==token)return;
 const s=ctx.createBufferSource();s.buffer=b;s.connect(lp);s.connect(dry);
 document.querySelectorAll('.ln.now').forEach(e=>e.classList.remove('now'));el.classList.add('now');
 cur=s;s.start();return new Promise(r=>{s.onended=r});}
async function random(sec,btn){
 stop();const tk=token;btn.classList.add('on');
 const pool=[...sec.querySelectorAll('.ln:not(.cl)')];
 while(tk===token){
  let c=pool.filter(e=>e.dataset.id!==lastId);const el=c[Math.floor(Math.random()*c.length)];
  lastId=el.dataset.id;await play(el,tk);if(tk!==token)break;
  await new Promise(r=>setTimeout(r,400+Math.random()*1600));}
}
async function finish(sec){
 stop();const tk=token;const pool=[...sec.querySelectorAll('.ln:not(.cl)')];
 for(let i=0;i<3&&tk===token;i++){let c=pool.filter(e=>e.dataset.id!==lastId);const el=c[Math.floor(Math.random()*c.length)];
  lastId=el.dataset.id;await play(el,tk);await new Promise(r=>setTimeout(r,300+Math.random()*700));}
 if(tk===token)await play(sec.querySelector('.ln.cl'),tk);
}
document.addEventListener('click',e=>{
 const t=e.target;const sec=t.closest('.play');
 if(t.classList.contains('pb')){stop();play(t.closest('.ln'),token);}
 else if(t.classList.contains('rnd')){if(t.classList.contains('on'))stop();else random(sec,t);}
 else if(t.classList.contains('fin'))finish(sec);
 else if(t.id==='stop')stop();
 else if(t.id==='muffle'){muffled=!muffled;t.classList.toggle('on',muffled);t.textContent=muffled?'ローパス ON（壁越し）':'ローパス OFF（素の音）';route();}
});
"""


def write_page(key: str, st: dict, lines: list[dict], plays: dict) -> Path:
    d = V2_OUT / vm.cell_dir(key)
    secs = []
    for pid, pinfo in plays.items():
        items = []
        for ln in [x for x in lines if x["play"] == pid]:
            v = st.get(ln["id"])
            if not v or not v.get("chosen"):
                continue
            chosen = next(t for t in v["takes"] if t["seed"] == v["chosen"])
            sim = chosen.get("sim")
            n = len(v["takes"])
            s_txt = f"{sim:.3f}" if sim is not None else "-"
            if n > 1:
                s_txt += f"（{n}回目中）"
            items.append(
                f'<div class="ln{" cl" if ln.get("climax") else ""}" data-id="{ln["id"]}" '
                f'data-src="final/{ln["id"]}.wav"><button class="pb">▶</button>'
                f'<span class="id">{ln["id"].split("_")[1]}</span>'
                f'<span class="t">{html.escape(ln["text"])}</span>'
                f'<span class="s{" retake" if n > 1 else ""}" '
                f'title="{html.escape(ln["style"])}">{s_txt}</span></div>'
            )
        if not items:
            continue
        secs.append(
            f'<section class="play"><h2>{html.escape(pinfo["name"])} <span class="dir">{pid}</span></h2>'
            f'<div class="dir">{html.escape(pinfo["direction"])}</div>'
            f'<div class="ctl"><button class="rnd">ランダム再生（部屋の外で聞く）</button>'
            f'<button class="fin">終わりまで（最後に絶頂）</button></div>{"".join(items)}</section>'
        )
    meta = st.get("_meta", {})
    page = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>部屋の声 v2 {html.escape(key)}</title><style>{PAGE_CSS}</style></head><body>
<h1>部屋の声 v2 — {html.escape(key)}（seed {meta.get("seed", "")}）</h1>
<p class="note">{html.escape(meta.get("caption", ""))}<br>
数値は参照音声との話者類似度（セル平均 {st.get("_mean", 0):.3f}）。黄色は取り直した行。★は絶頂行（ランダム再生には入らない）。
後処理済み（無音トリム・-16 LUFS・ピーク -1 dBTP）。</p>
<div class="bar"><button id="muffle">ローパス OFF（素の音）</button><button id="stop">停止</button>
<span class="note" style="margin:0">ランダム再生は同じ行の連続を避け、0.4〜2 秒の間を空けて流し続ける</span></div>
<div class="grid">{"".join(secs)}</div><script>{PAGE_JS}</script></body></html>
"""
    dst = d / "index.html"
    dst.write_text(page, encoding="utf-8")
    return dst


def cmd_v2_page(args: argparse.Namespace) -> None:
    m = vm.load_matrix()
    lines, plays = load_v2()
    for k in vm.select_cells(m, args.cells):
        tj = V2_OUT / vm.cell_dir(k) / "takes.json"
        if tj.is_file():
            write_page(k, json.loads(tj.read_text(encoding="utf-8")), lines, plays)
            print(SERVE_URL.format(dir=vm.cell_dir(k)))


def add_subcommands(sub) -> None:
    b = sub.add_parser("v2-build", help="台本 v2 を生成（品質ゲート・後処理・Godot 書き出し）")
    b.add_argument("--cells", help="race:body をカンマ区切り（* 可）。既定: 採用済み全セル")
    b.add_argument("--plays", nargs="+", help="プレイの種類（既定: 全部）")
    b.add_argument("--lines", nargs="+", help="行 id を指定")
    b.add_argument("--redo", action="store_true", help="記録済みの行も作り直す")
    b.add_argument("--max-retakes", type=int, default=2)
    b.add_argument("--sim-margin", type=float, default=0.08)
    b.add_argument("--godot-dir", default=str(vm.GODOT_VOICES))
    b.add_argument("--no-godot", action="store_true", help="Godot へ書き出さない（パイロット用）")
    b.add_argument("--force", action="store_true", help="Godot 側の同名 ogg も上書きする")
    b.add_argument("--dry-run", action="store_true")
    b.set_defaults(func=cmd_v2_build)

    e = sub.add_parser("v2-export", help="ラウドネス基準で後処理 → ogg → 実測・補正 → Godot")
    e.add_argument("--cells", help="race:body をカンマ区切り（* 可）。既定: 生成済み全セル")
    e.add_argument("--plays", nargs="+", help="プレイの種類（既定: 全部）。他のプレイの ogg・実測は残す")
    e.add_argument("--godot-dir", default=str(vm.GODOT_VOICES))
    e.add_argument("--no-godot", action="store_true", help="ogg 化と実測だけ（Godot へ書かない）")
    e.add_argument("--force", action="store_true", help="Godot 側の同名 ogg も上書きする")
    e.add_argument("--no-repost", action="store_true", help="final wav を作り直さない")
    e.set_defaults(func=cmd_v2_export)

    p = sub.add_parser("v2-page", help="v2 の試聴ページだけ作り直す")
    p.add_argument("--cells")
    p.set_defaults(func=cmd_v2_page)
