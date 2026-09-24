#!/usr/bin/env python3
"""喘ぎ声の作り方を 2 方式で比べる試作（1 セル分）。

- 方式A: Irodori（v4.1-Small）で、対象の声の参照音声を使って直接生成する
  （喘ぎ・吐息の行は対象の喘ぎ、台詞の行は対象の台詞を参照にする）。
- 方式B: 別の声（演者）で同じ行を生成し、Seed-VC で対象の声へ変換する
  （変換先の参照は方式A と同じ使い分け。既定は F0 条件付き 44.1kHz モデル＋音域合わせ）。
- 自動評価: 対象の参照音声との話者類似度（resemblyzer の cos。参考値）。

    uv run python projects/dorei_shokan_voices/compare_voices.py run [--dry-run]
    uv run python projects/dorei_shokan_voices/compare_voices.py page

台本は lines_compare.toml。出力は outputs/dorei_shokan_voices/compare/<name>/
（A/・B_src/・B/・scores.json・index.html）。
"""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import time
from pathlib import Path

import build_voices as bv

LINES = bv.HERE / "lines_compare.toml"
SEEDVC_SH = bv.REPO / "scripts/seedvc.sh"
SEEDVC_BATCH = bv.HERE / "seedvc_batch.py"
TAKES = [101, 202, 303]
AUD = bv.OUT / "audition"

# 比較の設定（今回は inu:standard = プリセット inu seed101、演者は onesan seed202）
NAME = "inu_standard"
TARGET = {"preset": "inu", "seed": 101}
ACTOR = {"preset": "onesan", "seed": 202}


def refs(p: dict) -> dict:
    """行の種類ごとの参照音声（喘ぎ・吐息 = オーディションの喘ぎ、台詞 = 台詞）。"""
    moan = AUD / p["preset"] / f"seed{p['seed']}_mid_m01.wav"
    talk = AUD / p["preset"] / f"seed{p['seed']}_talk.wav"
    return {"moan": moan, "breath": moan, "speech": talk}


def plan(args: argparse.Namespace) -> dict:
    cfg = bv.load_presets(bv.PRESETS_TOML)
    presets = {p["id"]: p for p in cfg["preset"]}
    lines = bv.load_lines(LINES)
    ckpt = bv.checkpoint_path(cfg)
    out = bv.OUT / "compare" / NAME
    t_refs, a_refs = refs(TARGET), refs(ACTOR)
    for r in [*t_refs.values(), *a_refs.values()]:
        if not r.is_file():
            raise SystemExit(f"参照音声が無い: {r}")
    rows, vc_jobs, score_jobs = [], [], []
    for ln in lines:
        for seed in TAKES:
            name = f"{ln['id']}_s{seed}.wav"
            for method, who, ref in (
                ("A", TARGET, t_refs[ln["kind"]]),
                ("B_src", ACTOR, a_refs[ln["kind"]]),
            ):
                cap = bv.caption_for(presets[who["preset"]], ln)
                rows.append(
                    bv.row(
                        text=ln["text"],
                        caption=cap,
                        ckpt=ckpt,
                        out=out / method / name,
                        seed=seed,
                        line=ln,
                        ref_wav=ref,
                        meta={"method": method, "line": ln["id"]},
                    )
                )
            vc_jobs.append(
                {
                    "source": str(out / "B_src" / name),
                    "target": str(t_refs[ln["kind"]]),
                    "out": str(out / "B" / name),
                }
            )
            for method in ("A", "B", "B_src"):
                score_jobs.append(
                    {
                        "wav": str(out / method / name),
                        "ref": str(t_refs[ln["kind"]]),
                        "key": f"{method}/{name}",
                    }
                )
    return {"out": out, "rows": rows, "vc": vc_jobs, "score": score_jobs, "lines": lines}


def seedvc(mode: str, jobs_path: Path, *extra: str) -> None:
    cmd = ["bash", str(SEEDVC_SH), "python", str(SEEDVC_BATCH), mode, "--jobs", str(jobs_path)]
    cmd += list(extra)
    print("[run]", " ".join(cmd), flush=True)
    t0 = time.monotonic()
    rc = subprocess.call(cmd, cwd=bv.REPO)
    print(f"[run] seedvc {mode} rc={rc}（{time.monotonic() - t0:.0f} 秒）", flush=True)
    if rc != 0:
        raise SystemExit(f"seedvc {mode} が失敗した（rc={rc}）")


def cmd_run(args: argparse.Namespace) -> None:
    p = plan(args)
    out: Path = p["out"]
    manifest = bv.write_manifest(p["rows"], f"compare_{NAME}")
    if args.dry_run:
        bv.show_dry_run(manifest, p["rows"])
        print(f"[dry-run] Seed-VC 変換 {len(p['vc'])} 本、類似度 {len(p['score'])} 本")
        return
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    if not args.skip_tts:
        bv.run_batch(manifest, out / "timings.jsonl")
    t1 = time.monotonic()
    vc_jobs = out / "vc_jobs.json"
    vc_jobs.write_text(json.dumps(p["vc"], ensure_ascii=False, indent=1), encoding="utf-8")
    vc_extra = [] if args.no_f0 else ["--f0-condition"]
    if not args.skip_vc:
        seedvc("convert", vc_jobs, *vc_extra)
    t2 = time.monotonic()
    sc_jobs = out / "score_jobs.json"
    sc_jobs.write_text(json.dumps(p["score"], ensure_ascii=False, indent=1), encoding="utf-8")
    seedvc("score", sc_jobs, "--out", str(out / "scores.json"))
    meta = {
        "tts_sec": round(t1 - t0),
        "vc_sec": round(t2 - t1),
        "score_sec": round(time.monotonic() - t2),
        "vc_model": "v1 speech 22kHz" if args.no_f0 else "v1 F0 44.1kHz + auto-f0-adjust",
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"[compare] {meta}")
    print(f"[compare] 比較ページ: {write_page(p)}")


# --------------------------------------------------------------------------- page

CSS = """
:root{--bg:#16161a;--fg:#e8e6e3;--mut:#9a968f;--card:#222228;--line:#34343c;--acc:#e07a9b;--best:#e0c060}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);padding:14px;
font:13px/1.4 "Yu Gothic UI","Hiragino Sans",sans-serif}
h1{font-size:18px;margin:0 0 4px}.note{color:var(--mut);margin:0 0 10px}
.bar{display:flex;gap:14px;align-items:center;margin-bottom:10px;flex-wrap:wrap}
.avg b{font-size:16px}table{border-collapse:collapse;width:100%}
th,td{border:1px solid var(--line);padding:6px;vertical-align:top}th{background:var(--card);text-align:left}
td.ln{width:200px}.ln .id{color:var(--acc);font-family:monospace}.ln .t{font-size:14px}.ln .k{color:var(--mut);font-size:11px}
.take{display:inline-block;margin:0 6px 4px 0;padding:4px;border:1px solid var(--line);border-radius:4px;background:var(--card)}
.take.best{border-color:var(--best);box-shadow:0 0 0 1px var(--best)}
.take .h{display:flex;justify-content:space-between;gap:6px;font-size:11px;color:var(--mut)}
.take .sim{font-family:monospace;color:var(--fg)}.take.best .sim::before{content:"★ ";color:var(--best)}
audio{width:170px;height:30px;display:block}audio.small{width:130px;height:24px;opacity:.8}
.src{font-size:10px;color:var(--mut);margin-top:2px}
.mh{font-size:15px;font-weight:bold}
body.blind .src,body.blind .avg,body.blind .sim,body.blind .take.best .sim::before{display:none}
"""

JS = """
const rows=[...document.querySelectorAll('tr.row')];
function apply(){
  const blind=document.getElementById('blind').checked;
  document.body.classList.toggle('blind',blind);
  rows.forEach(tr=>{
    const a=tr.querySelector('td.mA'),b=tr.querySelector('td.mB');
    const swap=blind&&tr.dataset.swap==='1';
    if(swap) tr.insertBefore(b,a); else tr.insertBefore(a,b);
    a.querySelector('.mh').textContent=blind?(swap?'Y':'X'):'A';
    b.querySelector('.mh').textContent=blind?(swap?'X':'Y'):'B';
  });
  document.querySelectorAll('th.hA').forEach(t=>t.textContent=blind?'X':'方式A（Irodori 直接）');
  document.querySelectorAll('th.hB').forEach(t=>t.textContent=blind?'Y':'方式B（演者→Seed-VC）');
}
function shuffle(){rows.forEach(tr=>tr.dataset.swap=Math.random()<.5?'1':'0');apply();}
document.getElementById('blind').addEventListener('change',()=>{shuffle();});
document.addEventListener('play',e=>{document.querySelectorAll('audio').forEach(x=>{if(x!==e.target)x.pause()})},true);
shuffle();
"""


def write_page(p: dict) -> Path:
    out: Path = p["out"]
    scores = json.loads((out / "scores.json").read_text(encoding="utf-8"))
    meta_p = out / "meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.is_file() else {}
    t_refs = refs(TARGET)

    def rel(path: Path) -> str:
        return html.escape(Path("..", "..", path.relative_to(bv.OUT)).as_posix())

    avg = {}
    for m in ("A", "B", "B_src"):
        v = [s for k, s in scores.items() if k.startswith(f"{m}/")]
        avg[m] = sum(v) / len(v) if v else 0.0
    body = []
    for ln in p["lines"]:
        cells = {}
        for m in ("A", "B"):
            sims = {s: scores.get(f"{m}/{ln['id']}_s{s}.wav") for s in TAKES}
            best = max(
                (s for s in TAKES if sims[s] is not None), key=lambda s: sims[s], default=None
            )
            takes = []
            for s in TAKES:
                name = f"{ln['id']}_s{s}.wav"
                src = ""
                if m == "B":
                    src_sim = scores.get(f"B_src/{name}")
                    src = (
                        f'<div class="src">変換前（{ACTOR["preset"]}）'
                        f"{'' if src_sim is None else f' {src_sim:.3f}'}"
                        f'<audio class="small" controls preload="none" '
                        f'src="{rel(out / "B_src" / name)}"></audio></div>'
                    )
                sim = "" if sims[s] is None else f"{sims[s]:.3f}"
                takes.append(
                    f'<div class="take{" best" if s == best else ""}">'
                    f'<div class="h"><span>take s{s}</span><span class="sim">{sim}</span></div>'
                    f'<audio controls preload="none" src="{rel(out / m / name)}"></audio>{src}</div>'
                )
            cells[m] = f'<td class="m{m}"><div class="mh">{m}</div>{"".join(takes)}</td>'
        ref = t_refs[ln["kind"]]
        body.append(
            f'<tr class="row"><td class="ln"><div class="id">{ln["id"]} '
            f'<span class="k">{ln["phase"]} / {ln["kind"]}</span></div>'
            f'<div class="t">{html.escape(ln["text"])}</div>'
            f'<div class="k">参照（{TARGET["preset"]}・{ref.stem}）</div>'
            f'<audio controls preload="none" src="{rel(ref)}"></audio></td>'
            f"{cells['A']}{cells['B']}</tr>"
        )
    page = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>喘ぎ声 方式比較</title><style>{CSS}</style></head><body>
<h1>喘ぎ声の作り方 比較（{TARGET["preset"]} seed{TARGET["seed"]} ／ {NAME}）</h1>
<p class="note">方式A = Irodori が {TARGET["preset"]} の参照音声で直接生成。方式B = {ACTOR["preset"]}
（seed{ACTOR["seed"]}）の声で生成し、Seed-VC（{html.escape(meta.get("vc_model", ""))}）で
{TARGET["preset"]} へ変換。各行 3 テイク（seed {", ".join(map(str, TAKES))}）。
数値は参照音声との話者類似度（resemblyzer cos、参考値）。★ は各方式・各行で最も高いテイク。</p>
<div class="bar"><label><input type="checkbox" id="blind"> 方式名を隠す（行ごとに X/Y をランダムに入れ替え）</label>
<span class="avg">類似度の平均: A <b>{avg["A"]:.3f}</b> ／ B <b>{avg["B"]:.3f}</b>
（変換前の {ACTOR["preset"]} {avg["B_src"]:.3f}）</span></div>
<table><tr><th>行 ／ 参照</th><th class="hA">方式A（Irodori 直接）</th><th class="hB">方式B（演者→Seed-VC）</th></tr>
{"".join(body)}</table><script>{JS}</script></body></html>
"""
    dst = out / "index.html"
    dst.write_text(page, encoding="utf-8")
    return dst


def cmd_page(args: argparse.Namespace) -> None:
    print(write_page(plan(args)))


def main() -> None:
    ap = argparse.ArgumentParser(description="喘ぎ声の作り方 2 方式の比較（1 セル）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="生成 → Seed-VC 変換 → 類似度 → ページ")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--skip-tts", action="store_true", help="Irodori 生成を飛ばす（作り直し用）")
    r.add_argument("--skip-vc", action="store_true", help="Seed-VC 変換を飛ばす")
    r.add_argument("--no-f0", action="store_true", help="Seed-VC を F0 なし 22kHz モデルにする")
    r.set_defaults(func=cmd_run)
    pg = sub.add_parser("page", help="比較ページだけ作り直す")
    pg.set_defaults(func=cmd_page)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
