#!/usr/bin/env python3
"""奴隷娼館リメイク用ボイスプリセットのオーディション・採用・一括生成。

プリセット定義は ``presets.toml``、台本は ``lines.toml``（どちらもこのフォルダ）。
生成はモデル常駐バッチ ``scripts/batch_infer.py`` にマニフェスト（JSONL）を渡して行う。

段階1（オーディション）: caption のみ（no-ref）で各プリセット × seed 6 本 × 2 行を作り、
試聴ページを書き出す::

    uv run python projects/dorei_shokan_voices/build_voices.py audition
    uv run python projects/dorei_shokan_voices/build_voices.py page      # ページだけ作り直す

段階2（採用 → 本生成）: 選んだ seed の「talk」を参照音声として固定し、lines.toml 全本を作る::

    uv run python projects/dorei_shokan_voices/build_voices.py adopt onesan 303
    uv run python projects/dorei_shokan_voices/build_voices.py build
    uv run python projects/dorei_shokan_voices/build_voices.py build --presets onesan --phases mid

出力:
    outputs/dorei_shokan_voices/audition/<preset>/seed<N>_<line>.wav
    outputs/dorei_shokan_voices/audition/index.html
    outputs/dorei_shokan_voices/voices/<preset>/<line>.wav      （48kHz 生成そのまま）
    outputs/dorei_shokan_voices/ogg/<preset>/<line>.ogg         （Godot 取り込み用 44.1kHz/mono）

種族×体型の割り当て（voice_matrix.json、実体は voice_matrix.py）::

    uv run python projects/dorei_shokan_voices/build_voices.py matrix-serve     # 設定ページ
    uv run python projects/dorei_shokan_voices/build_voices.py matrix-audition --cells "inu:*,*:kogara"
    uv run python projects/dorei_shokan_voices/build_voices.py matrix-adopt inu:kogara 303
    uv run python projects/dorei_shokan_voices/build_voices.py matrix-build     # → Godot assets/voices

``--dry-run`` はマニフェストを書いて先頭行と本数を表示するだけで GPU を使わない。
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # 本体環境は 3.10
    import tomli as tomllib

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PRESETS_TOML = HERE / "presets.toml"
LINES_TOML = HERE / "lines.toml"
OUT = REPO / "outputs/dorei_shokan_voices"
JSONL_DIR = HERE / "jsonl"
REFS_DIR = HERE / "refs"  # 採用した参照音声（adopt がコピーする）
BATCH_INFER = REPO / "scripts/batch_infer.py"
BUILD_PHASES = ["start", "mid", "climax"]
TAIL_FADE_MS = 120
TAIL_PAD_OUT_MS = 250
OGG_RATE = 44100
OGG_QUALITY = "5"  # libvorbis -q:a（約 160kbps 相当、声には十分）


# --------------------------------------------------------------------------- 読み込み


def load_presets(path: Path) -> dict:
    with path.open("rb") as f:
        data = tomllib.load(f)
    presets = data.get("preset", [])
    ids = [p["id"] for p in presets]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"presets.toml に重複 id がある: {ids}")
    for p in presets:
        for key in ("id", "name", "caption"):
            if not str(p.get(key, "")).strip():
                raise SystemExit(f"preset {p.get('id')!r}: {key} が空")
    return data


def load_lines(path: Path) -> list[dict]:
    with path.open("rb") as f:
        lines = tomllib.load(f).get("line", [])
    ids = [ln["id"] for ln in lines]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"lines.toml に重複 id がある: {ids}")
    return lines


def pick(items: list[dict], wanted: list[str] | None, what: str) -> list[dict]:
    if not wanted:
        return items
    by_id = {it["id"]: it for it in items}
    missing = [w for w in wanted if w not in by_id]
    if missing:
        raise SystemExit(f"未定義の {what}: {missing}（定義済み: {list(by_id)}）")
    return [by_id[w] for w in wanted]


def caption_for(preset: dict, line: dict) -> str:
    style = str(line.get("style", "")).strip()
    return preset["caption"] if not style else f"{preset['caption']}{style}"


def checkpoint_path(cfg: dict) -> Path:
    ckpt = REPO / cfg["checkpoint"]
    if not ckpt.is_file():
        raise SystemExit(f"チェックポイントが無い: {ckpt}")
    return ckpt


def row(
    *,
    text: str,
    caption: str,
    ckpt: Path,
    out: Path,
    seed: int,
    line: dict,
    ref_wav: Path | None,
    meta: dict,
) -> dict:
    r = {
        "text": text,
        "caption": caption,
        "checkpoint": str(ckpt),
        "out_path": str(out),  # batch_infer は絶対パスが安全
        "seed": int(seed),
        "tail_fade_ms": TAIL_FADE_MS,
        "tail_pad_out_ms": TAIL_PAD_OUT_MS,
        "duration_scale": float(line.get("duration_scale", 1.0)),
    }
    if ref_wav is not None:
        r["ref_wav"] = str(ref_wav)
    r.update(meta)  # batch_infer が読まないメタ情報（page が使う）
    return r


# --------------------------------------------------------------------------- 実行


def write_manifest(rows: list[dict], name: str) -> Path:
    JSONL_DIR.mkdir(parents=True, exist_ok=True)
    dst = JSONL_DIR / f"{name}.jsonl"
    dst.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    return dst


def show_dry_run(manifest: Path, rows: list[dict]) -> None:
    print(f"[dry-run] {len(rows)} 行 -> {manifest}")
    for r in rows[:2]:
        print(json.dumps(r, ensure_ascii=False, indent=2))
    print("[dry-run] GPU は使っていない。実行は --dry-run を外す。")


def run_batch(manifest: Path, timings: Path) -> None:
    cmd = [
        sys.executable,
        str(BATCH_INFER),
        "--manifest",
        str(manifest),
        "--timings-jsonl",
        str(timings),
    ]
    print("[run]", " ".join(cmd), flush=True)
    t0 = time.monotonic()
    rc = subprocess.call(cmd, cwd=REPO)
    print(f"[run] batch_infer 終了 rc={rc}（{time.monotonic() - t0:.0f} 秒）", flush=True)
    if rc != 0:
        raise SystemExit(f"batch_infer が失敗した（rc={rc}）。ログを確認すること。")


def to_ogg(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            str(OGG_RATE),
            "-c:a",
            "libvorbis",
            "-q:a",
            OGG_QUALITY,
            str(dst),
        ],
        check=True,
    )


# --------------------------------------------------------------------------- audition


def cmd_audition(args: argparse.Namespace) -> None:
    cfg = load_presets(args.presets_file)
    lines = load_lines(args.lines_file)
    presets = pick(cfg["preset"], args.presets, "プリセット")
    aud = cfg["audition"]
    seeds = args.seeds or aud["seeds"]
    aud_lines = pick(lines, aud["lines"], "オーディション行")
    ckpt = checkpoint_path(cfg)
    rows = []
    for p in presets:
        for seed in seeds:
            for ln in aud_lines:
                out = OUT / "audition" / p["id"] / f"seed{seed}_{ln['id']}.wav"
                if args.skip_existing and out.is_file():
                    continue
                rows.append(
                    row(
                        text=ln["text"],
                        caption=caption_for(p, ln),
                        ckpt=ckpt,
                        out=out,
                        seed=seed,
                        line=ln,
                        ref_wav=None,
                        meta={"preset": p["id"], "line": ln["id"]},
                    )
                )
    manifest = write_manifest(rows, "audition")
    if args.dry_run:
        show_dry_run(manifest, rows)
        return
    if rows:
        run_batch(manifest, OUT / "audition" / "timings.jsonl")
    else:
        print("[audition] 生成対象なし（全部既にある）")
    page = write_page(cfg, lines)
    print(f"[audition] 試聴ページ: {page}")


# --------------------------------------------------------------------------- page

PAGE_CSS = """
:root{--bg:#16161a;--fg:#e8e6e3;--mut:#9a968f;--card:#222228;--line:#34343c;--acc:#e07a9b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font-family:"Yu Gothic UI","Hiragino Sans",sans-serif;padding:16px}
h1{font-size:20px;margin:0 0 4px}p.note{color:var(--mut);margin:0 0 16px;font-size:13px}
table{border-collapse:collapse;width:100%}th,td{border:1px solid var(--line);vertical-align:top;padding:8px}
th.p{width:260px;text-align:left;background:var(--card)}th.p .n{font-size:17px}
th.p .id{color:var(--acc);font-family:monospace}th.p .c{font-weight:normal;color:var(--mut);font-size:12px;margin-top:6px}
td.cell{min-width:190px}.badge{background:#2f6b3a;color:#fff;border-radius:4px;padding:2px 6px;font-size:13px;font-family:sans-serif}
.rnd{color:var(--mut)}td.cell.adopted{box-shadow:inset 0 0 0 3px #2f6b3a}tr.done th.p{opacity:.75}
td.cell.sel{background:#3a2330;outline:2px solid var(--acc)}
.seed{font-size:30px;font-weight:bold;font-family:monospace;display:flex;align-items:center;gap:8px}
.seed input{width:20px;height:20px}.lab{font-size:11px;color:var(--mut);margin-top:6px}
audio{width:180px;height:32px;display:block}
textarea{width:100%;background:var(--card);color:var(--fg);border:1px solid var(--line);font-size:12px}
#out{position:sticky;top:0;background:var(--bg);padding:8px 0;z-index:2}
#cmd{font-family:monospace;height:110px}
"""

PAGE_JS = """
const KEY='dorei_shokan_audition_v1';
function load(){try{return JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){return {}}}
function save(s){try{localStorage.setItem(KEY,JSON.stringify(s))}catch(e){}}
function refresh(){
  const s=load();const cmds=[];
  document.querySelectorAll('td.cell').forEach(td=>{
    const on=s[td.dataset.p]&&s[td.dataset.p].seed==td.dataset.s;
    td.classList.toggle('sel',!!on);td.querySelector('input').checked=!!on;});
  document.querySelectorAll('textarea.memo').forEach(t=>{t.value=(s[t.dataset.p]||{}).memo||''});
  for(const p of PRESETS){if(s[p]&&s[p].seed)cmds.push(
    'uv run python projects/dorei_shokan_voices/build_voices.py adopt '+p+' '+s[p].seed);}
  document.getElementById('cmd').value=cmds.join('\\n');
}
document.addEventListener('change',e=>{
  const s=load();
  if(e.target.matches('input[type=radio]')){const p=e.target.name;s[p]=s[p]||{};s[p].seed=e.target.value;}
  if(e.target.matches('textarea.memo')){const p=e.target.dataset.p;s[p]=s[p]||{};s[p].memo=e.target.value;}
  save(s);refresh();});
document.addEventListener('play',e=>{document.querySelectorAll('audio').forEach(a=>{if(a!==e.target)a.pause()})},true);
refresh();
"""


def write_page(cfg: dict, lines: list[dict]) -> Path:
    aud = cfg["audition"]
    seeds = aud["seeds"]
    by_id = {ln["id"]: ln for ln in lines}
    root = OUT / "audition"
    root.mkdir(parents=True, exist_ok=True)
    head = "".join(f"<th>seed {s}</th>" for s in seeds)
    body = []
    # 新しいラウンドを上に。同じラウンド内は presets.toml の順
    ordered = sorted(cfg["preset"], key=lambda p: -int(p.get("round", 1)))
    for p in ordered:
        cells = []
        for s in seeds:
            clips = []
            for lid in aud["lines"]:
                rel = f"{p['id']}/seed{s}_{lid}.wav"
                exists = (root / rel).is_file()
                txt = html.escape(by_id[lid]["text"])
                player = (
                    f'<audio controls preload="none" src="{html.escape(rel)}"></audio>'
                    if exists
                    else '<div class="lab">（未生成）</div>'
                )
                clips.append(f'<div class="lab">{html.escape(lid)}：{txt}</div>{player}')
            cells.append(
                f'<td class="cell{" adopted" if int(p.get("seed") or 0) == s else ""}"'
                f' data-p="{p["id"]}" data-s="{s}">'
                f'<label class="seed"><input type="radio" name="{p["id"]}" value="{s}">{s}</label>'
                + "".join(clips)
                + "</td>"
            )
        adopted = f'<span class="badge">採用済み seed {p["seed"]}</span>' if p.get("seed") else ""
        rnd = f'<span class="rnd">R{int(p.get("round", 1))}</span>'
        body.append(
            f'<tr><th class="p"><div class="n">{html.escape(p["name"])}</div>'
            f'<div class="id">{rnd} {p["id"]} {adopted}</div>'
            f'<div class="c">{html.escape(p["caption"])}</div>'
            f'<textarea class="memo" data-p="{p["id"]}" rows="3" placeholder="メモ"></textarea>'
            f"</th>{''.join(cells)}</tr>"
        )
    presets_js = json.dumps([p["id"] for p in cfg["preset"]])
    page = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>奴隷娼館 ボイスオーディション</title><style>{PAGE_CSS}</style></head><body>
<h1>奴隷娼館リメイク ボイスオーディション</h1>
<p class="note">v4.1-Small・caption のみ（参照音声なし）。行＝プリセット、列＝seed。
各セル上が普通の台詞、下が mid の喘ぎ。気に入った seed のラジオを選ぶと下の欄に採用コマンドが出る
（選択とメモはこのブラウザに保存）。</p>
<div id="out"><textarea id="cmd" readonly placeholder="seed を選ぶと adopt コマンドがここに出る"></textarea></div>
<table><tr><th class="p">プリセット</th>{head}</tr>
{"".join(body)}
</table><script>const PRESETS={presets_js};{PAGE_JS}</script></body></html>
"""
    dst = root / "index.html"
    dst.write_text(page, encoding="utf-8")
    return dst


def cmd_page(args: argparse.Namespace) -> None:
    print(write_page(load_presets(args.presets_file), load_lines(args.lines_file)))


# --------------------------------------------------------------------------- adopt


def cmd_adopt(args: argparse.Namespace) -> None:
    cfg = load_presets(args.presets_file)
    pick(cfg["preset"], [args.preset], "プリセット")
    line_id = args.line or cfg["audition"]["lines"][0]
    src = OUT / "audition" / args.preset / f"seed{args.seed}_{line_id}.wav"
    if not src.is_file():
        raise SystemExit(f"オーディション音声が無い: {src}")
    REFS_DIR.mkdir(parents=True, exist_ok=True)
    dst = REFS_DIR / f"{args.preset}.wav"
    shutil.copy2(src, dst)
    rel = dst.relative_to(REPO).as_posix()

    text = args.presets_file.read_text(encoding="utf-8")
    # 該当プリセットのブロック（[[preset]] から次の [[preset]] まで）だけ書き換える
    blocks = re.split(r"(?m)^(?=\[\[preset\]\])", text)
    hit = False
    for i, b in enumerate(blocks):
        if re.search(rf'(?m)^id\s*=\s*"{re.escape(args.preset)}"\s*$', b):
            b = re.sub(r"(?m)^seed\s*=.*$", f"seed = {int(args.seed)}", b, count=1)
            b = re.sub(r"(?m)^ref_wav\s*=.*$", f'ref_wav = "{rel}"', b, count=1)
            blocks[i] = b
            hit = True
    if not hit:
        raise SystemExit(f"presets.toml に {args.preset} のブロックが見つからない")
    args.presets_file.write_text("".join(blocks), encoding="utf-8")
    load_presets(args.presets_file)  # 壊していないか確認
    print(f"[adopt] {args.preset}: seed={args.seed} ref_wav={rel}（{src.name} をコピー）")


# --------------------------------------------------------------------------- build


def cmd_build(args: argparse.Namespace) -> None:
    cfg = load_presets(args.presets_file)
    lines = load_lines(args.lines_file)
    presets = pick(cfg["preset"], args.presets, "プリセット")
    phases = args.phases or BUILD_PHASES
    targets = (
        pick(lines, args.lines, "行")
        if args.lines
        else [ln for ln in lines if ln["phase"] in phases]
    )
    ckpt = checkpoint_path(cfg)
    rows = []
    for p in presets:
        ref = str(p.get("ref_wav", "")).strip()
        if ref:
            ref_path = REPO / ref
            if not ref_path.is_file():
                raise SystemExit(f"{p['id']}: ref_wav が無い: {ref_path}")
        elif args.allow_no_ref:
            ref_path = None
        else:
            raise SystemExit(f"{p['id']}: ref_wav 未決。先に adopt するか --allow-no-ref を付ける")
        seed = int(p.get("seed") or 0) or cfg["audition"]["seeds"][0]
        for ln in targets:
            out = OUT / "voices" / p["id"] / f"{ln['id']}.wav"
            if args.skip_existing and out.is_file():
                continue
            rows.append(
                row(
                    text=ln["text"],
                    caption=caption_for(p, ln),
                    ckpt=ckpt,
                    out=out,
                    seed=seed,
                    line=ln,
                    ref_wav=ref_path,
                    meta={"preset": p["id"], "line": ln["id"], "phase": ln["phase"]},
                )
            )
    manifest = write_manifest(rows, "build")
    if args.dry_run:
        show_dry_run(manifest, rows)
        return
    if rows:
        run_batch(manifest, OUT / "voices" / "timings.jsonl")
    if args.no_ogg:
        return
    n = 0
    for r in rows:
        wav = Path(r["out_path"])
        if wav.is_file():
            to_ogg(wav, OUT / "ogg" / r["preset"] / f"{r['line']}.ogg")
            n += 1
    print(f"[build] ogg {n} 本 -> {OUT / 'ogg'}（{OGG_RATE}Hz/mono/vorbis q{OGG_QUALITY}）")


# --------------------------------------------------------------------------- CLI


def main() -> None:
    ap = argparse.ArgumentParser(description="奴隷娼館リメイク用ボイスプリセットの生成。")
    ap.add_argument("--presets-file", type=Path, default=PRESETS_TOML)
    ap.add_argument("--lines-file", type=Path, default=LINES_TOML)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser(
        "audition", help="caption のみで プリセット × seed × オーディション行 を生成"
    )
    a.add_argument("--presets", nargs="+", help="対象プリセット id（既定: 全部）")
    a.add_argument("--seeds", nargs="+", type=int, help="seed（既定: presets.toml の [audition]）")
    a.add_argument("--skip-existing", action="store_true", help="既にある wav は作らない")
    a.add_argument("--dry-run", action="store_true", help="マニフェストを書いて表示するだけ")
    a.set_defaults(func=cmd_audition)

    pg = sub.add_parser("page", help="試聴ページだけ作り直す")
    pg.set_defaults(func=cmd_page)

    ad = sub.add_parser("adopt", help="オーディションの 1 本を参照音声として固定する")
    ad.add_argument("preset")
    ad.add_argument("seed", type=int)
    ad.add_argument("--line", help="参照にするオーディション行（既定: [audition].lines の先頭）")
    ad.set_defaults(func=cmd_adopt)

    b = sub.add_parser("build", help="ref_wav + caption + seed で lines.toml を一括生成（+ ogg）")
    b.add_argument("--presets", nargs="+", help="対象プリセット id（既定: 全部）")
    b.add_argument(
        "--phases",
        nargs="+",
        choices=["audition", *BUILD_PHASES],
        help="対象段階（既定: start mid climax）",
    )
    b.add_argument("--lines", nargs="+", help="対象の行 id（指定時は --phases より優先）")
    b.add_argument(
        "--allow-no-ref",
        action="store_true",
        help="ref_wav 未決のプリセットを caption のみで作る（動作確認用）",
    )
    b.add_argument("--skip-existing", action="store_true", help="既にある wav は作らない")
    b.add_argument("--no-ogg", action="store_true", help="ogg 変換をしない")
    b.add_argument("--dry-run", action="store_true", help="マニフェストを書いて表示するだけ")
    b.set_defaults(func=cmd_build)

    # 種族×体型の割り当て（voice_matrix.json）。実体は voice_matrix.py
    import voice_matrix

    voice_matrix.add_subcommands(sub)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
