"""種族×体型の声割り当て（voice_matrix.json）のオーディション・採用・一括生成・設定サーバ。

``build_voices.py`` の matrix-* サブコマンドの実体。真実源は ``voice_matrix.json``::

    cells["<race>:<body>"] = {"mode": "compose" | "preset", "preset": "<presets.toml の id>",
                              "seed": 0, "ref_wav": ""}

- compose: caption = bodies[].voice（年齢感）+ races[].tone（口調）
- preset : caption = presets.toml の該当プリセットの caption
- seed / ref_wav は採用後に固定（0 / "" = 未採用）。台本行の style は caption の後ろに足す。

出力:
    outputs/dorei_shokan_voices/matrix/<race>_<body>/seed<N>_<line>.wav   （オーディション）
    outputs/dorei_shokan_voices/matrix_voices/<race>_<body>/<line>.wav    （本生成 48kHz）
    <Godot>/game/assets/voices/<race>/<body>/<line>.ogg + voices_index.json
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import shutil
import socket
import subprocess
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import build_voices as bv

MATRIX_JSON = bv.HERE / "voice_matrix.json"
PAGE_HTML = bv.HERE / "matrix_page.html"
MATRIX_OUT = bv.OUT / "matrix"
MATRIX_VOICES = bv.OUT / "matrix_voices"
MATRIX_OGG = bv.OUT / "matrix_ogg"
REFS_MATRIX = bv.REFS_DIR / "matrix"
GODOT_VOICES = Path("/mnt/c/Users/mao0202/Desktop/Godot/projects/dorei-shokan/game/assets/voices")
GODOT_RES_PREFIX = "res://assets/voices"
BRAVE = Path("/mnt/c/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe")
MODES = ("compose", "preset")

# --------------------------------------------------------------------------- データ


def load_matrix(path: Path = MATRIX_JSON) -> dict:
    m = json.loads(path.read_text(encoding="utf-8"))
    validate(m)
    return m


def save_matrix(m: dict, path: Path = MATRIX_JSON) -> None:
    validate(m)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def validate(m: dict) -> None:
    races = [r["id"] for r in m["races"]]
    bodies = [b["id"] for b in m["bodies"]]
    want = {f"{r}:{b}" for r in races for b in bodies}
    have = set(m["cells"])
    if want != have:
        raise SystemExit(
            f"cells のキーが races×bodies と合わない: 不足={want - have} 余分={have - want}"
        )
    for k, c in m["cells"].items():
        if c.get("mode") not in MODES:
            raise SystemExit(f"{k}: mode は {MODES} のどれか（{c.get('mode')!r}）")
        if c["mode"] == "preset" and not c.get("preset"):
            raise SystemExit(f"{k}: mode=preset なのに preset が空")


def cell_dir(key: str) -> str:
    return key.replace(":", "_")


def presets_by_id(cfg: dict) -> dict:
    return {p["id"]: p for p in cfg["preset"]}


def cell_caption(m: dict, key: str, presets: dict) -> str:
    c = m["cells"][key]
    if c["mode"] == "preset":
        if c["preset"] not in presets:
            raise SystemExit(f"{key}: presets.toml に無いプリセット {c['preset']!r}")
        return presets[c["preset"]]["caption"]
    race, body = key.split(":")
    r = next(x for x in m["races"] if x["id"] == race)
    b = next(x for x in m["bodies"] if x["id"] == body)
    return f"{b['voice']}{r['tone']}"


def select_cells(m: dict, spec: str | None) -> list[str]:
    """`inu:kogara,neko:*,*:glamour` 形式（* は全部）。None なら全セル。"""
    keys = list(m["cells"])
    if not spec:
        return keys
    races = [r["id"] for r in m["races"]]
    bodies = [b["id"] for b in m["bodies"]]
    out: list[str] = []
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        if ":" not in tok:
            raise SystemExit(f"--cells は race:body 形式: {tok!r}")
        r, b = tok.split(":", 1)
        rs = races if r == "*" else [r]
        bs = bodies if b == "*" else [b]
        for rr in rs:
            for bb in bs:
                k = f"{rr}:{bb}"
                if k not in m["cells"]:
                    raise SystemExit(f"未定義のセル: {k}（races={races} bodies={bodies}）")
                if k not in out:
                    out.append(k)
    return out


def parse_seeds(raw: str | None, m: dict) -> list[int]:
    if not raw:
        return list(m["audition_seeds"])
    return [int(s) for s in raw.replace(" ", "").split(",") if s]


# --------------------------------------------------------------------------- matrix-audition


def cmd_matrix_audition(args: argparse.Namespace) -> None:
    m = load_matrix()
    cfg = bv.load_presets(args.presets_file)
    presets = presets_by_id(cfg)
    lines = bv.load_lines(args.lines_file)
    aud_lines = bv.pick(lines, cfg["audition"]["lines"], "オーディション行")
    seeds = parse_seeds(args.seeds, m)
    ckpt = bv.checkpoint_path(cfg)
    rows, skipped = [], []
    for key in select_cells(m, args.cells):
        c = m["cells"][key]
        if c["mode"] != "compose" or int(c.get("seed") or 0):
            # preset セルはプリセットのオーディション（audition/）を聞く。採用済みは作らない
            skipped.append(key)
            continue
        cap = cell_caption(m, key, presets)
        for seed in seeds:
            for ln in aud_lines:
                out = MATRIX_OUT / cell_dir(key) / f"seed{seed}_{ln['id']}.wav"
                if args.skip_existing and out.is_file():
                    continue
                rows.append(
                    bv.row(
                        text=ln["text"],
                        caption=bv.caption_for({"caption": cap}, ln),
                        ckpt=ckpt,
                        out=out,
                        seed=seed,
                        line=ln,
                        ref_wav=None,
                        meta={"cell": key, "line": ln["id"]},
                    )
                )
    if skipped:
        print(f"[matrix-audition] 対象外（preset セル or 採用済み）: {', '.join(skipped)}")
    manifest = bv.write_manifest(rows, "matrix_audition")
    if args.dry_run:
        bv.show_dry_run(manifest, rows)
        return
    if rows:
        bv.run_batch(manifest, MATRIX_OUT / "timings.jsonl")
    else:
        print("[matrix-audition] 生成対象なし")


# --------------------------------------------------------------------------- matrix-adopt


def adopt_cell(m: dict, key: str, seed: int, cfg: dict) -> str:
    """セルの seed を採用し、talk を refs/matrix/ にコピーして m を更新する（保存はしない）。"""
    c = m["cells"][key]
    talk = cfg["audition"]["lines"][0]
    if c["mode"] == "preset":
        src = bv.OUT / "audition" / c["preset"] / f"seed{seed}_{talk}.wav"
    else:
        src = MATRIX_OUT / cell_dir(key) / f"seed{seed}_{talk}.wav"
    if not src.is_file():
        raise SystemExit(f"{key}: オーディション音声が無い: {src}")
    REFS_MATRIX.mkdir(parents=True, exist_ok=True)
    dst = REFS_MATRIX / f"{cell_dir(key)}.wav"
    shutil.copy2(src, dst)
    c["seed"] = int(seed)
    c["ref_wav"] = dst.relative_to(bv.REPO).as_posix()
    return f"{key}: seed={seed} ref_wav={c['ref_wav']}（{src.relative_to(bv.OUT)} をコピー）"


def cmd_matrix_adopt(args: argparse.Namespace) -> None:
    m = load_matrix()
    cfg = bv.load_presets(args.presets_file)
    key = select_cells(m, args.cell)
    if len(key) != 1:
        raise SystemExit("matrix-adopt はセルを 1 つだけ指定する（race:body）")
    msg = adopt_cell(m, key[0], args.seed, cfg)
    save_matrix(m)
    print(f"[matrix-adopt] {msg}")


# --------------------------------------------------------------------------- matrix-build


def write_godot_index(godot_dir: Path, lines: list[dict]) -> Path:
    """godot_dir 配下の <race>/<body>/<line>.ogg を走査して voices_index.json を書く。"""
    phase_of = {ln["id"]: ln["phase"] for ln in lines}
    kind_of = {ln["id"]: ln.get("kind", "") for ln in lines}
    index: dict = {}
    for ogg in sorted(godot_dir.glob("*/*/*.ogg")):
        body_dir = ogg.parent
        race, body, line = body_dir.parent.name, body_dir.name, ogg.stem
        phase = phase_of.get(line, "other")
        rel = ogg.relative_to(godot_dir).as_posix()
        index.setdefault(race, {}).setdefault(body, {}).setdefault(phase, {})[line] = {
            "path": f"{GODOT_RES_PREFIX}/{rel}",
            "kind": kind_of.get(line, ""),
        }
    dst = godot_dir / "voices_index.json"
    dst.write_text(
        json.dumps(
            {"format": "race > body > phase > line", "voices": index}, ensure_ascii=False, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    return dst


def cmd_matrix_build(args: argparse.Namespace) -> None:
    m = load_matrix()
    cfg = bv.load_presets(args.presets_file)
    presets = presets_by_id(cfg)
    lines = bv.load_lines(args.lines_file)
    phases = args.phases or bv.BUILD_PHASES
    targets = (
        bv.pick(lines, args.lines, "行")
        if args.lines
        else [ln for ln in lines if ln["phase"] in phases]
    )
    ckpt = bv.checkpoint_path(cfg)
    godot_dir = None if (args.no_godot or args.no_ogg) else Path(args.godot_dir)
    rows, pending, kept = [], [], []
    for key in select_cells(m, args.cells):
        c = m["cells"][key]
        ref = str(c.get("ref_wav") or "").strip()
        if not (int(c.get("seed") or 0) and ref):
            pending.append(key)
            continue
        ref_path = bv.REPO / ref
        if not ref_path.is_file():
            raise SystemExit(f"{key}: ref_wav が無い: {ref_path}")
        cap = cell_caption(m, key, presets)
        for ln in targets:
            out = MATRIX_VOICES / cell_dir(key) / f"{ln['id']}.wav"
            if args.skip_existing and out.is_file():
                continue
            race, body = key.split(":")
            if godot_dir is not None and not args.force:
                # Godot 側に同名があれば手修正を守って作らない（--force で上書き）
                if (godot_dir / race / body / f"{ln['id']}.ogg").is_file():
                    kept.append(f"{race}/{body}/{ln['id']}.ogg")
                    continue
            rows.append(
                bv.row(
                    text=ln["text"],
                    caption=bv.caption_for({"caption": cap}, ln),
                    ckpt=ckpt,
                    out=out,
                    seed=int(c["seed"]),
                    line=ln,
                    ref_wav=ref_path,
                    meta={"cell": key, "line": ln["id"], "phase": ln["phase"]},
                )
            )
    if pending:
        print(f"[matrix-build] 未採用のため対象外: {', '.join(pending)}")
    if kept:
        print(f"[matrix-build] Godot に既存のため作らない（--force で上書き）: {len(kept)} 本")
    manifest = bv.write_manifest(rows, "matrix_build")
    if args.dry_run:
        bv.show_dry_run(manifest, rows)
        return
    if rows:
        bv.run_batch(manifest, MATRIX_VOICES / "timings.jsonl")
    if args.no_ogg:
        return
    n = 0
    for r in rows:
        wav = Path(r["out_path"])
        if not wav.is_file():
            continue
        race, body = r["cell"].split(":")
        ogg = MATRIX_OGG / race / body / f"{r['line']}.ogg"
        bv.to_ogg(wav, ogg)
        if godot_dir is not None:
            dst = godot_dir / race / body / ogg.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ogg, dst)
        n += 1
    print(f"[matrix-build] ogg {n} 本 -> {MATRIX_OGG}")
    if godot_dir is not None:
        print(f"[matrix-build] Godot へ書き出し + 索引: {write_godot_index(godot_dir, lines)}")


# --------------------------------------------------------------------------- matrix-serve


def _exists_map(base: Path, names: list[str], seeds: list[int], line_ids: list[str]) -> dict:
    out = {}
    for name in names:
        d = base / name
        out[name] = {
            str(s): [lid for lid in line_ids if (d / f"seed{s}_{lid}.wav").is_file()] for s in seeds
        }
    return out


def build_state(args: argparse.Namespace) -> dict:
    m = load_matrix()
    cfg = bv.load_presets(args.presets_file)
    presets = presets_by_id(cfg)
    lines = {ln["id"]: ln for ln in bv.load_lines(args.lines_file)}
    aud_ids = cfg["audition"]["lines"]
    captions = {}
    for k in m["cells"]:
        try:
            captions[k] = cell_caption(m, k, presets)
        except SystemExit as e:
            captions[k] = f"(エラー) {e}"
    return {
        "matrix": m,
        "captions": captions,
        "presets": [
            {k: p.get(k) for k in ("id", "name", "caption", "seed", "ref_wav", "round")}
            for p in cfg["preset"]
        ],
        "preset_seeds": cfg["audition"]["seeds"],
        "audition_lines": [{"id": i, "text": lines[i]["text"]} for i in aud_ids],
        "cell_audio": _exists_map(
            MATRIX_OUT, [cell_dir(k) for k in m["cells"]], m["audition_seeds"], aud_ids
        ),
        "preset_audio": _exists_map(
            bv.OUT / "audition", list(presets), cfg["audition"]["seeds"], aud_ids
        ),
    }


def apply_save(payload: dict, args: argparse.Namespace) -> list[str]:
    """ページからの保存。races/bodies の断片・セルの mode/preset を書き、adopt を実行する。"""
    m = load_matrix()
    cfg = bv.load_presets(args.presets_file)
    presets = presets_by_id(cfg)
    log = []
    tone = {r["id"]: r.get("tone", "") for r in payload.get("races", [])}
    for r in m["races"]:
        if r["id"] in tone and tone[r["id"]].strip():
            r["tone"] = tone[r["id"]].strip()
    voice = {b["id"]: b.get("voice", "") for b in payload.get("bodies", [])}
    for b in m["bodies"]:
        if b["id"] in voice and voice[b["id"]].strip():
            b["voice"] = voice[b["id"]].strip()
    for key, nc in payload.get("cells", {}).items():
        if key not in m["cells"]:
            continue
        c = m["cells"][key]
        mode, preset = nc.get("mode", c["mode"]), nc.get("preset", c.get("preset", ""))
        if mode not in MODES:
            continue
        if mode == "compose":
            preset = ""
        elif preset not in presets:
            continue
        if (mode, preset) != (c["mode"], c.get("preset", "")):
            c["mode"], c["preset"] = mode, preset
            p = presets.get(preset) if mode == "preset" else None
            if p and int(p.get("seed") or 0) and p.get("ref_wav"):
                # 採用済みプリセットを割り当てたら、その seed と参照音声を引き継ぐ
                c["seed"], c["ref_wav"] = int(p["seed"]), p["ref_wav"]
            else:
                c["seed"], c["ref_wav"] = 0, ""
            log.append(f"{key}: {mode}{' ' + preset if preset else ''}（seed={c['seed']}）")
    for key, seed in (payload.get("adopt") or {}).items():
        if key in m["cells"] and seed:
            try:
                log.append(adopt_cell(m, key, int(seed), cfg))
            except SystemExit as e:
                log.append(f"{key}: 採用失敗 {e}")
    save_matrix(m)
    return log or ["変更なし（保存済み）"]


def make_handler(args: argparse.Namespace):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):  # noqa: D401 - 静かにする
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json")

        def do_GET(self):  # noqa: N802
            path = urllib.parse.unquote(urllib.parse.urlparse(self.path).path)
            if path in ("/", "/index.html"):
                self._send(200, PAGE_HTML.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/state":
                self._json(build_state(args))
            elif path.startswith("/audio/"):
                f = (bv.OUT / path[len("/audio/") :]).resolve()
                if bv.OUT.resolve() not in f.parents or not f.is_file():
                    self._send(404, b"not found", "text/plain")
                    return
                ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
                self._send(200, f.read_bytes(), ctype)
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):  # noqa: N802
            if urllib.parse.urlparse(self.path).path != "/api/save":
                self._send(404, b"not found", "text/plain")
                return
            n = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
                log = apply_save(payload, args)
                self._json({"ok": True, "log": log, "state": build_state(args)})
            except (SystemExit, ValueError, KeyError) as e:
                self._json({"ok": False, "error": str(e)}, 400)

    return Handler


def free_port(preferred: int) -> int:
    for port in (preferred, 0):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise SystemExit("空きポートが無い")


def cmd_matrix_serve(args: argparse.Namespace) -> None:
    load_matrix()  # 壊れていたら起動前に止める
    port = free_port(args.port)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(args))
    url = f"http://localhost:{port}/"
    print(f"[matrix-serve] {url}（Ctrl+C で終了）", flush=True)
    if not args.no_open and BRAVE.exists():
        threading.Timer(
            0.5, lambda: subprocess.Popen([str(BRAVE), url], stdout=subprocess.DEVNULL)
        ).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


# --------------------------------------------------------------------------- CLI


def add_subcommands(sub) -> None:
    a = sub.add_parser("matrix-audition", help="未採用 compose セルを seed 候補×2行で生成")
    a.add_argument("--cells", help="race:body をカンマ区切り（* 可。例 inu:*,*:kogara）")
    a.add_argument("--seeds", help="カンマ区切り（既定: voice_matrix.json の audition_seeds）")
    a.add_argument("--skip-existing", action="store_true")
    a.add_argument("--dry-run", action="store_true")
    a.set_defaults(func=cmd_matrix_audition)

    ad = sub.add_parser("matrix-adopt", help="セルの seed を採用し参照音声を固定")
    ad.add_argument("cell", help="race:body")
    ad.add_argument("seed", type=int)
    ad.set_defaults(func=cmd_matrix_adopt)

    b = sub.add_parser("matrix-build", help="採用済みセルの全行を生成 → ogg → Godot へ書き出し")
    b.add_argument("--cells", help="race:body をカンマ区切り（* 可）。既定: 全セル")
    b.add_argument("--phases", nargs="+", choices=["audition", *bv.BUILD_PHASES])
    b.add_argument("--lines", nargs="+")
    b.add_argument("--godot-dir", default=str(GODOT_VOICES))
    b.add_argument("--no-godot", action="store_true", help="Godot へ書き出さない")
    b.add_argument(
        "--force", action="store_true", help="Godot 側に同名 ogg があっても作り直して上書きする"
    )
    b.add_argument("--no-ogg", action="store_true")
    b.add_argument("--skip-existing", action="store_true")
    b.add_argument("--dry-run", action="store_true")
    b.set_defaults(func=cmd_matrix_build)

    s = sub.add_parser("matrix-serve", help="設定ページのローカルサーバを起動し Brave で開く")
    s.add_argument("--port", type=int, default=8766, help="使用中なら空きポートを自動で選ぶ")
    s.add_argument("--no-open", action="store_true", help="Brave を開かない")
    s.set_defaults(func=cmd_matrix_serve)
