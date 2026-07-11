"""scripts/_tts_testgen.py の回帰テスト（GPU・実モデル不要）.

フェイクエンジンを sys.modules に注入し、生成ループの呼び出し順・
出力ファイル名・manifest 内容を検証する。
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import _tts_testgen as testgen  # noqa: E402

LINES = {
    "English": [("moan", "Ahh..."), ("whisper", "Come closer...")],
    "Korean": [("moan", "아앙..."), ("whisper", "이리 와...")],
}
LANG_CODE = {"English": "en", "Korean": "ko"}
INSTRUCTS = {"moan": "moaning voice", "whisper": "whispering voice"}


def test_build_specs_expansion_order() -> None:
    specs = testgen.build_specs(
        LINES,
        LANG_CODE,
        lambda language, typ: INSTRUCTS[typ],
        languages=["English", "Korean"],
        types=["moan", "whisper"],
        seeds=[0, 1],
    )
    assert [s.filename for s in specs] == [
        "en_moan_s0.wav", "en_moan_s1.wav",
        "en_whisper_s0.wav", "en_whisper_s1.wav",
        "ko_moan_s0.wav", "ko_moan_s1.wav",
        "ko_whisper_s0.wav", "ko_whisper_s1.wav",
    ]
    assert specs[0].instruct == "moaning voice"
    assert specs[0].log == "Ahh......"  # text[:40] + "..."


def test_build_specs_filters_types_and_languages() -> None:
    specs = testgen.build_specs(
        LINES,
        LANG_CODE,
        lambda language, typ: INSTRUCTS[typ],
        languages=["Korean"],
        types=["whisper"],
        seeds=[7],
    )
    assert [s.filename for s in specs] == ["ko_whisper_s7.wav"]
    assert specs[0].seed == 7


def _install_fake_engines(monkeypatch, calls: list) -> None:
    torch_mod = types.ModuleType("torch")
    torch_mod.bfloat16 = "bf16"
    torch_mod.manual_seed = lambda seed: calls.append(("seed", seed))

    sf_mod = types.ModuleType("soundfile")
    sf_mod.write = lambda path, wav, sr: calls.append(("write", str(path), sr))

    qwen_mod = types.ModuleType("qwen_tts")

    class Qwen3TTSModel:
        @classmethod
        def from_pretrained(cls, model_dir, **kwargs):
            calls.append(("load", str(model_dir)))
            return cls()

        def generate_voice_design(self, **kwargs):
            calls.append(("gen", kwargs["text"], kwargs["instruct"]))
            return (["WAV"], 24000)

    qwen_mod.Qwen3TTSModel = Qwen3TTSModel

    voxcpm_mod = types.ModuleType("voxcpm")

    class VoxCPM:
        tts_model = types.SimpleNamespace(sample_rate=16000)

        @classmethod
        def from_pretrained(cls, model_dir, **kwargs):
            calls.append(("load", str(model_dir)))
            return cls()

        def generate(self, **kwargs):
            calls.append(("gen", kwargs["text"]))
            return "WAV"

    voxcpm_mod.VoxCPM = VoxCPM

    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "soundfile", sf_mod)
    monkeypatch.setitem(sys.modules, "qwen_tts", qwen_mod)
    monkeypatch.setitem(sys.modules, "voxcpm", voxcpm_mod)


def test_run_qwen_design_calls_and_manifest(monkeypatch, tmp_path) -> None:
    calls: list = []
    _install_fake_engines(monkeypatch, calls)
    specs = testgen.build_specs(
        LINES,
        LANG_CODE,
        lambda language, typ: INSTRUCTS[typ],
        languages=["English"],
        types=["moan"],
        seeds=[0, 1],
    )
    testgen.run_qwen_design(
        specs, model_dir="m", output_dir=str(tmp_path), temperature=0.9, top_p=0.9
    )
    # seed 発行 → 生成 → 書き込みの順で spec ごとに呼ばれる
    assert calls == [
        ("load", "m"),
        ("seed", 0), ("gen", "Ahh...", "moaning voice"),
        ("write", str(tmp_path / "en_moan_s0.wav"), 24000),
        ("seed", 1), ("gen", "Ahh...", "moaning voice"),
        ("write", str(tmp_path / "en_moan_s1.wav"), 24000),
    ]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest[0] == {
        "file": "en_moan_s0.wav", "language": "English", "type": "moan",
        "seed": 0, "text": "Ahh...", "instruct": "moaning voice", "sr": 24000,
    }
    # 現行スクリプトと同じキー順を維持する
    assert list(manifest[0].keys()) == ["file", "language", "type", "seed", "text", "instruct", "sr"]


def test_run_qwen_design_persona_fields(monkeypatch, tmp_path) -> None:
    calls: list = []
    _install_fake_engines(monkeypatch, calls)
    specs = [
        testgen.GenSpec(
            filename="en_genki_s0.wav", language="English", type_key="genki",
            text="Hi!", instruct="cheerful", seed=0, log="genki",
        )
    ]
    testgen.run_qwen_design(
        specs, model_dir="m", output_dir=str(tmp_path), temperature=0.9, top_p=0.9,
        type_field="persona", manifest_name="manifest_en.json",
    )
    manifest = json.loads((tmp_path / "manifest_en.json").read_text())
    assert manifest[0]["persona"] == "genki"
    assert "type" not in manifest[0]


def test_run_voxcpm_embeds_description(monkeypatch, tmp_path) -> None:
    calls: list = []
    _install_fake_engines(monkeypatch, calls)
    specs = [
        testgen.GenSpec(
            filename="sw_fear_s0.wav", language="Swahili", type_key="fear",
            text="Hapana...", instruct="a scared child", seed=0, log="Hapana......",
        )
    ]
    testgen.run_voxcpm(specs, model_dir="m", output_dir=str(tmp_path), cfg=2.0, timesteps=10)
    assert ("gen", "(a scared child)Hapana...") in calls
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest[0]["description"] == "a scared child"
    assert manifest[0]["sr"] == 16000
    assert list(manifest[0].keys()) == [
        "file", "language", "type", "seed", "text", "description", "sr",
    ]


def test_dry_run_skips_engine_imports(capsys, tmp_path) -> None:
    # フェイク未注入でも dry_run なら import されず動く
    specs = [
        testgen.GenSpec(
            filename="x.wav", language="English", type_key="moan",
            text="Ahh...", instruct="v", seed=0, log="Ahh......",
        )
    ]
    testgen.run_qwen_design(
        specs, model_dir="m", output_dir=str(tmp_path / "none"), temperature=0.9, top_p=0.9,
        dry_run=True,
    )
    out = capsys.readouterr().out
    assert "total: 1 files" in out
    assert not (tmp_path / "none").exists()
