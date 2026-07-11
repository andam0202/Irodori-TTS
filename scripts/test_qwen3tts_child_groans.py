"""Qwen3-TTS 子供ボイス「非性的うめき声・非言語音」多言語生成（EN/RU/ZH/KO/ES/PT）.

ゲーム制作（ホラー/戦闘/ストーリー/病弱・衰弱シーン）用。8歳女児の **非身体的・非性的** な
音声を、**言語別の声質特徴** を付けて生成する:
  - RU / ZH : 低い声（low, deep, husky な子供）
  - ES      : ダウンナー系（無気力・陰気・flat）
  - EN/KO/PT: 標準

9タイプ（いずれも NON-sexual）:
  fear / pain / grief / exhaustion / dread / cold   — セリフ＋擬音
  cough / wordless_groan / slow_sad                 — 台詞でない・声にならない純粋音（擬音のみ）

bash scripts/qwen3tts.sh python scripts/test_qwen3tts_child_groans.py
bash scripts/qwen3tts.sh python scripts/test_qwen3tts_child_groans.py --languages Spanish --types cough
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _tts_testgen as testgen  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "tools" / "qwen3-tts" / "models" / "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
OUTPUT_DIR = PROJECT_DIR / "data" / "output" / "qwen3tts_child_groans"

LANG_CODE = {"English": "en", "Russian": "ru", "Chinese": "zh", "Korean": "ko",
             "Spanish": "es", "Portuguese": "pt"}

# 言語別の声質ベース（特徴付け）
VOICE_BASE: dict[str, str] = {
    "English": "A little 8-year-old girl",
    "Russian": "A little 8-year-old girl with a noticeably low, deep, husky voice for a child",
    "Chinese": "A little 8-year-old girl with a low, deep voice for a child",
    "Korean": "A little 8-year-old girl",
    "Spanish": "A little 8-year-old girl with a downer, listless, low-energy, flat and gloomy personality",
    "Portuguese": "A little 8-year-old girl",
}

# タイプ別の感情記述（声質ベースに連結して使う）
TYPE_DESC: dict[str, str] = {
    "fear": (
        "whimpering in pure fear, trembling and cowering, a shaky voice catching her breath, "
        "terrified of something. This is NON-sexual: just a frightened child."
    ),
    "pain": (
        "crying out softly from a hurt arm or scraped knee, a pained whimper, sniffling, "
        "small trembling voice in ordinary physical pain. This is NON-sexual."
    ),
    "grief": (
        "sobbing quietly in sadness, a broken crying voice, sniffling and hiccuping through "
        "tears, heartbroken. This is NON-sexual."
    ),
    "exhaustion": (
        "completely out of breath after running hard, panting from pure physical exertion, "
        "a weak tired voice. This is NON-sexual."
    ),
    "dread": (
        "with a sinking anxious feeling, a low uneasy groan of worry, nervous and afraid. "
        "This is NON-sexual."
    ),
    "cold": (
        "shivering hard in the freezing cold, teeth chattering, a shaky weak voice from the "
        "chill. This is NON-sexual."
    ),
    # --- 台詞でない・声にならない純粋音 ---
    "cough": (
        "coughing repeatedly, a sick child's dry weak cough, persistent and unwell, chest "
        "rattling. NON-sexual, non-verbal sound only, no words."
    ),
    "wordless_groan": (
        "letting out wordless groans of discomfort and pain, no articulated words at all, "
        "just low pained sounds from the throat like a child who feels sick or hurt. "
        "NON-sexual, non-verbal sound only."
    ),
    "slow_sad": (
        "making slow long sad sounds, deep sighing breaths and soft whimpering with no words, "
        "deeply sorrowful and heavy. NON-sexual, non-verbal sound only."
    ),
}

# 言語ごとのライン: (タイプ, テキスト)
LINES: dict[str, list[tuple[str, str]]] = {
    "English": [
        ("fear", "N-no... stay back... please don't come any closer... I'm scared..."),
        ("pain", "Ow... ow, my arm... it hurts so much... ow..."),
        ("grief", "Uu... why did it have to end up like this... uu... I'm so sad..."),
        ("exhaustion", "Haa... haa... I can't run anymore... I'm so tired... haa..."),
        ("dread", "Something's not right... I've got a really bad feeling about this..."),
        ("cold", "Brr... it's so cold... I can't stop shivering... brr..."),
        ("cough", "Cough... cough, cough... uhh... cough..."),
        ("wordless_groan", "Ngh... ugh... hnnn... urr... ngh..."),
        ("slow_sad", "Haaah... huu... uuh... haaah... huu..."),
    ],
    "Russian": [
        ("fear", "Н-нет... не подходи... пожалуйста, не приближайся... мне страшно..."),
        ("pain", "Ой... ой, моя рука... так сильно болит... ой..."),
        ("grief", "Уу... за что всё так обернулось... уу... мне так грустно..."),
        ("exhaustion", "Ха... ха... больше не могу бежать... я так устала... ха..."),
        ("dread", "Что-то не так... у меня очень нехорошее предчувствие..."),
        ("cold", "Бр... так холодно... у меня никак не проходит дрожь... бр..."),
        ("cough", "Кх... кх-кх... эх... кх..."),
        ("wordless_groan", "Нгх... угх... хннн... урр... нгх..."),
        ("slow_sad", "Хааа... хуу... уух... хааа... хуу..."),
    ],
    "Chinese": [
        ("fear", "不……不要过来……求求你别再靠近了……我好害怕……"),
        ("pain", "哎哟……哎哟，我的手……好痛好痛……哎哟……"),
        ("grief", "呜……为什么会变成这样……呜……我好难过……"),
        ("exhaustion", "呼……呼……我真的跑不动了……好累……呼……"),
        ("dread", "不对劲……我有种很不好的预感……"),
        ("cold", "好冷好冷……我一直在发抖，停不下来……"),
        ("cough", "咳……咳咳……唔……咳……"),
        ("wordless_groan", "嗯……唔……哼……呃……嗯……"),
        ("slow_sad", "呼……呜……唔……呼……呜……"),
    ],
    "Korean": [
        ("fear", "아... 않... 가지 마... 제발 더 다가오지 마... 무서워..."),
        ("pain", "아야... 아야, 내 팔... 너무 많이 아파... 아야..."),
        ("grief", "흑... 왜 이렇게 됐을까... 흑... 너무 슬퍼..."),
        ("exhaustion", "하아... 하아... 더는 못 뛰겠어... 너무 힘들어... 하아..."),
        ("dread", "이상해... 너무 안 좋은 예감이 들어..."),
        ("cold", "으... 너무 추워... 계속 떨려서 멈추질 않아..."),
        ("cough", "콜록... 콜록 콜록... 으... 콜록..."),
        ("wordless_groan", "읏... 윽... 흣... 으... 읏..."),
        ("slow_sad", "하아아... 후우... 으응... 하아아... 후우..."),
    ],
    "Spanish": [
        ("fear", "N-no... no te acerques... por favor... tengo miedo..."),
        ("pain", "Ay... ay, mi brazo... me duele mucho... ay..."),
        ("grief", "Uu... ¿por qué tuvo que ser así...? uu... estoy tan triste..."),
        ("exhaustion", "Ha... ha... no puedo correr más... estoy tan cansada... ha..."),
        ("dread", "Algo no está bien... tengo un muy mal presentimiento..."),
        ("cold", "Brr... hace tanto frío... no puedo dejar de tiritar... brr..."),
        ("cough", "Tos... tos, tos... uh... tos..."),
        ("wordless_groan", "Ngh... ugh... hnnn... urr... ngh..."),
        ("slow_sad", "Haaah... huu... uuh... haaah... huu..."),
    ],
    "Portuguese": [
        ("fear", "N-não... não se aproxime... por favor... estou com medo..."),
        ("pain", "Ai... ai, meu braço... dói muito... ai..."),
        ("grief", "Uu... por que teve que ser assim... uu... estou tão triste..."),
        ("exhaustion", "Ha... ha... não consigo correr mais... estou tão cansada... ha..."),
        ("dread", "Algo não está certo... tenho um mau pressentimento..."),
        ("cold", "Brr... está tão frio... não consigo parar de tremer... brr..."),
        ("cough", "Toss... toss, toss... uh... toss..."),
        ("wordless_groan", "Ngh... ugh... hnnn... urr... ngh..."),
        ("slow_sad", "Haaah... huu... uuh... haaah... huu..."),
    ],
}


def main() -> None:
    parser = testgen.make_parser(
        "Qwen3-TTS child (8yo) NON-sexual groan/non-verbal multilingual generation",
        output_dir=OUTPUT_DIR,
        model_dir=MODEL_DIR,
        languages=list(LINES.keys()),
        types=list(TYPE_DESC.keys()),
        seeds=[0, 1],
        engine="qwen",
    )
    args = parser.parse_args()
    specs = testgen.build_specs(
        LINES,
        LANG_CODE,
        lambda language, typ: f"{VOICE_BASE[language]}, {TYPE_DESC[typ]}",
        languages=args.languages,
        types=args.types,
        seeds=args.seeds,
    )
    testgen.run_qwen_design(
        specs,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        temperature=args.temperature,
        top_p=args.top_p,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
