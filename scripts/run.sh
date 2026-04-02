#!/bin/bash
# ══════════════════════════════════════════════════════════
#  絵文字スタイル制御 一括サンプル生成スクリプト
#
#  EMOJI_STYLE_GUIDE.md に基づき、各感情カテゴリの
#  サンプル音声を generate_mamimi.sh で一括生成します。
#
#  使い方:
#    bash scripts/run.sh
# ══════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run() {
    local text="$1"
    local name="$2"
    echo ""
    echo ">>> $name : $text"
    bash "${SCRIPT_DIR}/generate_mamimi.sh" "$text" "$name"
}

echo "========================================================"
echo " 絵文字スタイル サンプル一括生成"
echo "========================================================"

# ── 喜び・明るさ系 ─────────────────────────────────────
run "今日は本当に楽しかったです！😊"         emoji_happy_01
run "やった！うまくいった！😄"               emoji_happy_02
run "ふふっ、もう笑いが止まらない😆"         emoji_happy_03
run "すごい！信じられない！🤩"               emoji_excited_01
run "今夜も星が綺麗ですね😍"                 emoji_dreamy_01
run "ありがとう、大好きだよ🥰"               emoji_love_01
run "さすが、クールだね😎"                   emoji_cool_01

# ── 悲しみ・感情系 ─────────────────────────────────────
run "もうどうしたらいいの…😢"               emoji_sad_01
run "悲しくて、もう涙が出てきた😭"           emoji_sad_02
run "お願い、行かないで🥺"                   emoji_plead_01
run "がっかり、また失敗しちゃった😞"         emoji_down_01
run "なんか今日は気分が沈んでいます😔"       emoji_down_02

# ── 怒り・不満系 ───────────────────────────────────────
run "なんでそんなことするの！😡"             emoji_angry_01
run "もう、ちょっと納得いかない😤"           emoji_sulky_01
run "そのやり方は間違っていると思う😠"       emoji_stern_01

# ── 驚き・緊張系 ───────────────────────────────────────
run "え、本当に？それは驚いた！😲"           emoji_surprise_01
run "怖い、誰かそこにいるの？😱"             emoji_scared_01
run "どうしよう、不安で仕方がない😨"         emoji_anxious_01
run "あ、えっと、少し恥ずかしいな🤭"         emoji_shy_01

# ── 落ち着き・その他 ───────────────────────────────────
run "穏やかな一日でした😌"                   emoji_calm_01
run "うーん、どうしようかなぁ🤔"             emoji_thinking_01
run "眠い、もう寝てもいいですか😴"           emoji_sleepy_01
run "ふふ、秘密にしておきましょう😏"         emoji_smirk_01
run "はぁ、もうあきれた😙"                   emoji_sigh_01

echo ""
echo "========================================================"
echo " 全サンプル生成完了"
echo "========================================================"
echo "出力先: outputs/mamimi_samples/"
