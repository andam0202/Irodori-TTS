uv run python infer.py --hf-checkpoint Aratako/Irodori-TTS-500M-v2-VoiceDesign --text "今日のライブ、本当に楽しかった！" --caption "😊🌸 春の花のように明るくやわらかく話す女性の声で" --no-ref --output-wav outputs/emoji_test.wav


bash scripts/generate_mamimi.sh "おはよう！" morning.wav