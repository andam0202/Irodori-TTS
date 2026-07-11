"""pytest 共通設定: リポジトリルートを import パスに追加する.

irodori_tts はリポジトリルート直下のソースパッケージなので、
インストール状態に依存せずテストできるようにする。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
