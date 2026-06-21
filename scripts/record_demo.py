"""M6: 要求 5 PAD の MP4 クリップを録画する。

同一パス・同一カメラで [0,0,0],[1,1,1],[-1,-1,-1],[-1,1,-1],[-1,1,1] を出力。
ラベル(ロボット名/タスク/PAD/冗長性)を焼き込む。M7 でこれらを連結する。

使い方:
    uv run python scripts/record_demo.py            # 5 本すべて
    uv run python scripts/record_demo.py 1 1 1      # 単発 PAD を 1 本だけ
"""
from __future__ import annotations

import os
import sys

from pepper_pad.record import record_clip

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

PADS = [
    ((0, 0, 0), "pad_000_baseline.mp4"),
    ((1, 1, 1), "pad_111_joy.mp4"),
    ((-1, -1, -1), "pad_nnn_sad.mp4"),
    ((-1, 1, -1), "pad_n1n_fear.mp4"),
    ((-1, 1, 1), "pad_n11_anger.mp4"),
]


def main() -> None:
    if len(sys.argv) == 4:
        pad = tuple(int(v) for v in sys.argv[1:4])
        name = f"pad_{pad[0]}{pad[1]}{pad[2]}.mp4".replace("-", "n")
        print(f"recording single clip PAD={pad}")
        record_clip(pad, os.path.join(OUT_DIR, name))
        return

    print("recording 5 PAD clips (S-curve, 8s, 30fps)")
    for pad, name in PADS:
        record_clip(pad, os.path.join(OUT_DIR, name))
    print("done. outputs/pad_*.mp4")


if __name__ == "__main__":
    main()
