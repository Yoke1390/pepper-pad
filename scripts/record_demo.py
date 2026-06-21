"""M6: 要求 5 PAD の MP4 クリップを録画する。

同一パス・同一カメラで [0,0,0],[1,1,1],[-1,-1,-1],[-1,1,-1],[-1,1,1] を出力。
PAD 値と感情ラベル(Joy など)を焼き込み、地面にパスを表示する。M7 で連結する。

各クリップは互いに独立で、律速は CPU 描画 (getCameraImage / ER_TINY_RENDERER,
~0.5s/frame) なので、5 本を multiprocessing.Pool で**別プロセス並列**に描画して
実時間を短縮する (画質は据え置き)。macOS の spawn では各子プロセスが独自の
pybullet DIRECT 接続を持つため安全 (親は録画前に PepperScene を作らない)。

進捗表示: 子で個別のバーを描くとカーソルが衝突して崩れ、さらに pybullet/qibullet
が標準出力に出すノイズもバーを壊す。そこで
  - 進捗は **親プロセスの単一 tqdm バー**(1行) に集約し、各クリップの % は
    バーの postfix に出す (単一行なので崩れない)。
  - 子は進捗を **Queue で送るだけ**。子の標準出力は devnull に捨てる。
  - pybullet のインポート時バナーも抑制する。

使い方:
    uv run python scripts/record_demo.py            # 5 本を並列録画 (単一バー)
    uv run python scripts/record_demo.py 1 1 1      # 単発 PAD を 1 本
"""
from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool, Manager

# pybullet のインポート時 banner ("pybullet build time: ...") を抑制 (親・各子共通)。
_dn = os.open(os.devnull, os.O_WRONLY)
_o1 = os.dup(1)
os.dup2(_dn, 1)
from tqdm import tqdm
from pepper_pad.record import record_clip, EMOTION_SHORT
os.dup2(_o1, 1)
os.close(_dn)
os.close(_o1)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

PADS = [
    ((0, 0, 0), "pad_000_baseline.mp4"),
    ((1, 1, 1), "pad_111_joy.mp4"),
    ((-1, -1, -1), "pad_nnn_sad.mp4"),
    ((-1, 1, -1), "pad_n1n_fear.mp4"),
    ((-1, 1, 1), "pad_n11_anger.mp4"),
]

# 1 クリップの制御ステップ数 (record_clip と同式: duration 8s, dt 1/60 → 481)
TOTAL = int(8.0 / (1.0 / 60.0)) + 1


def _silence():
    """子プロセスの標準出力を捨てる (pybullet/qibullet の b3Warning 等でバーを汚さない)。
    stderr は本当のエラー確認用に残す。"""
    os.dup2(os.open(os.devnull, os.O_WRONLY), 1)


def _worker(args):
    """別プロセスで 1 クリップ録画。進捗は Queue で親へ送るだけ (バーは描かない)。"""
    pad, out_path, idx, q = args
    err = record_clip(pad, out_path, progress=False, verbose=False,
                      on_step=lambda: q.put(("step", idx)))
    q.put(("done", idx, err))
    return out_path, err


def main() -> None:
    # 単発 PAD: 1 本だけ録画 (内蔵バー表示)
    if len(sys.argv) == 4:
        pad = tuple(int(v) for v in sys.argv[1:4])
        name = f"pad_{pad[0]}{pad[1]}{pad[2]}.mp4".replace("-", "n")
        record_clip(pad, os.path.join(OUT_DIR, name), progress=True,
                    desc=EMOTION_SHORT.get(pad, name))
        return

    workers = min(len(PADS), os.cpu_count() or 1)
    names = [EMOTION_SHORT.get(pad, name) for pad, name in PADS]
    mgr = Manager()
    q = mgr.Queue()
    tasks = [(pad, os.path.join(OUT_DIR, name), i, q)
             for i, (pad, name) in enumerate(PADS)]

    counts = [0] * len(PADS)
    errs: dict[int, float] = {}

    def postfix() -> str:
        return " ".join(f"{names[i][:3]}{100 * min(counts[i], TOTAL) // TOTAL:3d}%"
                        for i in range(len(PADS)))

    print(f"recording {len(PADS)} clips in parallel ({workers} workers, "
          f"S-curve, 8s, 30fps)")

    # 親プロセスが 1 本の tqdm バーで全体進捗を描画。各クリップ % は postfix に表示。
    bar = tqdm(total=len(PADS) * TOTAL, desc="render", unit="step")
    done = 0
    last = 0.0
    with Pool(processes=workers, initializer=_silence) as pool:
        res = pool.map_async(_worker, tasks)
        while done < len(PADS):
            msg = q.get()
            if msg[0] == "step":
                counts[msg[1]] += 1
                bar.update(1)
            else:                              # ("done", idx, err)
                _, idx, err = msg
                bar.update(TOTAL - counts[idx])
                counts[idx] = TOTAL
                errs[idx] = err
                done += 1
            now = time.time()
            if now - last > 0.2 or done == len(PADS):
                last = now
                bar.set_postfix_str(postfix())
        res.get()
    bar.close()

    for i, (pad, name) in enumerate(PADS):
        print(f"  {name:24s} max task err = {errs[i]:.2e} m")
    print("done. outputs/pad_*.mp4")


if __name__ == "__main__":
    main()
