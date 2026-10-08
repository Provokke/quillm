#!/usr/bin/env python3
"""Create quillm .qlm models.

  random model:     make_model.py random out.qlm [--dtype q8_0] [--dim 288 --layers 6 ...]
  llama2.c import:  make_model.py llama2c ckpt.bin out.qlm [--dtype q4_0]
  tiny real model:  make_model.py fetch-tinystories out.qlm   (needs huggingface.co; skips if unreachable)
"""
import argparse
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qlm  # noqa: E402

CORPUS = (
    "Once upon a time there was a little fox who lived in a quiet forest. Every morning the fox "
    "ran to the river to drink and to watch the fish jump. One day the fox met a small bird. "
    "The bird was sad because it had lost its way home. The fox said: do not be sad, I will help you. "
    "Together they walked through the forest, past the tall trees and the old stone bridge. "
    "When the sun went down they found the bird's nest high in an oak tree. The bird was happy and "
    "thanked the fox. From that day on the fox and the bird were the best of friends, and they met "
    "every morning by the river to watch the fish jump in the sun. The end. "
) * 2


def train_merges(n: int):
    return qlm.bpe_train(CORPUS.encode(), n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("random")
    r.add_argument("out")
    r.add_argument("--dtype", default="f32", choices=list(qlm.DTYPES))
    r.add_argument("--dim", type=int, default=288)
    r.add_argument("--hidden", type=int, default=768)
    r.add_argument("--layers", type=int, default=6)
    r.add_argument("--heads", type=int, default=6)
    r.add_argument("--kv-heads", type=int, default=6)
    r.add_argument("--vocab", type=int, default=512)
    r.add_argument("--max-seq", type=int, default=256)
    r.add_argument("--merges", type=int, default=200)
    r.add_argument("--seed", type=int, default=1)
    c = sub.add_parser("llama2c")
    c.add_argument("ckpt")
    c.add_argument("out")
    c.add_argument("--dtype", default="f32", choices=list(qlm.DTYPES))
    f = sub.add_parser("fetch-tinystories")
    f.add_argument("out")
    f.add_argument("--dtype", default="f32", choices=list(qlm.DTYPES))
    f.add_argument("--url", default="https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/stories260K.bin")
    a = ap.parse_args()

    if a.cmd == "random":
        cfg = qlm.Config(dim=a.dim, hidden_dim=a.hidden, n_layers=a.layers, n_heads=a.heads, n_kv_heads=a.kv_heads,
                         vocab_size=a.vocab, max_seq=a.max_seq)
        merges = train_merges(min(a.merges, a.vocab - qlm.FIRST_MERGE))
        W = qlm.random_weights(cfg, a.seed)
        qlm.save_model(a.out, cfg, W, a.dtype, merges)
        print(f"wrote {a.out}: random {a.dtype} model, {sum(w.size for w in W.values())/1e6:.2f}M params, "
              f"{len(merges)} BPE merges")
    elif a.cmd == "llama2c":
        cfg, W = qlm.read_llama2c(a.ckpt)
        qlm.save_model(a.out, cfg, W, a.dtype, None, name=os.path.basename(a.ckpt))
        print(f"wrote {a.out}: converted llama2.c checkpoint ({cfg}) as {a.dtype}; "
              "note: the sentencepiece tokenizer is not converted")
    else:
        try:
            tmp = a.out + ".download"
            with urllib.request.urlopen(a.url, timeout=20) as resp, open(tmp, "wb") as fh:
                fh.write(resp.read())
        except Exception as e:  # network policy may block huggingface.co
            print(f"SKIP: could not download {a.url}: {e}")
            return 0
        cfg, W = qlm.read_llama2c(tmp)
        qlm.save_model(a.out, cfg, W, a.dtype, None, name="stories260K")
        os.remove(tmp)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
