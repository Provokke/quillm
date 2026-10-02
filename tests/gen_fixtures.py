#!/usr/bin/env python3
"""Generate test fixtures: models in every dtype, numpy reference logits, tokenizer and
quantizer cross-check vectors, and benchmark-sized models."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import qlm  # noqa: E402
from make_model import train_merges  # noqa: E402

PROMPT = b"Once upon a time the fox ran to the river"


def write_case(out, name, cfg, W, dtype, merges, tokens):
    path = os.path.join(out, f"{name}.qlm")
    deq = qlm.save_model(path, cfg, W, dtype, merges)
    ref = qlm.forward_ref(cfg, deq, tokens).astype(np.float32)
    reffp = qlm.forward_ref(cfg, W, tokens).astype(np.float32) if dtype != "f32" else ref
    ref.tofile(os.path.join(out, f"{name}.ref.f32"))
    reffp.tofile(os.path.join(out, f"{name}.reffp.f32"))
    with open(os.path.join(out, f"{name}.tokens"), "w") as f:
        f.write(f"{len(tokens)} {cfg.vocab_size}\n" + " ".join(map(str, tokens)) + "\n")
    return path


def main(out):
    os.makedirs(out, exist_ok=True)
    merges = train_merges(200)
    vocab = qlm.FIRST_MERGE + len(merges)
    tokens = qlm.bpe_encode(merges, PROMPT)
    tokens = (tokens * 3)[:20]  # 20 positions

    # model A: GQA (4 q heads / 2 kv heads), hidden 172 (not a multiple of 32 -> partial blocks)
    cfg_a = qlm.Config(dim=64, hidden_dim=172, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=vocab, max_seq=64)
    Wa = qlm.random_weights(cfg_a, 11)
    # model B: dim 72 (partial blocks in every matvec), head_dim 12, 3 layers
    cfg_b = qlm.Config(dim=72, hidden_dim=200, n_layers=3, n_heads=6, n_kv_heads=3, vocab_size=vocab, max_seq=64)
    Wb = qlm.random_weights(cfg_b, 22)
    for dt in ("f32", "q8_0", "q4_0"):
        write_case(out, f"small_{dt}", cfg_a, Wa, dt, merges, tokens)
        write_case(out, f"odd_{dt}", cfg_b, Wb, dt, merges, tokens)

    # llama2.c checkpoint round trip (converter path; shared classifier, MHA)
    cfg_c = qlm.Config(dim=48, hidden_dim=128, n_layers=2, n_heads=4, n_kv_heads=4, vocab_size=300, max_seq=32)
    Wc = qlm.random_weights(cfg_c, 33)
    Wc["output"] = Wc["tok_embd"]
    ck = os.path.join(out, "tiny_llama2c.bin")
    qlm.write_llama2c(ck, cfg_c, Wc, shared=True)
    cfg_c2, Wc2 = qlm.read_llama2c(ck)
    assert cfg_c2 == cfg_c
    write_case(out, "l2c_f32", cfg_c2, Wc2, "f32", None, [5, 17, 200, 9, 299, 0, 42, 42])

    # tokenizer vectors: text (hex) -> ids
    rng = np.random.default_rng(5)
    texts = [b"", b"a", PROMPT, b"the the the the", b"aaaaaaaaaaa", "héllo wörld — ünïcödé ✓".encode(),
             b"\x00\xff\x01binary\x80"] + [bytes(rng.integers(0, 256, rng.integers(1, 60)).astype(np.uint8))
                                            for _ in range(20)]
    words = PROMPT.split()
    for _ in range(20):
        texts.append(b" ".join(words[i] for i in rng.integers(0, len(words), 8)))
    with open(os.path.join(out, "tok.txt"), "w") as f:
        f.write(f"{len(texts)}\n")
        for t in texts:
            ids = qlm.bpe_encode(merges, t, add_bos=False)
            assert qlm.bpe_decode(merges, ids) == t
            f.write(f"{t.hex() or '-'} {len(ids)} {' '.join(map(str, ids))}\n")
    np.asarray(merges, np.uint32).tofile(os.path.join(out, "merges.u32"))

    # quantizer cross-check: same input quantized by numpy must be byte-identical in C
    x = (rng.standard_normal((5, 100)) * rng.uniform(0.01, 10, (5, 1))).astype(np.float32)
    x[2, 40:72] = 0.0  # an all-zero block
    x.tofile(os.path.join(out, "qin.f32"))
    qlm.quantize_q8(x).tofile(os.path.join(out, "qin.q8"))
    qlm.quantize_q4(x).tofile(os.path.join(out, "qin.q4"))


    # benchmark models: stories15M-shaped (dim 288, 6 layers, 6 heads, hidden 768, vocab 32000)
    cfg_bench = qlm.Config(dim=288, hidden_dim=768, n_layers=6, n_heads=6, n_kv_heads=6, vocab_size=32000,
                           max_seq=256)
    Wbench = qlm.random_weights(cfg_bench, 7)
    for dt in ("f32", "q8_0", "q4_0"):
        qlm.save_model(os.path.join(out, f"bench_{dt}.qlm"), cfg_bench, Wbench, dt, merges)
    print(f"fixtures written to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "build/fixtures")
