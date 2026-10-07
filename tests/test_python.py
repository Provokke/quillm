"""Python-side tests: quantizer properties, file format round trip, BPE, llama2.c
converter, and the C CLI driven end-to-end on freshly generated models."""
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import qlm  # noqa: E402

BIN = os.environ.get("QM_BIN", os.path.join(ROOT, "build", "quillm"))


@pytest.mark.parametrize("n", [1, 31, 32, 33, 100, 257])
@pytest.mark.parametrize("kind", ["q8_0", "q4_0"])
def test_quant_roundtrip_bounds(n, kind):
    rng = np.random.default_rng(n)
    x = (rng.standard_normal((7, n)) * 3).astype(np.float32)
    raw, deq = qlm.encode_tensor(x, kind)
    assert len(raw) == 7 * ((n + 31) // 32) * (34 if kind == "q8_0" else 18)
    qmax = 127 if kind == "q8_0" else 7
    xb = qlm._blocks(x)
    amax = np.abs(xb).max(-1)
    d = (amax / np.float32(qmax)).astype(np.float16).astype(np.float32)
    bound = np.repeat(0.5 * d + np.maximum(0, amax - qmax * d), 32, axis=1)[:, :n] + 1e-6
    assert np.all(np.abs(x - deq) <= bound)


def test_q8_scale_is_fp16_of_amax_over_127():
    x = np.zeros((1, 32), np.float32)
    x[0, 3] = -254.0
    b = qlm.quantize_q8(x)
    assert b["d"][0, 0] == np.float16(2.0).view(np.uint16) and b["qs"][0, 0, 3] == -127


def test_qlm_roundtrip(tmp_path):
    cfg = qlm.Config(dim=32, hidden_dim=40, n_layers=1, n_heads=2, n_kv_heads=1, vocab_size=300, max_seq=16)
    W = qlm.random_weights(cfg, 0)
    merges = qlm.bpe_train(b"abababab cdcdcd abab", 10)
    for dt in ("f32", "q8_0", "q4_0"):
        p = tmp_path / f"m_{dt}.qlm"
        deq = qlm.save_model(str(p), cfg, W, dt, merges)
        cfg2, merges2, W2 = qlm.load_weights(str(p))
        for f in ("dim", "hidden_dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "max_seq"):
            assert getattr(cfg2, f) == getattr(cfg, f)
        assert cfg2.norm_eps == pytest.approx(cfg.norm_eps, rel=1e-6)  # stored as f32
        assert np.array_equal(merges2, merges)
        for k in W:
            assert np.array_equal(W2[k], deq[k]), k
        assert os.path.getsize(p) % 64 == 0


def test_bpe_roundtrip_and_compression():
    text = b"the cat sat on the mat. the cat sat on the hat. " * 4
    merges = qlm.bpe_train(text, 40)
    ids = qlm.bpe_encode(merges, text)
    assert ids[0] == qlm.BOS and len(ids) < len(text) // 2
    assert qlm.bpe_decode(merges, ids) == text
    rng = np.random.default_rng(1)
    for _ in range(50):
        s = bytes(rng.integers(0, 256, rng.integers(0, 80)).astype(np.uint8))
        assert qlm.bpe_decode(merges, qlm.bpe_encode(merges, s, add_bos=False)) == s


def test_llama2c_roundtrip(tmp_path):
    cfg = qlm.Config(dim=32, hidden_dim=64, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=280, max_seq=8)
    W = qlm.random_weights(cfg, 3)

    p = tmp_path / "ck.bin"
    qlm.write_llama2c(str(p), cfg, W, shared=False)
    cfg2, W2 = qlm.read_llama2c(str(p))
    assert cfg2 == cfg
    for k in W:
        assert np.array_equal(W[k], W2[k]), k


def test_reference_forward_causal():
    """logits at position t must not depend on tokens after t"""
    cfg = qlm.Config(dim=32, hidden_dim=48, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=270, max_seq=16)
    W = qlm.random_weights(cfg, 5)
    a = qlm.forward_ref(cfg, W, [1, 5, 9, 12, 100])
    b = qlm.forward_ref(cfg, W, [1, 5, 9, 200, 3])
    assert np.allclose(a[:3], b[:3]) and not np.allclose(a[3], b[3])


@pytest.mark.skipif(not os.path.exists(BIN), reason="C binary not built")
@pytest.mark.parametrize("dt", ["f32", "q8_0", "q4_0"])
def test_cli_greedy_matches_numpy_argmax(tmp_path, dt):
    """Greedy generation from the C CLI equals greedy decoding with the numpy reference
    (f32 exactly; quantized models must at least produce the right first token)."""
    cfg = qlm.Config(dim=64, hidden_dim=96, n_layers=2, n_heads=4, n_kv_heads=4, vocab_size=300, max_seq=48)
    W = qlm.random_weights(cfg, 77)
    merges = qlm.bpe_train(b"hello world hello there world hello", 20)
    p = str(tmp_path / "m.qlm")
    deq = qlm.save_model(p, cfg, W, dt, merges)
    prompt = b"hello world"
    toks = qlm.bpe_encode(merges, prompt)
    n_gen = 12 if dt == "f32" else 1
    seq = list(toks)
    for _ in range(n_gen):
        nxt = int(np.argmax(qlm.forward_ref(cfg, deq, seq)[-1]))
        if nxt == qlm.EOS:
            break
        seq.append(nxt)
    expect = prompt + qlm.bpe_decode(merges, seq[len(toks):])
    out = subprocess.run([BIN, "run", p, "-p", prompt.decode(), "-n", str(n_gen), "--temp", "0", "--no-stats"],
                         capture_output=True, check=True).stdout
    assert out.rstrip(b"\n") == expect


def test_make_model_cli(tmp_path):
    out = tmp_path / "r.qlm"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "make_model.py"), "random", str(out), "--dim", "32",
                        "--hidden", "64", "--layers", "1", "--heads", "2", "--kv-heads", "2", "--vocab", "300",
                        "--dtype", "q4_0"], capture_output=True, text=True, check=True)
    assert "wrote" in r.stdout and out.exists()
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "make_model.py"), "fetch-tinystories",
                        str(tmp_path / "s.qlm")], capture_output=True, text=True, check=True)
    assert "SKIP" in r.stdout or (tmp_path / "s.qlm").exists()
