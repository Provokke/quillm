"""quillm Python side: .qlm writer/reader, Q8_0/Q4_0 quantizers (bit-exact with the C
reference), byte-level BPE trainer/encoder, a numpy reference forward pass, and a
llama2.c checkpoint reader. Only depends on numpy."""
from __future__ import annotations

import struct
from collections import Counter
from dataclasses import dataclass, asdict

import numpy as np

MAGIC = b"QLMF"
VERSION = 1
ALIGN = 64
DT_F32, DT_Q8_0, DT_Q4_0 = 0, 2, 3
DTYPES = {"f32": DT_F32, "q8_0": DT_Q8_0, "q4_0": DT_Q4_0}
KV_U32, KV_F32, KV_STR, KV_ARR_U32 = 0, 1, 2, 3
BOS, EOS, BYTE0, FIRST_MERGE = 0, 1, 2, 258

Q8_DT = np.dtype([("d", "<u2"), ("qs", "i1", 32)])
Q4_DT = np.dtype([("d", "<u2"), ("qs", "u1", 16)])
assert Q8_DT.itemsize == 34 and Q4_DT.itemsize == 18


# ---------------------------------------------------------------- quantization
def _blocks(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x2 = x.reshape(-1, x.shape[-1])
    n = x2.shape[1]
    nb = (n + 31) // 32
    pad = np.zeros((x2.shape[0], nb * 32), dtype=np.float32)
    pad[:, :n] = x2
    return pad.reshape(x2.shape[0], nb, 32)


def _scale(xb: np.ndarray, qmax: float):
    amax = np.abs(xb).max(axis=-1)
    d = (amax / np.float32(qmax)).astype(np.float32)
    dh = d.astype(np.float16)
    return dh, dh.astype(np.float32)


def _q(xb, dd, qmin, qmax):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.rint(xb / dd[..., None])
    r = np.where(dd[..., None] == 0, 0, r)
    return np.clip(r, qmin, qmax).astype(np.int32)


def quantize_q8(x: np.ndarray) -> np.ndarray:
    """rows x n float -> structured array [rows, nblocks] of Q8_0 blocks."""
    xb = _blocks(x)
    dh, dd = _scale(xb, 127.0)
    out = np.zeros(xb.shape[:2], dtype=Q8_DT)
    out["d"] = dh.view(np.uint16)
    out["qs"] = _q(xb, dd, -127, 127).astype(np.int8)
    return out


def quantize_q4(x: np.ndarray) -> np.ndarray:
    xb = _blocks(x)
    dh, dd = _scale(xb, 7.0)
    q = (_q(xb, dd, -8, 7) + 8).astype(np.uint8)
    out = np.zeros(xb.shape[:2], dtype=Q4_DT)
    out["d"] = dh.view(np.uint16)
    out["qs"] = q[..., 0::2] | (q[..., 1::2] << 4)
    return out


def dequantize(blocks: np.ndarray, n: int) -> np.ndarray:
    d = blocks["d"].view(np.float16).astype(np.float32)
    if blocks.dtype == Q8_DT:
        q = blocks["qs"].astype(np.float32)
    else:
        qs = blocks["qs"]
        q = np.empty(qs.shape[:-1] + (32,), dtype=np.float32)
        q[..., 0::2] = (qs & 0xF).astype(np.float32) - 8
        q[..., 1::2] = (qs >> 4).astype(np.float32) - 8
    v = (d[..., None] * q).reshape(blocks.shape[0], -1)
    return v[:, :n]


def encode_tensor(w: np.ndarray, dtype: str):
    """returns (raw bytes, dequantized float32 array with w's shape)"""
    w = np.asarray(w, dtype=np.float32)
    if dtype == "f32" or w.ndim == 1:
        return w.tobytes(), w
    n = w.shape[-1]
    b = quantize_q8(w) if dtype == "q8_0" else quantize_q4(w)
    return b.tobytes(), dequantize(b, n).reshape(w.shape)


# ---------------------------------------------------------------- file format
def write_qlm(path: str, kvs: list, tensors: list) -> None:
    """kvs: [(key, type, value)]; tensors: [(name, dtype_id, shape, raw_bytes)]"""
    head = bytearray()

    def s(b: bytes):
        head.extend(struct.pack("<I", len(b)))
        head.extend(b)

    for key, ty, val in kvs:
        s(key.encode())
        head.extend(struct.pack("<I", ty))
        if ty == KV_U32:
            head.extend(struct.pack("<I", val))
        elif ty == KV_F32:
            head.extend(struct.pack("<f", val))
        elif ty == KV_STR:
            s(val.encode())
        elif ty == KV_ARR_U32:
            arr = np.asarray(val, dtype="<u4")
            head.extend(struct.pack("<I", arr.size))
            head.extend(arr.tobytes())
    offsets, off = [], 0
    for _, _, _, raw in tensors:
        offsets.append(off)
        off += (len(raw) + ALIGN - 1) // ALIGN * ALIGN
    for (name, dt, shape, raw), o in zip(tensors, offsets):
        s(name.encode())
        head.extend(struct.pack("<II", dt, len(shape)))
        head.extend(struct.pack(f"<{len(shape)}Q", *shape))
        head.extend(struct.pack("<Q", o))
    pre = 4 + 4 + 4 + 4 + 8
    data_off = (pre + len(head) + ALIGN - 1) // ALIGN * ALIGN
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<IIIQ", VERSION, len(kvs), len(tensors), data_off))
        f.write(head)
        f.write(b"\0" * (data_off - pre - len(head)))
        for (_, _, _, raw), o in zip(tensors, offsets):
            f.write(raw)
            f.write(b"\0" * ((len(raw) + ALIGN - 1) // ALIGN * ALIGN - len(raw)))


def read_qlm(path: str):
    """returns (kv dict, {name: (dtype, shape, raw bytes)})"""
    buf = open(path, "rb").read()
    assert buf[:4] == MAGIC
    ver, n_kv, n_t, data_off = struct.unpack_from("<IIIQ", buf, 4)
    assert ver == VERSION
    p = 24

    def rs():
        nonlocal p
        (n,) = struct.unpack_from("<I", buf, p)
        p += 4
        v = buf[p:p + n]
        p += n
        return v

    kv = {}
    for _ in range(n_kv):
        k = rs().decode()
        (ty,) = struct.unpack_from("<I", buf, p)
        p += 4
        if ty == KV_U32:
            (kv[k],) = struct.unpack_from("<I", buf, p)
            p += 4
        elif ty == KV_F32:
            (kv[k],) = struct.unpack_from("<f", buf, p)
            p += 4
        elif ty == KV_STR:
            kv[k] = rs().decode()
        else:
            (n,) = struct.unpack_from("<I", buf, p)
            p += 4
            kv[k] = np.frombuffer(buf, "<u4", n, p).copy()
            p += 4 * n
    tens = {}
    for _ in range(n_t):
        name = rs().decode()
        dt, nd = struct.unpack_from("<II", buf, p)
        p += 8
        shape = struct.unpack_from(f"<{nd}Q", buf, p)
        p += 8 * nd
        (o,) = struct.unpack_from("<Q", buf, p)
        p += 8
        rows = int(np.prod(shape[:-1])) if nd > 1 else 1
        nb = (shape[-1] + 31) // 32
        size = {DT_F32: 4 * shape[-1], DT_Q8_0: 34 * nb, DT_Q4_0: 18 * nb}[dt] * rows
        tens[name] = (dt, shape, buf[data_off + o:data_off + o + size])
    return kv, tens


def load_weights(path: str):
    """read a .qlm model back into (Config, merges, dequantized float32 weights)"""
    kv, tens = read_qlm(path)
    cfg = Config(dim=kv["llama.dim"], hidden_dim=kv["llama.hidden_dim"], n_layers=kv["llama.n_layers"],
                 n_heads=kv["llama.n_heads"], n_kv_heads=kv["llama.n_kv_heads"], vocab_size=kv["llama.vocab_size"],
                 max_seq=kv["llama.max_seq"], rope_theta=kv["llama.rope_theta"], norm_eps=kv["llama.norm_eps"])
    W = {}
    for name, (dt, shape, raw) in tens.items():
        if dt == DT_F32:
            W[name] = np.frombuffer(raw, np.float32).reshape(shape).copy()
        else:
            b = np.frombuffer(raw, Q8_DT if dt == DT_Q8_0 else Q4_DT).reshape(shape[0], -1)
            W[name] = dequantize(b, shape[-1])
    merges = kv.get("tokenizer.merges", np.zeros(0, np.uint32)).reshape(-1, 2)
    return cfg, merges, W


# ---------------------------------------------------------------- model
@dataclass
class Config:
    dim: int = 64
    hidden_dim: int = 172
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 2
    vocab_size: int = 458
    max_seq: int = 64
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5


def random_weights(cfg: Config, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    D, H, V = cfg.dim, cfg.hidden_dim, cfg.vocab_size
    KV = D // cfg.n_heads * cfg.n_kv_heads

    def mat(o, i):
        return (rng.standard_normal((o, i)) / np.sqrt(i)).astype(np.float32)

    def norm(n):
        return (1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32)

    W = {"tok_embd": rng.standard_normal((V, D)).astype(np.float32)}
    for l in range(cfg.n_layers):
        p = f"layers.{l}."
        W[p + "attn_norm"] = norm(D)
        W[p + "wq"], W[p + "wk"], W[p + "wv"], W[p + "wo"] = mat(D, D), mat(KV, D), mat(KV, D), mat(D, D)
        W[p + "ffn_norm"] = norm(D)
        W[p + "w1"], W[p + "w2"], W[p + "w3"] = mat(H, D), mat(D, H), mat(H, D)
    W["norm"] = norm(D)
    W["output"] = mat(V, D) * 4.0  # sharper logits so sampling tests are meaningful
    return W


def tensor_order(cfg: Config):
    names = ["tok_embd"]
    for l in range(cfg.n_layers):
        names += [f"layers.{l}.{n}" for n in ("attn_norm", "wq", "wk", "wv", "wo", "ffn_norm", "w1", "w2", "w3")]
    return names + ["norm", "output"]


def save_model(path: str, cfg: Config, W: dict, dtype: str, merges=None, name: str = "quillm-random") -> dict:
    """write model; returns the dequantized weights (what the C engine actually sees)."""
    kvs = [("general.name", KV_STR, name)]
    for k, v in asdict(cfg).items():
        kvs.append((f"llama.{k}", KV_F32 if isinstance(v, float) else KV_U32, v))
    if merges is not None and len(merges):
        kvs.append(("tokenizer.merges", KV_ARR_U32, np.asarray(merges, np.uint32).reshape(-1)))
    tensors, deq = [], {}
    for n in tensor_order(cfg):
        w = W[n]
        raw, dq = encode_tensor(w, dtype)
        dt = DT_F32 if (dtype == "f32" or w.ndim == 1) else DTYPES[dtype]
        tensors.append((n, dt, list(w.shape), raw))
        deq[n] = dq
    write_qlm(path, kvs, tensors)
    return deq


def _rmsnorm(x, w, eps):
    return x / np.sqrt(np.mean(x * x) + eps) * w


def _rope(v, n_heads, hd, pos, theta):
    v = v.reshape(n_heads, hd).copy()
    i = np.arange(hd // 2)
    ang = pos * theta ** (-2.0 * i / hd)
    c, s = np.cos(ang), np.sin(ang)
    x0, x1 = v[:, 0::2].copy(), v[:, 1::2].copy()
    v[:, 0::2] = x0 * c - x1 * s
    v[:, 1::2] = x0 * s + x1 * c
    return v.reshape(-1)


def forward_ref(cfg: Config, W: dict, tokens) -> np.ndarray:
    """float64 numpy reference forward pass; returns logits [len(tokens), vocab]"""
    W = {k: np.asarray(v, np.float64) for k, v in W.items()}
    D, hd = cfg.dim, cfg.dim // cfg.n_heads
    group = cfg.n_heads // cfg.n_kv_heads
    kc = [[] for _ in range(cfg.n_layers)]
    vc = [[] for _ in range(cfg.n_layers)]
    out = []
    for pos, tok in enumerate(tokens):
        x = W["tok_embd"][tok].copy()
        for l in range(cfg.n_layers):
            p = f"layers.{l}."
            xb = _rmsnorm(x, W[p + "attn_norm"], cfg.norm_eps)
            q = _rope(W[p + "wq"] @ xb, cfg.n_heads, hd, pos, cfg.rope_theta)
            k = _rope(W[p + "wk"] @ xb, cfg.n_kv_heads, hd, pos, cfg.rope_theta)
            kc[l].append(k)
            vc[l].append(W[p + "wv"] @ xb)
            K, V = np.stack(kc[l]), np.stack(vc[l])
            att_out = np.zeros(D)
            for h in range(cfg.n_heads):
                kh = h // group
                sc = K[:, kh * hd:(kh + 1) * hd] @ q[h * hd:(h + 1) * hd] / np.sqrt(hd)
                sc = np.exp(sc - sc.max())
                sc /= sc.sum()
                att_out[h * hd:(h + 1) * hd] = sc @ V[:, kh * hd:(kh + 1) * hd]
            x = x + W[p + "wo"] @ att_out
            xb = _rmsnorm(x, W[p + "ffn_norm"], cfg.norm_eps)
            g = W[p + "w1"] @ xb
            x = x + W[p + "w2"] @ (g / (1 + np.exp(-g)) * (W[p + "w3"] @ xb))
        x = _rmsnorm(x, W["norm"], cfg.norm_eps)
        out.append(W["output"] @ x)
    return np.stack(out)


# ---------------------------------------------------------------- tokenizer
def bpe_train(text: bytes, n_merges: int):
    ids = [BYTE0 + b for b in text]
    merges = []
    for m in range(n_merges):
        counts = Counter(zip(ids, ids[1:]))
        if not counts:
            break
        pair, cnt = max(counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
        if cnt < 2:
            break
        ids = _merge(ids, pair, FIRST_MERGE + m)
        merges.append(pair)
    return np.array(merges, dtype=np.uint32).reshape(-1, 2)


def _merge(ids, pair, new):
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def bpe_encode(merges, text: bytes, add_bos=True):
    rank = {}
    for i, (a, b) in enumerate(np.asarray(merges).reshape(-1, 2).tolist()):
        rank.setdefault((a, b), i)
    ids = [BYTE0 + b for b in text]
    while True:
        cands = [rank[p] for p in zip(ids, ids[1:]) if p in rank]
        if not cands:
            break
        r = min(cands)
        a, b = (int(v) for v in merges[r])
        ids = _merge(ids, (a, b), FIRST_MERGE + r)
    return ([BOS] if add_bos else []) + ids


def bpe_decode(merges, ids) -> bytes:
    merges = np.asarray(merges).reshape(-1, 2)
    table = {BYTE0 + i: bytes([i]) for i in range(256)}
    for i, (a, b) in enumerate(merges.tolist()):
        table[FIRST_MERGE + i] = table[a] + table[b]
    return b"".join(table.get(i, b"") for i in ids)


# ---------------------------------------------------------------- llama2.c checkpoints
def read_llama2c(path: str):
    """Karpathy llama2.c legacy (v0) .bin: 7 x int32 header then fp32 weights."""
    buf = open(path, "rb").read()
    dim, hidden, n_layers, n_heads, n_kv, vocab, seq = struct.unpack_from("<7i", buf, 0)
    shared = vocab > 0
    vocab = abs(vocab)
    arr = np.frombuffer(buf, np.float32, offset=28)
    hd = dim // n_heads
    kvd = hd * n_kv
    p = 0

    def take(*shape):
        nonlocal p
        n = int(np.prod(shape))
        a = arr[p:p + n].reshape(shape)
        p += n
        return a

    emb = take(vocab, dim)
    rms_att = take(n_layers, dim)
    wq, wk, wv = take(n_layers, dim, dim), take(n_layers, kvd, dim), take(n_layers, kvd, dim)
    wo = take(n_layers, dim, dim)
    rms_ffn = take(n_layers, dim)
    w1, w2, w3 = take(n_layers, hidden, dim), take(n_layers, dim, hidden), take(n_layers, hidden, dim)
    rms_final = take(dim)
    take(seq, hd // 2)  # freq_cis_real (unused, RoPE is recomputed)
    take(seq, hd // 2)  # freq_cis_imag
    wcls = emb if shared else take(vocab, dim)
    cfg = Config(dim=dim, hidden_dim=hidden, n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_kv,
                 vocab_size=vocab, max_seq=seq)
    W = {"tok_embd": emb, "norm": rms_final, "output": wcls}
    for l in range(n_layers):
        p_ = f"layers.{l}."
        W.update({p_ + "attn_norm": rms_att[l], p_ + "wq": wq[l], p_ + "wk": wk[l], p_ + "wv": wv[l],
                  p_ + "wo": wo[l], p_ + "ffn_norm": rms_ffn[l], p_ + "w1": w1[l], p_ + "w2": w2[l],
                  p_ + "w3": w3[l]})
    return cfg, {k: np.ascontiguousarray(v, np.float32) for k, v in W.items()}


def write_llama2c(path: str, cfg: Config, W: dict, shared: bool) -> None:
    """write a llama2.c v0 checkpoint (used to test the converter without network access)"""
    hd = cfg.dim // cfg.n_heads
    L = cfg.n_layers
    with open(path, "wb") as f:
        v = cfg.vocab_size if shared else -cfg.vocab_size
        f.write(struct.pack("<7i", cfg.dim, cfg.hidden_dim, L, cfg.n_heads, cfg.n_kv_heads, v, cfg.max_seq))
        parts = [W["tok_embd"]]
        for n in ("attn_norm", "wq", "wk", "wv", "wo", "ffn_norm", "w1", "w2", "w3"):
            parts.append(np.stack([W[f"layers.{l}.{n}"] for l in range(L)]))
        parts.append(W["norm"])
        parts.append(np.zeros((cfg.max_seq, hd // 2 * 2), np.float32))  # freq cis real+imag
        if not shared:
            parts.append(W["output"])
        for a in parts:
            f.write(np.ascontiguousarray(a, np.float32).tobytes())
