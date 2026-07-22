from __future__ import annotations

import random
import re

from hba.schemas import Message, Sample

_NEIGHBORS = {
    "q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh",
    "u": "yij", "i": "uok", "o": "ipl", "p": "ol", "a": "qsz", "s": "awdz",
    "d": "sefx", "f": "drgc", "g": "fthv", "h": "gyjb", "j": "hukn", "k": "jilm",
    "l": "kop", "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn",
    "n": "bhjm", "m": "njk",
}

_HINGLISH_VARIANTS = {
    "nahi": ["nahin", "nhi", "nahiiii"],
    "hai": ["h", "hain", "hei"],
    "kar": ["kr", "karo", "kro"],
    "karo": ["kro", "kar do", "karoo"],
    "kya": ["kia", "kyaa", "kyà"],
    "kyun": ["kyu", "kyon", "q"],
    "mujhe": ["muje", "mjhe", "mujhee"],
    "aap": ["ap", "aap", "aapp"],
    "please": ["pls", "plz", "pleej"],
    "yaar": ["yar", "yr", "yaarr"],
    "abhi": ["abi", "abhii", "abhee"],
    "mera": ["mra", "meraa", "mere"],
    "batao": ["btao", "bata do", "bataao"],
    "chahiye": ["chahiy", "cahiye", "chaiye"],
    "paise": ["pese", "paisa", "paisey"],
    "account": ["acount", "acc", "a/c"],
    "balance": ["balnce", "balence", "bal"],
    "transaction": ["transacton", "txn", "transaction"],
}

_WORD = re.compile(r"[A-Za-z]+")


def _swap_adjacent(w: str, rng: random.Random) -> str:
    if len(w) < 2:
        return w
    i = rng.randint(0, len(w) - 2)
    return w[:i] + w[i + 1] + w[i] + w[i + 2 :]


def _drop_char(w: str, rng: random.Random) -> str:
    if len(w) < 3:
        return w
    i = rng.randint(0, len(w) - 1)
    return w[:i] + w[i + 1 :]


def _neighbor_sub(w: str, rng: random.Random) -> str:
    if not w:
        return w
    i = rng.randint(0, len(w) - 1)
    c = w[i].lower()
    if c not in _NEIGHBORS:
        return w
    rep = rng.choice(_NEIGHBORS[c])
    return w[:i] + rep + w[i + 1 :]


def _repeat_char(w: str, rng: random.Random) -> str:
    if not w:
        return w
    i = rng.randint(0, len(w) - 1)
    return w[: i + 1] + w[i] + w[i + 1 :]


def _hinglish_variant(w: str, rng: random.Random) -> str:
    low = w.lower()
    if low in _HINGLISH_VARIANTS:
        return rng.choice(_HINGLISH_VARIANTS[low])
    return w


_CHAR_OPS = [_swap_adjacent, _drop_char, _neighbor_sub, _repeat_char]


def noise_text(text: str, rng: random.Random, intensity: float = 0.15) -> str:
    tokens = text.split()
    out: list[str] = []
    for tok in tokens:
        m = _WORD.search(tok)
        if m and rng.random() < intensity * 2:
            core = m.group()
            noised = _hinglish_variant(core, rng)
            if noised == core and rng.random() < intensity:
                noised = rng.choice(_CHAR_OPS)(core, rng)
            tok = tok[: m.start()] + noised + tok[m.end() :]
        out.append(tok)
    result = " ".join(out)
    if rng.random() < 0.4:
        result = result.lower()
    if rng.random() < 0.3:
        result = result.rstrip("?.!")
    return result


def noise_sample(sample: Sample, rng: random.Random, intensity: float = 0.15) -> Sample:
    new_messages: list[Message] = []
    for m in sample.messages:
        if m.role == "user" and m.content:
            new_messages.append(Message(role="user", content=noise_text(m.content, rng, intensity)))
        else:
            new_messages.append(m)
    data = sample.model_dump()
    data["messages"] = [m.model_dump() for m in new_messages]
    data["noisy"] = True
    return Sample(**data)


def apply_noise(samples: list[Sample], fraction: float = 0.3, seed: int = 0) -> list[Sample]:
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    chosen = set(idx[: int(len(samples) * fraction)])
    out: list[Sample] = []
    for i, s in enumerate(samples):
        out.append(noise_sample(s, rng) if i in chosen else s)
    return out