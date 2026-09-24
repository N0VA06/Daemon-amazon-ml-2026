"""On-the-fly augmentation for bi-encoder training (both sides).

Names: swap abbreviations, drop/change legal suffix, reorder tokens.
Addresses: reorder comma components, drop PIN/ZIP/state, add/remove landmark.
Typos: 1–2 character edits.
Hard-negative generator: one-digit house change, W↔E, swap one core-name token.
"""

from __future__ import annotations

import random
import re

from src.normalize import (
    ABBREV_FR,
    ABBREV_IN,
    ABBREV_US,
    LEGAL_FORMS,
    basic_clean,
    tokens,
)

_DIGIT_RE = re.compile(r"\d")
_WE_RE = re.compile(r"\b(west|east|w|e)\b", re.I)
_LANDMARKS = [
    "near sbi atm",
    "opp bus stand",
    "behind temple",
    "next to post office",
    "proche de la gare",
]


def _rng(seed: int | None) -> random.Random:
    return random.Random(seed)


def swap_abbreviation(name: str, rng: random.Random) -> str:
    inv = {}
    for d in (ABBREV_US, ABBREV_IN, ABBREV_FR):
        for k, v in d.items():
            inv.setdefault(v, []).append(k)
            inv.setdefault(k, []).append(v)
    toks = tokens(name)
    if not toks:
        return name
    idxs = [i for i, t in enumerate(toks) if t in inv]
    if not idxs:
        return name
    i = rng.choice(idxs)
    toks[i] = rng.choice(inv[toks[i]])
    return " ".join(toks)


def drop_or_change_legal(name: str, rng: random.Random) -> str:
    text = name
    forms = [f for f, _ in LEGAL_FORMS]
    rng.shuffle(forms)
    for form in forms:
        if text.endswith(form):
            if rng.random() < 0.5:
                return text[: -len(form)].strip()
            alt = rng.choice(forms)
            return (text[: -len(form)] + " " + alt).strip()
    return text


def reorder_tokens(text: str, rng: random.Random) -> str:
    toks = tokens(text)
    if len(toks) < 2:
        return text
    rng.shuffle(toks)
    return " ".join(toks)


def reorder_address_components(address: str, rng: random.Random) -> str:
    parts = [p.strip() for p in address.split(",") if p.strip()]
    if len(parts) < 2:
        return address
    rng.shuffle(parts)
    return ", ".join(parts)


def drop_pin_or_state(address: str, rng: random.Random) -> str:
    parts = [p.strip() for p in address.split(",") if p.strip()]
    if not parts:
        return address
    # drop a trailing component (often state / PIN) or a 5–6 digit token
    kept = []
    dropped = False
    for p in parts:
        if (not dropped) and re.fullmatch(r"\d{5,6}", p.replace(" ", "")):
            dropped = True
            continue
        kept.append(p)
    if not dropped and len(kept) > 2 and rng.random() < 0.7:
        kept = kept[:-1]
    return ", ".join(kept) if kept else address


def add_or_remove_landmark(address: str, rng: random.Random) -> str:
    low = address.lower()
    if any(m in low for m in ("near ", "opp ", "behind ", "proche ")):
        parts = [p for p in address.split(",") if not any(
            m in p.lower() for m in ("near", "opp", "behind", "proche")
        )]
        return ", ".join(parts) if parts else address
    return address.rstrip() + ", " + rng.choice(_LANDMARKS)


def inject_typos(text: str, rng: random.Random, n: int | None = None) -> str:
    if not text:
        return text
    n = n if n is not None else rng.choice((1, 2))
    chars = list(text)
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    for _ in range(n):
        if not chars:
            break
        op = rng.choice(("swap", "drop", "insert", "replace"))
        i = rng.randrange(len(chars))
        if op == "swap" and i + 1 < len(chars):
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        elif op == "drop" and len(chars) > 3:
            chars.pop(i)
        elif op == "insert":
            chars.insert(i, rng.choice(alphabet))
        else:
            chars[i] = rng.choice(alphabet)
    return "".join(chars)


def augment_record(
    name: str,
    address: str,
    rng: random.Random | None = None,
    p: float = 0.7,
) -> tuple[str, str]:
    rng = rng or random.Random()
    if rng.random() > p:
        return name, address
    ops_name = [swap_abbreviation, drop_or_change_legal, reorder_tokens, inject_typos]
    ops_addr = [reorder_address_components, drop_pin_or_state, add_or_remove_landmark, inject_typos]
    name_fn = rng.choice(ops_name)
    addr_fn = rng.choice(ops_addr)
    try:
        name = name_fn(name, rng)
    except TypeError:
        name = name_fn(name, rng)
    try:
        address = addr_fn(address, rng)
    except TypeError:
        address = addr_fn(address, rng)
    return name, address


def flip_we(address: str) -> str:
    def _sub(m: re.Match) -> str:
        tok = m.group(1)
        table = {
            "west": "east",
            "east": "west",
            "w": "e",
            "e": "w",
            "West": "East",
            "East": "West",
            "W": "E",
            "E": "W",
        }
        return table.get(tok, tok)

    return _WE_RE.sub(_sub, address)


def change_one_house_digit(address: str, rng: random.Random | None = None) -> str:
    rng = rng or random.Random()
    digits = list(_DIGIT_RE.finditer(address))
    if not digits:
        return address
    # prefer a short number (house), not a 5–6 digit postcode
    candidates = [m for m in digits if 1 <= len(m.group()) <= 4]
    m = rng.choice(candidates or digits)
    old = m.group()
    i = rng.randrange(len(old))
    new_d = str((int(old[i]) + rng.choice((1, 2, 3, 7, 8, 9))) % 10)
    new = old[:i] + new_d + old[i + 1 :]
    return address[: m.start()] + new + address[m.end() :]


def swap_core_token(name: str, rng: random.Random | None = None) -> str:
    rng = rng or random.Random()
    toks = tokens(basic_clean(name))
    if len(toks) < 2:
        return name + " group"
    i = rng.randrange(len(toks))
    pool = ["group", "center", "services", "associates", "partners", "global", "prime"]
    toks[i] = rng.choice(pool)
    return " ".join(toks)


def make_hard_negative(
    name: str,
    address: str,
    rng: random.Random | None = None,
) -> tuple[str, str]:
    rng = rng or random.Random()
    kind = rng.choice(("digit", "we", "token"))
    if kind == "digit":
        return name, change_one_house_digit(address, rng)
    if kind == "we":
        return name, flip_we(address)
    return swap_core_token(name, rng), address
