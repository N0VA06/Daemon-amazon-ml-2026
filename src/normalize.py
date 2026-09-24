"""Record normalisation, legal-form extraction, address parsing, abbreviation maps.

Country is an open set of strings. Legal-form and postcode patterns are applied
softly: known country labels pick a preferred regex, unknown labels try all
patterns. Nothing is hard-filtered to {US, India}.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd

# ---------------------------------------------------------------------------
# Legal forms (extracted into their own field; stripped from core_name)
# Longer phrases first.
# ---------------------------------------------------------------------------
LEGAL_FORMS: list[tuple[str, str]] = [
    # India
    ("private limited", "in_pvt_ltd"),
    ("pvt. ltd.", "in_pvt_ltd"),
    ("pvt ltd", "in_pvt_ltd"),
    ("pvt.ltd", "in_pvt_ltd"),
    ("pvt limited", "in_pvt_ltd"),
    ("private ltd", "in_pvt_ltd"),
    ("opc pvt ltd", "in_opc"),
    ("opc", "in_opc"),
    # US / shared
    ("incorporated", "us_inc"),
    ("corporation", "us_corp"),
    ("company", "us_co"),
    ("limited liability partnership", "llp"),
    ("limited liability company", "us_llc"),
    ("professional limited liability company", "us_pllc"),
    ("pllc", "us_pllc"),
    ("p.l.l.c.", "us_pllc"),
    ("l.l.c.", "us_llc"),
    ("llc", "us_llc"),
    ("llp", "llp"),
    ("l.l.p.", "llp"),
    ("l.p.", "us_lp"),
    ("p.c.", "us_pc"),
    ("p.a.", "us_pa"),
    ("inc.", "us_inc"),
    ("inc", "us_inc"),
    ("corp.", "us_corp"),
    ("corp", "us_corp"),
    ("ltd.", "ltd"),
    ("ltd", "ltd"),
    ("limited", "ltd"),
    ("co.", "us_co"),
    # France
    ("selarl", "fr_selarl"),
    ("sasu", "fr_sasu"),
    ("sarl", "fr_sarl"),
    ("eurl", "fr_eurl"),
    ("sas", "fr_sas"),
    ("sci", "fr_sci"),
    ("snc", "fr_snc"),
    ("s.a.s.u.", "fr_sasu"),
    ("s.a.r.l.", "fr_sarl"),
    ("s.a.s.", "fr_sas"),
    ("s.a.", "fr_sa"),
    ("cie", "fr_cie"),
    ("sté", "fr_ste"),
    ("ste", "fr_ste"),
    # generic last (dangerous short tokens — end-of-name only)
    ("lp", "us_lp"),
    ("pc", "us_pc"),
    ("pa", "us_pa"),
    ("co", "us_co"),
    ("sa", "fr_sa"),
]

# End-anchored only (too ambiguous mid-name)
_END_ONLY = {"lp", "pc", "pa", "co", "sa", "ste", "cie"}

# ---------------------------------------------------------------------------
# Hand-written abbreviation maps (no external lookup).
# Keys and values are lowercase, punctuation-stripped tokens.
# ---------------------------------------------------------------------------
ABBREV_US = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "av": "avenue",
    "ln": "lane",
    "dr": "drive",
    "blvd": "boulevard",
    "ct": "court",
    "hwy": "highway",
    "pkwy": "parkway",
    "cir": "circle",
    "pl": "place",
    "ter": "terrace",
    "trl": "trail",
    "sq": "square",
    "ste": "suite",
    "apt": "apartment",
    "fl": "floor",
    "bldg": "building",
    "dept": "department",
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
    "fwy": "freeway",
    "expy": "expressway",
    "tpke": "turnpike",
    "pob": "pobox",
    "po": "pobox",
}

ABBREV_IN = {
    "marg": "road",
    "ngr": "nagar",
    "nr": "near",
    "opp": "opposite",
    "bldg": "building",
    "bldng": "building",
    "soc": "society",
    "chs": "society",
    "sec": "sector",
    "sect": "sector",
    "rd": "road",
    "st": "street",
    "apt": "apartment",
    "flr": "floor",
    "gali": "lane",
    "hno": "housenumber",
    "h": "housenumber",
    "plot": "plot",
    "plt": "plot",
    "shop": "shop",
    "fl": "floor",
    "blk": "block",
    "ext": "extension",
    "extn": "extension",
    "dist": "district",
    "teh": "tehsil",
    "po": "postoffice",
}

ABBREV_FR = {
    "r": "rue",
    "av": "avenue",
    "ave": "avenue",
    "bd": "boulevard",
    "blvd": "boulevard",
    "pl": "place",
    "chem": "chemin",
    "fbg": "faubourg",
    "fg": "faubourg",
    "qu": "quai",
    "imp": "impasse",
    "rte": "route",
    "all": "allee",
    "allée": "allee",
    "crs": "cours",
    "sq": "square",
    "pass": "passage",
    "bat": "batiment",
    "bât": "batiment",
    "zac": "zac",
    "zi": "zi",
    "za": "za",
    "bp": "boitepostale",
    "cs": "cs",
    "cedex": "cedex",
    "bis": "bis",
    "ter": "ter",
    "quater": "quater",
}

# Common name-token variants seen in the challenge statement / train samples.
# Hand-written only (no scraped gazetteer).
NAME_VARIANTS = {
    "shree": "shri",
    "shri": "shri",
    "sree": "shri",
    "sri": "shri",
    "thiruvananthapuram": "trivandrum",
    "trivandrum": "trivandrum",
    "bengaluru": "bangalore",
    "bangalore": "bangalore",
    "mumbai": "mumbai",
    "bombay": "mumbai",
    "kolkata": "kolkata",
    "calcutta": "kolkata",
    "chennai": "chennai",
    "madras": "chennai",
    "pune": "pune",
    "poona": "pune",
    "gurugram": "gurgaon",
    "gurgaon": "gurgaon",
    "and": "and",
    "pvt": "private",
    "private": "private",
}

LANDMARK_MARKERS = (
    "near",
    "opp",
    "opposite",
    "behind",
    "beside",
    "next to",
    "nextto",
    "adj",
    "adjacent",
    "above",
    "below",
    "pas",
    "proche",
)

DIRECTION_TOKENS = {
    "n": "n",
    "s": "s",
    "e": "e",
    "w": "w",
    "north": "n",
    "south": "s",
    "east": "e",
    "west": "w",
    "ne": "ne",
    "nw": "nw",
    "se": "se",
    "sw": "sw",
}

_PUNCT_RE = re.compile(r"[^\w\s&+/.-]", re.UNICODE)
_MULTI_SPACE = re.compile(r"\s+")
_LEGAL_END_ONLY = set(_END_ONLY)

# House / unit / plot / shop
_HOUSE_RE = re.compile(
    r"(?i)(?:\b(?:h\.?\s*no\.?|houseno|house\s*no|plot|plt|shop|flat|unit|apt|"
    r"apartment|door|dno|d\.?\s*no\.?|#|no\.?)\s*)?(\d+[a-z]?)(?:\s*(?:bis|ter|quater))?"
)
_UNIT_RE = re.compile(
    r"(?i)\b(?:unit|apt|apartment|suite|ste|flat|floor|fl|#)\s*[:\-]?\s*([a-z0-9-]+)"
)
_US_ZIP_RE = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
_IN_PIN_RE = re.compile(r"\b(\d{6})\b")
_FR_POST_RE = re.compile(r"\b(\d{5})\b")
_CEDEX_RE = re.compile(r"(?i)\bcedex\s*(\d+)?")
_DIR_PAREN_RE = re.compile(r"\(([nsew])\)", re.I)
_DIGIT_RE = re.compile(r"\d+")


def nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "")


def fold_accents(text: str) -> str:
    """Latin accent fold only. Indic / CJK codepoints are kept intact."""
    out = []
    for ch in unicodedata.normalize("NFKD", text):
        if unicodedata.combining(ch):
            continue
        out.append(ch)
    return "".join(out)


def basic_clean(text: str, lowercase: bool = True, amp_to_and: bool = True) -> str:
    text = nfkc(text)
    if lowercase:
        text = text.lower()
    if amp_to_and:
        text = text.replace("&", " and ")
    text = text.replace("|", " ")
    text = _PUNCT_RE.sub(" ", text)
    text = text.replace("+", " ")
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text


def tokens(text: str) -> list[str]:
    if not text:
        return []
    return [t for t in text.split() if t]


def _legal_pattern(form: str) -> re.Pattern:
    return re.compile(rf"(?<![\w]){re.escape(form)}(?![\w])", re.I)


_LEGAL_COMPILED = [(form, tag, _legal_pattern(form)) for form, tag in LEGAL_FORMS]


def extract_legal_form(name_norm: str) -> tuple[str, str]:
    """Return (core_name, legal_tag). Prefers a match at the end of the name."""
    if not name_norm:
        return "", ""
    text = name_norm.strip()
    # end-of-name first
    for form, tag, cre in _LEGAL_COMPILED:
        if text.endswith(form) or text.endswith(" " + form):
            core = cre.sub(" ", text)
            return _MULTI_SPACE.sub(" ", core).strip(), tag
    for form, tag, cre in _LEGAL_COMPILED:
        if form in _LEGAL_END_ONLY:
            continue
        if cre.search(text):
            core = cre.sub(" ", text)
            return _MULTI_SPACE.sub(" ", core).strip(), tag
    return text, ""


def expand_abbreviations(text: str, country: str, extra: dict[str, str] | None = None) -> str:
    cmap = dict(ABBREV_US)
    cl = (country or "").strip().lower()
    if cl in {"india", "in"}:
        cmap.update(ABBREV_IN)
    elif cl in {"france", "fr"}:
        cmap.update(ABBREV_FR)
    else:
        # open-set: apply the union so unseen countries still expand common tokens
        cmap.update(ABBREV_IN)
        cmap.update(ABBREV_FR)
    if extra:
        cmap.update(extra)
    parts = []
    for tok in tokens(text):
        key = tok.replace(".", "")
        parts.append(cmap.get(key, NAME_VARIANTS.get(key, tok)))
    return " ".join(parts)


def extract_postcode(address: str, country: str) -> tuple[str, str]:
    """Return (postcode, kind) where kind is zip|pin|postcode|cedex|unknown."""
    cl = (country or "").strip().lower()
    cedex = _CEDEX_RE.search(address or "")
    if cl in {"france", "fr"} or cedex:
        if cedex:
            num = cedex.group(1) or ""
            m = _FR_POST_RE.search(address or "")
            return ((m.group(1) if m else "") + ("X" + num if num else ""), "cedex")
        m = _FR_POST_RE.search(address or "")
        if m:
            return m.group(1), "postcode"
    if cl in {"india", "in"}:
        m = _IN_PIN_RE.search(address or "")
        if m:
            return m.group(1), "pin"
    if cl in {"us", "usa", "united states", "united states of america"}:
        m = _US_ZIP_RE.search(address or "")
        if m:
            return m.group(1), "zip"
    # open-set fallback
    m6 = _IN_PIN_RE.search(address or "")
    if m6:
        return m6.group(1), "pin"
    m5 = _US_ZIP_RE.search(address or "")
    if m5:
        return m5.group(1), "zip"
    return "", "missing"


def extract_house_numbers(address: str) -> list[str]:
    found = []
    seen = set()
    for m in _HOUSE_RE.finditer(address or ""):
        num = m.group(1).lower()
        if num and num not in seen and not (len(num) >= 5 and num.isdigit()):
            # skip bare 5–6 digit postcodes captured as house numbers
            seen.add(num)
            found.append(num)
    return found


def extract_unit(address: str) -> str:
    m = _UNIT_RE.search(address or "")
    return (m.group(1).lower() if m else "")


def extract_directions(address: str) -> set[str]:
    dirs: set[str] = set()
    text = (address or "").lower()
    for m in _DIR_PAREN_RE.finditer(text):
        dirs.add(m.group(1).lower())
    for tok in tokens(text.replace("(", " ").replace(")", " ")):
        if tok in DIRECTION_TOKENS:
            dirs.add(DIRECTION_TOKENS[tok])
    return dirs


def extract_landmarks(address: str) -> list[str]:
    text = (address or "").lower()
    hits: list[str] = []
    for marker in LANDMARK_MARKERS:
        idx = text.find(marker)
        if idx < 0:
            continue
        tail = text[idx + len(marker) :]
        tail = tail.strip(" ,.-")
        words = tokens(tail)[:4]
        if words:
            hits.append(" ".join(words))
    return hits


def digits_in(text: str) -> list[str]:
    return _DIGIT_RE.findall(text or "")


def acronym(core_name: str) -> str:
    toks = [t for t in tokens(core_name) if t.isalpha() and len(t) > 1]
    if len(toks) < 2:
        return ""
    return "".join(t[0] for t in toks)


def build_record_text(
    name: str,
    address: str,
    country: str,
    prefix: str,
    template: str = "{name} | {address} | {country}",
) -> str:
    body = template.format(name=name or "", address=address or "", country=country or "")
    return f"{prefix}{body}"


def normalize_row(
    name: str,
    address: str,
    country: str,
    extra_abbrev: dict[str, str] | None = None,
) -> dict:
    raw_name, raw_addr, raw_country = name or "", address or "", country or ""
    name_n = basic_clean(raw_name)
    addr_n = basic_clean(raw_addr)
    country_n = basic_clean(raw_country)
    name_fold = fold_accents(name_n)
    addr_fold = fold_accents(addr_n)
    name_exp = expand_abbreviations(name_fold, raw_country, extra_abbrev)
    addr_exp = expand_abbreviations(addr_fold, raw_country, extra_abbrev)
    core, legal = extract_legal_form(name_exp)
    house = extract_house_numbers(raw_addr)
    unit = extract_unit(raw_addr)
    postcode, post_kind = extract_postcode(raw_addr, raw_country)
    dirs = extract_directions(raw_addr)
    landmarks = extract_landmarks(raw_addr)
    return {
        "name_raw": raw_name,
        "address_raw": raw_addr,
        "country_raw": raw_country,
        "country_norm": country_n,
        "name_norm": name_n,
        "address_norm": addr_n,
        "name_fold": name_fold,
        "address_fold": addr_fold,
        "name_exp": name_exp,
        "address_exp": addr_exp,
        "core_name": core,
        "legal_form": legal,
        "acronym": acronym(core),
        "house_numbers": "|".join(house),
        "house_number": house[0] if house else "",
        "unit": unit,
        "postcode": postcode,
        "postcode_kind": post_kind,
        "directions": "|".join(sorted(dirs)),
        "landmarks": "|".join(landmarks),
        "addr_digits": "|".join(digits_in(raw_addr)),
        "name_tokens": " ".join(tokens(name_exp)),
        "core_tokens": " ".join(tokens(core)),
        "addr_tokens": " ".join(tokens(addr_exp)),
        "name_len": str(len(name_n)),
        "addr_len": str(len(addr_n)),
        "missing_name": "1" if not name_n else "0",
        "missing_addr": "1" if not addr_n else "0",
        "missing_postcode": "1" if not postcode else "0",
        "missing_house": "1" if not house else "0",
    }


def normalize_frame(
    df: pd.DataFrame,
    extra_abbrev: dict[str, str] | None = None,
) -> pd.DataFrame:
    rows = [
        normalize_row(r.business_name, r.business_address, r.country, extra_abbrev)
        for r in df.itertuples(index=False)
    ]
    extra = pd.DataFrame(rows)
    out = pd.concat([df.reset_index(drop=True), extra], axis=1)
    return out


def mine_abbreviations(
    left: Iterable[str],
    right: Iterable[str],
    min_count: int = 8,
) -> dict[str, str]:
    """Mine token substitutions from aligned matched-pair strings."""
    counts: Counter = Counter()
    for a, b in zip(left, right):
        ta, tb = tokens(basic_clean(a)), tokens(basic_clean(b))
        if not ta or not tb:
            continue
        sa, sb = set(ta), set(tb)
        only_a = sa - sb
        only_b = sb - sa
        if len(only_a) == 1 and len(only_b) == 1:
            x, y = next(iter(only_a)), next(iter(only_b))
            if x > y:
                x, y = y, x
            if x != y and min(len(x), len(y)) <= 6:
                counts[(x, y)] += 1
    mined = {}
    for (x, y), c in counts.items():
        if c >= min_count:
            # expand the shorter token to the longer one
            if len(x) < len(y):
                mined[x] = y
            else:
                mined[y] = x
    return mined


def compute_idf(token_lists: Iterable[list[str]]) -> dict[str, float]:
    dfreq: Counter = Counter()
    n = 0
    for toks in token_lists:
        n += 1
        dfreq.update(set(toks))
    n = max(n, 1)
    return {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in dfreq.items()}


def save_idf(idf: dict[str, float], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(idf), encoding="utf-8")


def load_idf(path: str | Path) -> dict[str, float]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def idf_token_jaccard(a: list[str], b: list[str], idf: dict[str, float]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    inter = sa & sb
    union = sa | sb
    if not union:
        return 0.0
    num = sum(idf.get(t, 1.0) for t in inter)
    den = sum(idf.get(t, 1.0) for t in union)
    return num / den if den else 0.0


def attach_record_texts(df: pd.DataFrame, prefix: str, template: str) -> pd.DataFrame:
    df = df.copy()
    df["text_combined"] = [
        build_record_text(n, a, c, prefix, template)
        for n, a, c in zip(df["name_exp"], df["address_exp"], df["country_raw"])
    ]
    df["text_name"] = [f"{prefix}{n}" for n in df["name_exp"].astype(str)]
    df["text_address"] = [f"{prefix}{a}" for a in df["address_exp"].astype(str)]
    return df
