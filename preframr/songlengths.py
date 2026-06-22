"""Per-subtune frame budget from the HVSC Songlengths.md5, for the sid-only codec
path (``recover_from_sid`` needs an explicit frame count). Cycle math mirrors
headlessvice ``vsiddump.py`` so the budget matches the codec's gate fixtures."""

import functools
import hashlib

PAL_PHI = 985248
NTSC_PHI = 1022727
PAL_CPF = 19656
NTSC_CPF = 17095


def sid_md5(sid_path):
    """Lowercase hex MD5 of the .sid bytes (the Songlengths.md5 key)."""
    with open(sid_path, "rb") as handle:
        return hashlib.md5(handle.read()).hexdigest().lower()


@functools.lru_cache(maxsize=8)
def _index(songlengths_path):
    """Map md5 -> per-subtune time-token list (cached per file)."""
    out = {}
    with open(songlengths_path, encoding="utf-8") as handle:
        for line in handle:
            if "=" not in line or line.startswith((";", "[")):
                continue
            md5, _, times = line.strip().partition("=")
            out[md5.lower()] = times.split()
    return out


def _token_seconds(token):
    """``[H:]M:S[.mmm]`` -> seconds (fraction after ``.`` is milliseconds)."""
    base = token
    fraction = 0.0
    if "." in token:
        base, ms = token.split(".", 1)
        fraction = float(ms) / 1e3
    total = 0
    for part in base.split(":"):
        total = total * 60 + int(part)
    return total + fraction


def subtune_frames(sid_path, subtune, songlengths_path, ntsc=False):
    """Frame budget for ``subtune`` (1-based): ``cycles // cpf``."""
    times = _index(songlengths_path).get(sid_md5(sid_path))
    if not times or subtune < 1 or subtune > len(times):
        raise KeyError(f"no Songlengths entry for {sid_path} subtune {subtune}")
    phi, cpf = (NTSC_PHI, NTSC_CPF) if ntsc else (PAL_PHI, PAL_CPF)
    return int(phi * _token_seconds(times[subtune - 1])) // cpf
