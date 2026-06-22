"""Fixed BACC token alphabet adapter. The BACC codec emits a flat stream over a tiny fixed 33-symbol alphabet (base-16 LEB digits + a REPEAT marker); the model reserves id 0 = PAD (the ``y != 0`` loss mask and ``generate(pad_id=0)`` both assume it), so this adapter shifts program ids ``0..VOCAB-1`` up by one into model space ``1..VOCAB`` and exposes the attributes the model wrapper + predict path read. The alphabet is fixed, so ``encode``/``decode`` are pure round-trip wrappers."""

from preframr_tokens import VOCAB, ids_to_program, program_to_ids

PAD_ID = 0


def _label(model_id):
    """Human-readable label for a model-space id (embedding metadata)."""
    if model_id == PAD_ID:
        return "PAD"
    p = model_id - 1
    if p < 16:
        return f"c{p}"
    if p < 32:
        return f"t{p - 16}"
    return "REP"


class BaccTokenizer:
    """Adapter between BACC program ids and model-space ids (PAD-shifted)."""

    def __init__(self):
        self.VOCAB = VOCAB
        self.n_vocab = VOCAB + 1
        self.n_words = self.n_vocab
        self.tokens = [_label(i) for i in range(self.n_vocab)]
        self.tkmodel = None

    def token_metadata(self):
        """Per-id labels (length ``n_vocab``); used as embedding metadata."""
        return list(self.tokens)

    def encode(self, program):
        """BaccProgram -> model-space id list (program ids shifted up by one)."""
        return [i + 1 for i in program_to_ids(program)]

    def decode(self, ids, driver="generic"):
        """Model-space id list -> BaccProgram; PAD and out-of-range ids are dropped before the shift-down, so a padded stream round-trips, while a truncated/invalid generated stream raises (``ids_to_program`` indexes past the end) for callers to handle. ``driver`` defaults to ``generic`` -- the framework trains on sid-only generic-driver streams; pass the matching driver to round-trip a program serialized under a different backend."""
        prog_ids = [int(i) - 1 for i in ids if 1 <= int(i) <= self.VOCAB]
        return ids_to_program(prog_ids, driver=driver)
