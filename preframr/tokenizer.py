"""Fixed BACC token alphabet adapter. The BACC codec emits a flat stream over the
flat v2 typed alphabet (:mod:`preframr_tokens.bacc.flat_serialize`): contiguous id
RANGES encode an atom's KIND (structural / NOTE_* / INSTR_REF_* / CMD_* / BYTE_*),
no place value and no inline LZ. The model reserves id 0 = PAD (the ``y != 0`` loss
mask and ``generate(pad_id=0)`` both assume it), so this adapter shifts program ids
``0..VOCAB-1`` up by one into model space ``1..VOCAB`` and exposes the attributes the
model wrapper + predict path read. The alphabet is fixed, so ``encode``/``decode``
are pure round-trip wrappers."""

from preframr_tokens import VOCAB, ids_to_program, program_to_ids

PAD_ID = 0


def _flat_label(program_id):
    """Human-readable label for a flat v2 *program-space* id (0..VOCAB-1).

    The flat alphabet is typed by id range: the structural / control block carries
    individually-named tokens (BOS, ROW, PATTERN_BEGIN, ...), and the value ranges
    are NOTE_* (one id per A440 grid index + the REST/KEYOFF/KEYON/RAW markers),
    INSTR_REF_* (one id per instrument ordinal), CMD_* (one id per command) and
    BYTE_* (one id per 0..255 byte value). Labels are derived from the codec's own
    range constants so they cannot drift from the alphabet."""
    from preframr_tokens.bacc import flat_serialize as f

    if f.BYTE_BASE <= program_id < f.BYTE_BASE + f.BYTE_SPAN:
        return f"BYTE_{program_id - f.BYTE_BASE}"
    if f.CMD_BASE <= program_id < f.CMD_BASE + f.CMD_SPAN:
        return f"CMD_{program_id - f.CMD_BASE}"
    if f.INSTR_REF_BASE <= program_id < f.INSTR_REF_BASE + f.INSTR_REF_SPAN:
        return f"INSTR_REF_{program_id - f.INSTR_REF_BASE}"
    if f.NOTE_BASE <= program_id < f.NOTE_BASE + f.NOTE_SPAN:
        # Pitched grid index relative to A440 (n=0=A4); the non-pitch markers
        # (REST/KEYOFF/KEYON/RAW) live at fixed ids inside the structural block.
        return f"NOTE_{program_id - f.NOTE_ZERO:+d}"
    # Structural / control block: recover the constant NAME for this id.
    for name in dir(f):
        if (
            name.isupper()
            and getattr(f, name) == program_id
            and name
            not in (
                "VOCAB",
                "PAD_ID",
                "NREG",
                "NOTE_BASE",
                "NOTE_SPAN",
                "NOTE_ZERO",
                "NOTE_MIN",
                "NOTE_MAX",
                "INSTR_REF_BASE",
                "INSTR_REF_SPAN",
                "CMD_BASE",
                "CMD_SPAN",
                "BYTE_BASE",
                "BYTE_SPAN",
            )
        ):
            return name
    return f"RESERVED_{program_id}"


def _label(model_id):
    """Human-readable label for a model-space id (embedding metadata)."""
    if model_id == PAD_ID:
        return "PAD"
    return _flat_label(model_id - 1)


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
