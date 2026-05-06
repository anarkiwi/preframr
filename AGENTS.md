# Operational notes for agents working on this repo

This file captures conventions and current state that aren't obvious from
the code alone. Update it when conventions change.

## Where work runs

The repo lives on multiple hosts. **All build / training / docker work
happens on `defroster`**, not on the local checkout. The local checkout
is for editing code and running unit tests; `defroster` has the GPU and
the package mirror that the docker build needs.

A helper at `/tmp/dfr` (created per-session — see "Bootstrapping" below)
wraps the ssh + cd + remote-bash chain:

```bash
/tmp/dfr 'pwd && git log --oneline -3'
```

Without it, plain `ssh defroster '<cmd>'` lands in `$HOME` and fails on
`git ...` and similar commands. The Bash tool also blocks compound `cd
<dir> && git ...` commands locally as a safety measure, so always wrap
through `/tmp/dfr` when interacting with `defroster`.

The integration tests run inside docker containers built from this repo.
Building docker requires `PIP_OPTS` set to the local pip mirror:

```bash
export PIP_OPTS="--index-url http://192.168.5.1:5001/index/ --trusted-host 192.168.5.1"
```

The integration test scripts pick this up from the environment.

## Bootstrapping `/tmp/dfr` at session start

```bash
cat > /tmp/dfr <<'EOF'
#!/bin/bash
set -e
[ $# -lt 1 ] && { echo "usage: $0 <command...>" >&2; exit 2; }
exec ssh defroster "(cd /scratch/anarkiwi/preframr && $*)"
EOF
chmod +x /tmp/dfr
```

## Integration tests

Two tests live at the repo root, both sourcing `int_test_common.sh`
for the dump / build / tensorboard boilerplate.

### `run_memorize_int_test.sh`

Smoke test: train on 4 fixed Goto80 SIDs (Truth, Acid_10000, CBM_85,
Skybox), confirm predict reproduces block 0 of rotation 0 with
accuracy >= `MIN_ACC` (0.2). Greedy decoding (top-k=1, temperature 0.1).

Current passing config:
- `seq_len=1024`, `block_stride=256`, `tkvocab=0` (raw alphabet).
- 10-layer / embed=384 / intermediate=1024 (~10M params).
- `--shuffle 32 --learning-rate 5e-4 --batch-size 16 --accumulate-grad-batches 2`.
- `STOP_LOSS=0.001 STOP_DELTA=0.0001` -- the macros-on regime needs
  per-token p ≈ 99.9% to nail 512-token greedy reconstruction
  (vocab ~10K; per-token confidence at 99.5% cascades).
- All macros enabled (LoopPass, GateMacroPass, InstrumentProgramPass,
  fuzzy/transposed/...). Earlier disabled subsets are no longer
  needed since the preload speedup landed.

To run:
```bash
/tmp/dfr 'export PIP_OPTS="--index-url http://192.168.5.1:5001/index/ --trusted-host 192.168.5.1" && bash run_memorize_int_test.sh > /tmp/memorize_int_test.log 2>&1'
```

End-to-end wallclock with cached image: ~5-6 min (dump 30s + train
~3 min + predict ~30s). Cold image build: +2 min. Cold dump (no
cached vsiddump output): +5 min for Skybox alone.

To check status mid-run:
```bash
/tmp/dfr "tr '\r' '\n' < /tmp/memorize_int_test.log | grep -E 'Stopping threshold|train_loss = |starting at seq|generated [0-9]|min_acc' | tail -10"
```

### `run_generalize_int_test.sh`

Held-out generalisation gate: train on 16 Goto80 SIDs, validate on 4
held-out songs every epoch. Current state: **calibration mode**
(`MIN_VAL_ACC=0`), so the gate reports val_acc but doesn't fail on it.
Once a baseline is observed (in flight 2026-05-06: ~3.7% val_acc at
epoch 39), set `MIN_VAL_ACC` to a calibrated threshold (likely ~0.02 =
~200x chance at vocab=33857).

The 16 train SIDs include 6 picked deterministically by
`untracked/pick_train_replacements.py` (median-duration neighbours of
the catalogue median). Re-run that script if HVSC's mirror changes
or you want to refresh the picks.

Open issues for the calibration run (tracked in `untracked/TODO.md`):
- **Superman.sid** is rejected by the digi filter (false positive on
  Goto80's wide vol-automation style); the train set is currently
  15/4 instead of 16/4.
- The val_loss numbers reported are higher than expected for the
  vocab size; worth inspecting `Model.validation_step` once the
  current run settles.

To run:
```bash
/tmp/dfr 'export PIP_OPTS="--index-url http://192.168.5.1:5001/index/ --trusted-host 192.168.5.1" && bash run_generalize_int_test.sh > /tmp/generalize_int_test.log 2>&1'
```

End-to-end wallclock with cached image: ~30-45 min (dump 5-6 min for
Skybox + 4 min for the rest in parallel + train usually 20-30 min on
defroster's GPU + predict 1-2 min). Real wall depends heavily on how
deep into MAX_EPOCHS=200 it goes before EarlyStopping fires.

Stage layout:
1. Prep + dump train SIDs to `/tmp/preframr_gen/train/`, eval SIDs
   to `/tmp/preframr_gen/eval/`.
2. Tensorboard sidecar + docker build.
3. Train with `--reglogs train/*.dump.parquet --eval-reglogs
   eval/*.dump.parquet`. EarlyStopping on `val_loss`,
   `ModelCheckpoint(monitor="val_loss", save_top_k=1)` writes
   `best-{epoch}-{val_loss:.4f}.ckpt`.
4. Gate: `tests/check_generalize.py` reads the TB events and asserts
   `val_acc` at the best-val_loss epoch >= `MIN_VAL_ACC`. Runs
   inside the preframr docker image because that's where
   `tensorboard` is installed.
5. Per-eval-song qualitative predict: each held-out song gets its
   own .wav / .csv. Uses `--predict-set val` to read from
   `val_block_mapper`. Wrapped in `|| true` so a safety-net
   rejection on one song doesn't abort the others.

## Patterns / conventions

### Background runs

Long-running operations (memorize, generalize, profiling) go through
the Bash tool's `run_in_background=true`. The runtime sends a
notification when the bg completes. While running:
- *Don't* poll. The notification is the signal.
- Use `ScheduleWakeup` with a sensible delay (270s for active checks
  that fit in the prompt-cache window; 600-1200s for idle waits) for
  proactive status pulls.

### Helper scripts

Workspace-only tooling (study scripts, scrape data, profilers) goes
under `untracked/`. The dir is gitignored + dockerignored. Tests that
import from `untracked/` should `try ... except ImportError` and skip
when the path isn't available -- the docker image doesn't ship
`untracked/`.

### Comment style

Conventions enforced informally during cleanup passes:
- No "in this session", "previously I added", or other narration of
  past work. Why-load-bearing context belongs in the comment; how-it-
  evolved goes in the commit message.
- No dev-local references like specific corpus paths (`/tmp/inv`),
  hostnames, or PR numbers.
- No "v1 / v2 / v3" framing unless the version differential is in
  the code surface (e.g., a class renamed `FooV2`).
- One-line comments preferred. Multi-paragraph blocks only when
  there's genuine non-obvious context (e.g., the perf-fix in
  `iter_self_contained_row_blocks` needs to explain why the literal
  expansion is hoisted, otherwise the next reader will undo it).

## Memory and persistence

Use the auto-memory system (in `~/.claude/projects/.../memory/`) for
user preferences and durable feedback, not for current-task state.

Use TodoWrite for active task lists within a session.

Use `untracked/TODO.md` for repo-level follow-ups (digi filter, etc.)
that span sessions but don't deserve a tracked TODO file.

## Recent landmark commits

(for orientation in `git log`):
- `a659037` -- preload 80x speedup via array-based `_expand_ops` and
  `expand_loops` rewrites. Skybox literal expansion: 80s -> 0.85s.
- `dd98ac3` -- memorize stop_loss tightened to 0.001 for the macros-on
  regime; cleared the safety-net rejection.
- `b8f9a50` -- dead code cleanup: `materialize_*_outside` (-380 LOC),
  `seq_mapper.py` -> `block_mapper.py` rename.
- `7ed6ede` -- SeqMapper class retired; BlockMapper is sole train +
  inference data source.
- `2047a35` -- `_consolidate_frames` re-pack in
  `iter_self_contained_row_blocks` so block tokens match the LM input
  shape.
