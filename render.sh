#!/bin/bash
# Live playback of a SID dump via preframr_tokens.render_play. Replaces the
# legacy render.py offline path; supersedes the LM-driven prompt
# extraction with a direct dump-parse-and-play pipeline.
#
# Default mode: --via-wav --paplay --animate -- the working host
# recipe on Linux + PulseAudio. Pass extra args (e.g. --reglog <path>,
# --audio-device, --keep-wav <out.wav>) via $*.

set -o noglob

IMG=anarkiwi/preframr
PULSE_SOCK=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/pulse/native

PULSE_ARGS=()
if [ -S "${PULSE_SOCK}" ]; then
    PULSE_ARGS=(-e "PULSE_SERVER=unix:${PULSE_SOCK}" -v "${PULSE_SOCK}:${PULSE_SOCK}")
fi

exec docker run --rm --name preframr-render \
    -v /scratch:/scratch \
    --device /dev/snd \
    "${PULSE_ARGS[@]}" \
    -ti "${IMG}" \
    python3 -m preframr_tokens.render_play --via-wav --paplay --animate $*
