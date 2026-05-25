#!/bin/bash
# Live playback on a Jetson host: same shape as render.sh but uses
# the Jetson docker image and the ALSA path (jetsons typically don't
# run pulse). Pass --audio-device <plug> via $* if the system default
# isn't right; --reglog <path> is required.

set -o noglob

IMG=anarkiwi/preframr-jetson

exec docker run --rm --name preframr-render \
    -v /scratch:/scratch \
    --device /dev/snd \
    --device /dev/snd/seq \
    -ti "${IMG}" \
    python3 -m preframr_tokens.render_play --via-wav --animate $*
