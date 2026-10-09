#!/bin/bash

# Run panelbeater over and over until it exits 0.
#
# Usage:
#   ./run.sh                      # default panelbeater args
#   ./run.sh --no-train           # anything here is passed straight through
#
# Environment:
#   MAX_ATTEMPTS  give up after this many tries (default: 0 = keep going)
#   DELAY         seconds to wait after a failure (default: 10)
#   MAX_DELAY     ceiling for the backoff (default: 300)
#   NO_BACKOFF    set to 1 to wait a constant DELAY instead of doubling it
#   STOP_CODES    exit codes that end the loop instead of retrying
#                 (default: the deliberate-stop signals, 128+N --
#                  129 HUP, 130 INT/Ctrl-C, 131 QUIT, 137 KILL, 143 TERM)
#                 Crashes are NOT in this list on purpose: 139 (SIGSEGV),
#                 134 (SIGABRT) and friends get retried like any other failure.
#
# Note: no `set -e` here, unlike the other scripts -- this one has to keep
# running after a non-zero exit, since that is the whole point.

MAX_ATTEMPTS="${MAX_ATTEMPTS:-0}"
DELAY="${DELAY:-10}"
MAX_DELAY="${MAX_DELAY:-300}"
NO_BACKOFF="${NO_BACKOFF:-0}"
STOP_CODES="${STOP_CODES:-129 130 131 137 143}"

# Prefer an activated venv (as lint.sh/test.sh assume), but fall back to the
# project venv so this works from a plain shell too.
if command -v panelbeater >/dev/null 2>&1; then
    PANELBEATER="$(command -v panelbeater)"
elif [ -x "$(dirname "$0")/venv/bin/panelbeater" ]; then
    PANELBEATER="$(dirname "$0")/venv/bin/panelbeater"
else
    echo "Error: panelbeater not found on PATH or in ./venv/bin" >&2
    exit 127
fi

attempt=0
delay="$DELAY"
started=$SECONDS

# Without this, Ctrl-C would kill the current run and the loop would cheerfully
# start another one.
on_interrupt() {
    echo ""
    echo "Interrupted after $attempt attempt(s)." >&2
    exit 130
}
trap on_interrupt INT TERM

echo "Running $PANELBEATER until it exits 0..."

while true; do
    attempt=$((attempt + 1))
    echo "--------------------------"
    echo "Attempt $attempt ($(date '+%Y-%m-%d %H:%M:%S'))"

    "$PANELBEATER" "$@"
    code=$?

    if [ "$code" -eq 0 ]; then
        echo "--------------------------"
        echo "Succeeded on attempt $attempt after $((SECONDS - started))s."
        exit 0
    fi

    # 128+N means a signal killed it. Only the deliberate-stop signals should
    # break the loop -- a crash (SIGSEGV from a native lib, say) is exactly the
    # kind of flake worth retrying.
    if [ "$code" -ge 128 ]; then
        signal=$((code - 128))
        name="SIG$(kill -l "$signal" 2>/dev/null || echo "$signal")"

        for stop_code in $STOP_CODES; do
            if [ "$code" -eq "$stop_code" ]; then
                echo "Stopped by $name (exit $code) - not retrying." >&2
                exit "$code"
            fi
        done

        echo "Crashed with $name (exit $code)." >&2
    fi

    if [ "$MAX_ATTEMPTS" -gt 0 ] && [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
        echo "Giving up after $attempt attempts (last exit $code)." >&2
        exit "$code"
    fi

    echo "Failed with exit $code. Retrying in ${delay}s..." >&2
    sleep "$delay"

    if [ "$NO_BACKOFF" != "1" ]; then
        delay=$((delay * 2))
        if [ "$delay" -gt "$MAX_DELAY" ]; then
            delay="$MAX_DELAY"
        fi
    fi
done
