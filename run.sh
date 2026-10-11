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
#   PYTHONFAULTHANDLER  defaults to 1, so a SIGSEGV/SIGABRT/SIGBUS/SIGFPE/SIGILL
#                 dumps the Python traceback of every thread to stderr before
#                 the process dies. Set it to empty to turn this off.
#   HANG_TIMEOUT  seconds with no output before a run counts as hung
#                 (default: 1200; 0 turns the watchdog off). A hung run gets
#                 its stacks dumped with py-spy, is killed, and is retried.
#   HANG_POLL     how often the watchdog checks, in seconds (default: 5)
#   HANG_DIR      where stack dumps go (default: ./hangs next to this script)
#   PY_SPY        the py-spy command (default: py-spy). Attaching to another
#                 process needs root on macOS, and on Linux whenever
#                 kernel.yama.ptrace_scope >= 1, so use e.g.
#                 PY_SPY="sudo -n py-spy" with a passwordless sudo rule.
#
# Note: no `set -e` here, unlike the other scripts -- this one has to keep
# running after a non-zero exit, since that is the whole point.

MAX_ATTEMPTS="${MAX_ATTEMPTS:-0}"
DELAY="${DELAY:-10}"
MAX_DELAY="${MAX_DELAY:-300}"
NO_BACKOFF="${NO_BACKOFF:-0}"
STOP_CODES="${STOP_CODES:-129 130 131 137 143}"
HANG_TIMEOUT="${HANG_TIMEOUT:-1200}"
HANG_POLL="${HANG_POLL:-5}"
HANG_DIR="${HANG_DIR:-$(dirname "$0")/hangs}"
PY_SPY="${PY_SPY:-py-spy}"

# Crashes in native code (catboost, libgomp, torch...) otherwise die silently.
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER-1}"
# The watchdog judges liveness by output, and Python block-buffers stdout
# into a pipe, which would make a busy run look hung.
export PYTHONUNBUFFERED=1

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

if [ "$HANG_TIMEOUT" -gt 0 ] && ! command -v ${PY_SPY##* } >/dev/null 2>&1; then
    echo "Warning: ${PY_SPY##* } not found - hung runs will be killed without a stack dump." >&2
    echo "         Install it with: pip install py-spy" >&2
fi

# Seconds since a file was last written (GNU stat, then BSD stat).
file_age() {
    local mtime
    mtime=$(stat -c %Y "$1" 2>/dev/null || stat -f %m "$1")
    echo $(($(date +%s) - mtime))
}

attempt=0
delay="$DELAY"
started=$SECONDS
pid=""
log=""

# Without this, Ctrl-C would kill the current run and the loop would cheerfully
# start another one. The run is a background job, which ignores SIGINT in a
# non-interactive shell, so it has to be stopped explicitly.
on_interrupt() {
    echo ""
    echo "Interrupted after $attempt attempt(s)." >&2
    [ -n "$pid" ] && kill "$pid" 2>/dev/null
    [ -n "$log" ] && rm -f "$log"
    exit 130
}
trap on_interrupt INT TERM

echo "Running $PANELBEATER until it exits 0..."

while true; do
    attempt=$((attempt + 1))
    echo "--------------------------"
    echo "Attempt $attempt ($(date '+%Y-%m-%d %H:%M:%S'))"

    # Run in the background, teeing output to a file whose mtime tells the
    # watchdog when the run last printed anything.
    log="$(mktemp "${TMPDIR:-/tmp}/panelbeater-run.XXXXXX")"
    "$PANELBEATER" "$@" > >(tee "$log") 2>&1 &
    pid=$!
    hung=0

    while kill -0 "$pid" 2>/dev/null; do
        # Backgrounded so a Ctrl-C trap fires now, not after the sleep.
        sleep "$HANG_POLL" &
        wait $!

        [ "$HANG_TIMEOUT" -gt 0 ] || continue
        kill -0 "$pid" 2>/dev/null || break
        age=$(file_age "$log")
        [ "$age" -ge "$HANG_TIMEOUT" ] || continue

        mkdir -p "$HANG_DIR"
        dump="$HANG_DIR/hang-$(date '+%Y%m%d-%H%M%S')-attempt$attempt.txt"
        echo "" >&2
        echo "No output for ${age}s - looks hung. Dumping stacks to $dump" >&2
        {
            echo "=== py-spy dump --native --pid $pid ($(date '+%Y-%m-%d %H:%M:%S')) ==="
            $PY_SPY dump --native --pid "$pid" 2>&1
            echo ""
            echo "=== last 50 lines of output ==="
            tail -n 50 "$log"
        } > "$dump"
        cat "$dump" >&2

        kill -KILL "$pid" 2>/dev/null
        hung=1
        break
    done

    wait "$pid"
    code=$?
    pid=""
    rm -f "$log"
    log=""

    if [ "$code" -eq 0 ]; then
        echo "--------------------------"
        echo "Succeeded on attempt $attempt after $((SECONDS - started))s."
        exit 0
    fi

    # The watchdog's own SIGKILL must not count as a deliberate stop.
    if [ "$hung" -eq 1 ]; then
        echo "Killed hung run (stacks in $dump)." >&2
    # 128+N means a signal killed it. Only the deliberate-stop signals should
    # break the loop -- a crash (SIGSEGV from a native lib, say) is exactly the
    # kind of flake worth retrying.
    elif [ "$code" -ge 128 ]; then
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
