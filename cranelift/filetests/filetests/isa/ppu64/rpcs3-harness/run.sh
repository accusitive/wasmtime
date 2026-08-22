#!/bin/bash
# Build + run a batch of ppu64 tests under RPCS3.
#   ./run.sh <manifest>     -> exit 0 iff all cases pass
#
# Sentinel: RPCS3 logs "_sys_process_exit(status=N)" (continuously flushed)
# when the guest exits; N = number of failed cases. TTY lines are streamed
# for per-case detail, and rpcs3 is SIGKILLed once the sentinel lands.
set -e

killrpc() {
    [ -n "$(pidof rpcs3)" ] && kill -9 $(pidof rpcs3) 2>/dev/null
    [ -n "$(pidof rpcs3.AppRun)" ] && kill -9 $(pidof rpcs3.AppRun) 2>/dev/null
    return 0
}

HERE=/home/gl/data/p/ps3game
cd "$HERE"
MANIFEST=$(readlink -f "$1")
OUT=$HERE/work
rm -rf "$OUT" && mkdir -p "$OUT"

python3 "$HERE/build.py" "$MANIFEST" "$OUT"

GCC="WINEDEBUG=-all wine ~/PS3SDK/cell/host-win32/ppu/bin/ppu-lv2-gcc.exe"
FSELF="WINEDEBUG=-all wine ~/PS3SDK/cell/host-win32/bin/make_fself.exe"

scp -q "$OUT/main.c" "$OUT/testfns.s" agentvm:
ssh agentvm "$GCC main.c testfns.s -o t.elf" >"$OUT/build.log" 2>&1 \
    || { echo "BUILD FAILED"; grep -v XDG "$OUT/build.log" | head -8; exit 3; }
ssh agentvm "$FSELF t.elf t.self" >"$OUT/fself.log" 2>&1 \
    || { echo "FSELF FAILED"; grep -v XDG "$OUT/fself.log" | head -5; exit 4; }
scp -q agentvm:t.self "$OUT/ppu_tests.self"
echo "built $(stat -c%s $OUT/ppu_tests.self) bytes; launching RPCS3..."

TTY=~/.cache/rpcs3/TTY.log
RLOG=~/.cache/rpcs3/RPCS3.log

killrpc
sleep 2
: > "$TTY"
set +e
rpcs3 --headless "$OUT/ppu_tests.self" >/dev/null 2>&1 &

STATUS=""
seen=0
ever_seen=0
for i in $(seq 1 200); do
    running=$(pidof rpcs3 | awk '{print $1}')
    [ -n "$running" ] && ever_seen=1

    if [ -f "$TTY" ]; then
        n=$(wc -l < "$TTY" 2>/dev/null | tr -d ' '); n=${n:-0}
        if [ "$n" -gt "$seen" ]; then
            tail -n $((n - seen)) "$TTY" | grep -v '^$'
            seen=$n
        fi
    fi

    if [ -f "$RLOG" ] && grep -q "_sys_process_exit(status=" "$RLOG" 2>/dev/null; then
        STATUS=$(grep "_sys_process_exit(status=" "$RLOG" | tail -1 \
            | sed 's/.*status=\([0-9]*\).*/\1/')
        sleep 2   # let the final TTY flush land
        killrpc
        echo "[guest exited with status $STATUS -> killed rpcs3]"
        break
    fi

    # Only trust "rpcs3 died" after we've actually seen it running.
    if [ "$ever_seen" = "1" ] && [ -z "$running" ]; then
        STATUS=$(grep "_sys_process_exit(status=" "$RLOG" 2>/dev/null \
            | tail -1 | sed 's/.*status=\([0-9]*\).*/\1/')
        break
    fi
    sleep 1
done

killrpc

if [ "$STATUS" = "0" ]; then
    echo "ALL PASSED"
    exit 0
else
    echo "RESULT: failures=${STATUS:-UNKNOWN}"
    exit 1
fi
