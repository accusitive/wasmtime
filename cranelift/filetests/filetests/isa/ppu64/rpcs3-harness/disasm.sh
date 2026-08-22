#!/bin/bash
# Disassemble CLIF-compiled PPU code using the SDK's own objdump.
#   ./disasm.sh <clif-file> [function-name]
set -e
CLIF_UTIL=/media/mass/data/p/wasmtime/target/debug/clif-util
NAME=${2:-testfn}
WORK=$(mktemp -d)
trap "rm -rf $WORK" EXIT

BYTES=$($CLIF_UTIL compile -D --target powerpc64-unknown-linux-gnu "$1" 2>/dev/null | grep '^\.byte' | head -1 | sed 's/^\.byte //')
[ -n "$BYTES" ] || { echo "COMPILE FAILED"; exit 2; }

cat > "$WORK/d.s" <<EOF
	.text
	.globl $NAME
$NAME:
	.byte $BYTES
EOF
scp -q "$WORK/d.s" agentvm:d.s
ssh agentvm "WINEDEBUG=-all wine ~/PS3SDK/cell/host-win32/ppu/bin/ppu-lv2-gcc.exe -c d.s -o d.o" >/dev/null 2>&1
ssh agentvm "WINEDEBUG=-all wine ~/PS3SDK/cell/host-win32/ppu/bin/ppu-lv2-objdump.exe -d d.o" 2>/dev/null | grep -v XDG | sed -n '5,$p'
