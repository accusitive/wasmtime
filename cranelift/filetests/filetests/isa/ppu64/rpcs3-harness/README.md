# ppu64 RPCS3 cross-run harness

Executes compiled PPU code under RPCS3's Cell PPU emulation and checks
results against expected values.

## Requirements

- A PS3 SDK toolchain reachable as `ssh agentvm` with
  `cell/host-win32/ppu/bin/ppu-lv2-gcc.exe` (run under wine) and
  `cell/host-win32/bin/make_fself.exe`.
- RPCS3 installed locally (`rpcs3 --headless <file.self>`), with firmware.
- `clif-util` built with `--features all-arch`.

## Usage

    ./run.sh suite.manifest

Manifest lines: `<path.clif> <expected-i64-in-hex>`. Each case is a CLIF
function `%t() -> i64 system_v` computing a value with constants baked in;
float results are observed via `bitcast.i64`.

The harness compiles each case to machine code with the ppu64 backend,
wraps them in a C driver that calls each `tfN()` and compares, links via
the SDK (ELFv1 descriptors + `make_fself`), boots the `.self` in RPCS3,
and polls RPCS3's log for the guest's exit status (= number of failures).
TTY output is streamed for per-case detail; rpcs3 is SIGKILLed once the
sentinel lands.

## Debugging aids

- `./disasm.sh <clif>` disassembles a case through the SDK's own objdump
  (ground truth; note capstone prints X-form rotate/shift operands in a
  different order than the encoding fields).

## Suite manifests

- `suite2.manifest` — 24 scalar/integer/memory cases
- `fpu2.manifest` — 19 float cases: saturating conversions (overflow,
  underflow, NaN), synthesized floor/ceil/trunc with big-input and NaN
  guards, fmin/fmax, i32<->f32 bitcasts

Combine them (`cat suite2.manifest fpu2.manifest > all.manifest`) for a
single ~90 s full-suite run.

## Known limitations

- `fnearest` is not implemented (round-half-even synthesis is fiddly
  without fri* instructions).
- `fcvt_to_sint_sat.i32` / `fcvt_to_uint_sat.i32` are not implemented.
- `br_table`, atomics, and vectors are unimplemented.
