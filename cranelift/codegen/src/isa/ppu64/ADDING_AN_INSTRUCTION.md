# Adding a new instruction to the ppu64 backend

This is the end-to-end checklist for adding a machine instruction (or a
CLIR instruction that needs lowering) to the `ppu64` backend. It also
serves as a tour of the backend's structure.

## 0. Verify the encoding FIRST

Before writing any Rust/ISLE, get the exact opcode encoding from the PS3
SDK toolchain — never from memory. PowerPC has several field-layout
families and they do not mix:

| Family | Layout | Examples |
|---|---|---|
| Arithmetic X-form | dest in RT slot (`add rT, rA, rB`) | add, subf, mulld, divd, neg |
| Logical / shift X-form | dest in RA slot (`and rA, rS, rB`) | and/or/xor/nor family, slw/srw/sld/srd/sraw/srad |
| Extend X-form | source in RT slot (`extsw rA, rS`) | extsb, extsh, extsw, cntlzd/cntlzw |
| A-form (FPU) | RT, RA, RB, RC + 5-bit XO | fadd/fsub/fdiv; **fmul takes its second source in the C field** |

Probe procedure:

1. Write a `.s` snippet using the instruction with distinct register
   numbers (avoid same-register ambiguity), e.g.
   `cntlzd 3, 4`.
2. Copy it to the SDK VM and assemble:
   `ssh agentvm "WINEDEBUG=-all wine ~/PS3SDK/cell/host-win32/ppu/bin/ppu-lv2-gcc.exe -c probe.s -o probe.o"`
3. Disassemble with the SDK objdump and decode the bytes programmatically
   (see the decode script pattern in git history):
   fields are `(word>>21)&31`, `(word>>16)&31`, `(word>>11)&31`,
   XO = `(word>>1)&1023` (or `&63` for A-form), Rc = `word&1`.

Record the verified numbers as constants in `inst/emit.rs` (`mod xo`,
`mod opcode`, or inline tails for oddball forms).

## 1. Define the machine instruction (ISLE)

`cranelift/codegen/src/isa/ppu64/inst.isle`

- Add a variant to the `MInst` enum (or an operand enum like
  `AluOPRRR`) with named fields: destination first as `WritableReg`,
  sources as `Reg`.
- If it needs a new immediate or flag type, declare it
  (`(type Foo (primitive Foo))`) and define the Rust type in
  `inst/imms.rs` or wherever it belongs.

## 2. Teach regalloc about the operands

`inst/mod.rs`, in the big match inside
`fn riscv_get_operands(...)`:

```rust
Inst::Foo { rd, rs1, rs2 } => {
    collector.reg_use(rs1);
    collector.reg_use(rs2);
    collector.reg_def(rd);
}
```

Get this wrong and regalloc2 either panics with SSA errors or silently
misallocates. Rules of thumb:

- reads *before* writes: `reg_use`; if the def may overlap a use in the
  same instruction you must split into two temps at the ISLE level — a
  single vreg may never be both used and defined by one instruction
  (regalloc2 SSA violation).
- fixed registers: `reg_fixed_use` / `reg_fixed_def`.
- clobbers (calls): `collector.reg_clobbers(...)`.

## 3. Emit bytes + disassembly

`inst/emit.rs`:

1. Add an `encode()` helper on the operand enum, or an arm in
   `MachInstEmit::emit` that calls the right `enc_*` helper and
   `put_word(sink, ...)`.
2. Add the matching arm to `format_inst()` so filetests print readable
   text.

## 4. Lower CLIF to it (ISLE)

`lower.isle`:

- Declare extern constructors for anything implemented as a Context
  method in `lower/isle.rs`:
  `(decl gen_foo (...) RetType)` + `(extern constructor gen_foo gen_foo)`.
  Pure helpers can stay in ISLE; anything multi-step or conditional is
  usually clearer as a Rust constructor.
- Add `(rule ...)` patterns matching the generated `clif_lower.isle`
  term shapes. Check exact argument lists there — most polymorphic ops
  carry a leading type argument (e.g. `iadd` is `(Type Value Value)`).
- Watch ISLE sequencing: a `let` body is ONE term; chain side effects
  through nested lets or `_ Unit (...)` bindings. Constructor arguments
  evaluate eagerly, so emit order = binding order.

## 5. If the ABI needs it

`abi.rs` only needs changes for stack-layout-affecting things. Anything
that expands to multiple instructions at prologue time uses existing
helpers (`gen_sp_reg_adjust`, `gen_load_base_offset`, ...). Remember the
16-byte scratch area at `SPOffset(0..16)` is reserved for GPR<->FPR bit
moves — don't overlap it.

## 6. Build + iterate on errors

    cargo build -p cranelift-tools --bin clif-util --features all-arch

Common build failures, in typical order:

1. ISLE parse/type errors (build script output shows file:line).
2. Overlapping ISLE rules -> add rule priorities (`rule -1`, `rule 0`...).
3. Missing `Context` trait methods -> implement them in `lower/isle.rs`.
4. Non-exhaustive matches in `get_operands` / `emit` / `format_inst`.

## 7. Inspect codegen locally

    ./target/debug/clif-util compile -D --target powerpc64-unknown-linux-gnu x.clif

or, for ground-truth disassembly through the SDK's own objdump:

    ~/data/p/ps3game/disasm.sh x.clif

Prefer the SDK disassembler over capstone: capstone prints some X-form
operand pairs in display order rather than field order, which cost us a
real bug (shift family layout).

## 8. Filetest (compile-only)

Add `cranelift/filetests/filetests/isa/ppu64/<topic>.clif`:

```
test compile precise-output
target powerpc64-unknown-linux-gnu

function %foo(i64, i64) -> i64 system_v {
block0(v0: i64, v1: i64):
  ... ; CLIF here
}

; VCode:
; block0:
;   <expected vcode text>
;
; Disassembled:
; block0: ; offset 0x0
;   <expected disasm>
```

Generate the expectation with `CRANELIFT_TEST_BLESS=1`, then re-run
without it. Run: `clif-util test cranelift/filetests/filetests/isa/ppu64/`.

## 9. Semantic validation under RPCS3 (execution)

Compile-only checks cannot catch wrong encodings that happen to
disassemble plausibly — the shift-family bug survived several rounds of
filetests. For anything semantic:

1. Add a case to `~/data/p/ps3game/cases/`: `<name>.clif` defining
   `%t() -> i64` with constants baked in, plus `<name>.exp` holding the
   expected result as hex i64 (float results via `bitcast.i64`).
2. Regenerate the manifest and run:

       cd ~/data/p/ps3game
       for f in cases/*.clif; do b=$(basename $f .clif); \
         echo "$(readlink -f $f) $(cat cases/$b.exp)"; done > suite.manifest
       ./run.sh suite.manifest

   Exit status = number of failing cases; TTY lines stream per-case
   PASS/FAIL live. Budget ~90 s (RPCS3 warmup dominates).

## 10. Commit

One commit per logical feature, including: backend files, filetests,
harness cases, and any shared-code touches (e.g. new `Reloc` variants,
capstone features).
