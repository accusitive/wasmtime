# Inline assembly in Cranelift

Status: working implementation for x86-64 with GCC-style operand
constraints, assembled at emission time by the vendored `glasm` encoder.
This file documents the design, the current supported surface, and what is
still missing from full GCC/Clang parity.

## Overview

A new CLIF instruction:

```
v2: i64 = inline_asm asm0, v0, v1
```

`asm0` is an `AsmRef` entity (stored in `Function.dfg.inline_asms`) holding
an [`InlineAsmContents`]: the template text, one GCC-style constraint
string per operand (outputs first, then inputs), the clobber list, and the
assembler dialect. The instruction takes the input values as variable
arguments and produces one `i64` result per output constraint.

The x64 backend lowers the statement to a `MInst::InlineAsm` pseudo-instruction
that carries the resolved template plus its machine operands. Registers are
assembled into final machine code at *emission time*, after register
allocation — the same architecture LLVM uses — because the physical
registers substituted into the template are only known then.

## Text syntax (filetests)

The CLIF reader parses the statement with GCC's colon-separated groups:

```
v1 = inline_asm "movq %1, %0 ; addq $1, %0" : "=r" : "r"(v0) : "cc"
v2 = inline_asm "addq %0, %0" : "+a"(v0)                     ; inout
v3 = inline_asm intel "add rax, rax" : "+a"(v0)              ; Intel dialect
```

- Template: AT&T syntax by default; the `intel` marker switches dialects.
  Operand placeholders are `%N` (outputs first, then inputs); `%%` escapes
  a literal percent sign. A `%` followed by anything else (e.g. a register
  like `%rax`) is left as-is.
- Outputs: constraint strings; a `+` (inout) output may carry `(value)`,
  the value it reads.
- Inputs: `"constraint"(value)` pairs.
- Clobbers: register names (any width alias: `rax`, `eax`, `al`) or the
  pseudo-resources `cc` and `memory`.

## Constraints

Supported: `=`, `+`, `&` (accepted, only meaningful with matching
constraints, which are not yet supported); the register selections
`r` (any GPR, with the class subsets `q`/`Q`/`R`/`l` widened to it) and the
fixed letters `a`, `b`, `c`, `d`, `S`, `D`; immediates `i`/`n` (the input
must be a compile-time constant; `%N` renders as `$N` in AT&T / `N` in
Intel, `%cN` as the bare number); and memory operands `m`/`o`/`V` — the
input value is an address, and the template sees `disp(%reg)` (AT&T) or
`[reg]` (Intel). Register size modifiers `%b`/`%w`/`%k`/`%q` select
byte/word/double/quad register names.

Unsupported (rejected with a clear panic): matching constraints
(`"0"`-style ties), SIMD operands (`x`/`y`/`k`/...), and non-fixed inouts
(`+r` — see below).

Labels defined inside a template may be referenced from it (local loop
targets, `1:`/`1f`-style numeric labels included), in either direction;
the vendored encoder resolves those fixups itself. References to symbols
outside the template are rejected.

The CLIF reader does not process escapes inside string literals; templates
written in CLIF use `\n`/`\t` for statement separation, which the
expander converts to real newline/tab.

Operand values are passed in full 64-bit registers without implicit
extension, like GCC; the template must use mnemonics of the appropriate
width. Results are `i64`; narrow the value with `ireduce` if needed.

## How the pieces fit

| Layer | File(s) |
| --- | --- |
| `AsmRef` entity, `InlineAsmContents` | `codegen/src/ir/entities.rs`, `codegen/src/ir/inline_asm.rs` |
| Entity storage | `codegen/src/ir/dfg.rs` (`inline_asms`), result typing in `inst_result_types` |
| Meta DSL (format `InlineAsm`: entity + varargs) | `codegen/meta/src/shared/{entities,formats,instructions}.rs` |
| Constraint parsing + resolved template | `codegen/src/isa/x64/inline_asm.rs` |
| `MInst::InlineAsm` variant + lowering rule | `codegen/src/isa/x64/inst.isle`, `lower.isle`, `lower/isle.rs` (`inline_asm_lower`) |
| Regalloc operands | `codegen/src/isa/x64/inst/mod.rs` (`x64_get_operands`: uses, defs, clobbers) |
| Emit-time expansion + assembly | `codegen/src/isa/x64/inline_asm.rs::emit` |
| Assembler core (nasm-derived, table-driven) | `cranelift/assembler-glasm` (vendored from the glcc project's `glasm`) |
| Reader / printer / verifier | `reader/src/parser.rs`, `codegen/src/write.rs`, `codegen/src/verifier/mod.rs` |
| Frontend API | `frontend/src/frontend.rs::create_inline_asm` |

### Inout operands and regalloc2

regalloc2 (0.15) has no read-modify-write (`Mod`) operand kind and rejects
vregs with multiple defs. A `+` inout is therefore expressed as **two
vregs and one physical register**: the input value is copied into a fresh
vreg, which the asm statement uses with a *fixed* use on the constraint's
register while defining a second, *fixed* def on the same register. The
register allocator inserts the move into place; the asm reads and writes
one register. This is the same pattern `CheckedSRemSeq` uses for its
rax/rdx-tied operands.

Non-fixed inouts (`"+r"`) need the same register for read and write with a
register chosen by the allocator, which this operand model cannot express;
they are rejected. (A regalloc2 with `Mod` operands, or tied-operand
support, would lift this.)

### Emission-time assembly

`MInst::InlineAsm`'s fields hold the operand registers directly; after
regalloc, the standard VCode operand-patching pass rewrites them in place
to physical registers (via `get_operands`, which also declares the uses,
defs, and clobbers to the allocator). At emit time the template is expanded
(placeholders resolved to register names) and assembled instruction by
instruction with the vendored glasm encoder (`table::encode`), which
matches against nasm's instruction tables — full x86-64 coverage including
every addressing mode and encoding form nasm supports. Fixups (labels,
symbols) are rejected for now.

Emission-time assembly is repeated per function compile; there is no cache
keyed on (template, resolved registers) yet.

## Tests

- `filetests/filetests/isa/x64/inline-asm.clif` — precise-output emission
  for an output+input statement and a clobber statement.
- `filetests/filetests/isa/x64/inline-asm-inout.clif` — `"+a"` inout.
- `cranelift/assembler-glasm/tests/encode.rs` — encoder unit tests.
- `cranelift/jit/tests/inline_asm.rs` — end-to-end JIT execution:
  `"=r"`/`"r"` operands, `"+a"` fixed inout, and clobber preservation of
  live values.

## Remaining gaps to GCC/Clang parity

1. **Memory operands (`"m"`)**: pass an address/machine operand into the
   template instead of a register. Needs an operand kind carrying
   `Amode` (vreg + offset) through to emit-time text.
2. **Immediate operands (`"i"`/`"n"`, letter sets)**: substitute constant
   values into the template; needs constant extraction at lowering.
3. **Matching constraints (`"0"` ties)** and `&` early-clobber semantics
   that depend on them.
4. **Non-fixed inouts (`"+r"`)**: needs `Mod` operands or ties in
   regalloc2.
5. **Labels and symbols in templates**: jump targets and external symbol
   references need glasm fixups mapped onto `MachBuffer` label/reloc
   plumbing.
6. **Flags modeling (`"cc"`)**: advisory only; cranelift does not allocate
   flags. Soundness relies on side-effecting instructions not being
   reordered.
7. **Other ISAs**: aarch64/riscv64/s390x lowering rules (the IR design is
   backend-agnostic; each backend needs its own constraint letters and
   emitter).
8. **Template caching** at emission, plus `asm!`-style frontend conveniences
   (named operands, clobber_abi, options) — those live above cranelift.
9. **Interpreter support**: currently `unimplemented!`.
