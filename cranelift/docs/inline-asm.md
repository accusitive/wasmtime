# Inline assembly in Cranelift — prototype and roadmap

Status: prototype implemented and tested for x86-64 (this file documents both
what exists and what a complete implementation needs).

## What the prototype does

A new CLIF instruction:

```
inline_asm bytes, clobbers
```

- `bytes`: `Imm64` holding up to 8 machine-code bytes, little-endian. All 8
  bytes are emitted verbatim into the instruction stream.
- `clobbers`: `Imm64` bitmask of GPRs destroyed by the assembly; bit N is the
  GPR with x86 register encoding index N (0 = RAX, 1 = RCX, 2 = RDX, 3 = RBX,
  4 = RSP, 5 = RBP, 6 = RSI, 7 = RDI, 8–15 = R8–R15).

The instruction is `other_side_effects + can_load + can_store`, so the
mid-end treats it as a side-effecting skeleton node: it is never GVN'd, never
DCE'd, and is ordered correctly against loads/stores. The register allocator
sees the declared clobbers via `reg_clobbers`, the same mechanism used by
calls and `ElfTlsGetAddr`.

Touch points (all under `cranelift/`):

| Site | Change |
| --- | --- |
| `codegen/meta/src/shared/immediates.rs` | new `bytes` / `clobbers` immediate kinds |
| `codegen/meta/src/shared/formats.rs` | new `InlineAsm` format (two `Imm64` fields) |
| `codegen/meta/src/shared/instructions.rs` | the `inline_asm` instruction |
| `codegen/src/isa/x64/inst.isle` | `MInst.InlineAsm { bytes, clobbers }` variant |
| `codegen/src/isa/x64/lower.isle` | lowering rule via `u64_from_imm64` extractors |
| `codegen/src/isa/x64/inst/emit.rs` | emit arm: `sink.put1` per byte |
| `codegen/src/isa/x64/inst/mod.rs` | `x64_get_operands` (clobbers), `is_available`, `pretty_print` |
| `codegen/src/verifier/mod.rs`, `write.rs` | no-op arms |
| `codegen/src/ir/instructions.rs` | size tests bumped 16 → 24 bytes |
| `reader/src/parser.rs` | parse arm for the format |
| `interpreter/src/step.rs` | `unimplemented!` arm |

Tests:

- `filetests/filetests/isa/x64/inline-asm.clif` — precise-output emission +
  disassembly check.
- `jit/tests/inline_asm.rs` — JIT-compiles a function whose body is inline
  machine code (`mov dword [rdi], 42`), runs it, and asserts the store
  landed. Proves frontend API, regalloc, emission, and execution end to end.

## Known prototype limitations

1. **8-byte payload in an immediate.** `Imm64` can't hold longer sequences,
   and it makes `InstructionData` 24 bytes (up from 16). A production design
   would intern templates in a `Function` side table (new entity kind, like
   `JumpTable`) with the instruction holding a table index, restoring the
   16-byte enum and allowing arbitrary-length asm.
2. **No value operands.** Inputs must be set up in registers by surrounding
   CLIF, and outputs must be read back out of clobbered registers by hand (or
   via ABI assumptions). This is the single biggest gap versus GCC/Clang
   inline asm.
3. **No constraint strings.** Nothing like `"r"`, `"=a"`, `"m"`, `"N"`.
4. **Flags are not clobber-modelled.** x86 flags are not regalloc'd entities
   in this backend, so an asm statement that writes flags is invisible to the
   scheduler unless memory effects happen to order it.
5. **Interpreter, serde, disassembler** treat the instruction as opaque or
   unimplemented.

## Roadmap to a complete implementation

1. **Template side table** (`ir::InlineAsm` entity on `Function`): template
   text, per-operand constraint strings, clobber list, options (volatile,
   alignstack, intel dialect). Reader/printer support for the new entity.
   Keeps `InstructionData` at 16 bytes.
2. **Constraint model.** Parse GCC-style constraint strings into an enum:
   register classes (general/XMM), fixed registers (`a`, `b`, `c`, `d`, `S`,
   `D`), memory operands, immediates, in/out/inout directions, explicit
   clobbers. This is the most design-sensitive part; it needs to map onto
   regalloc2 operands: `reg_use`, `reg_fixed_use`, `reg_def`,
   `reg_fixed_def`, `reg_fixed_late_use` and `reg_clobbers` — all of which
   the x64 backend already uses for its own pseudo-instructions.
3. **Value operands.** Extend the IR instruction to take `varargs` inputs and
   produce outputs (likely mirroring `call`'s shape). Lowering: emit each
   input per its constraint, then the template, then read outputs. Fixed
   constraints (e.g. `=a`) become `reg_fixed_def` on the machine operand;
   `+r` (inout) becomes a def that must reuse the input's register
   (`reg_fixed_late_use` + `reg_fixed_def` on the same PReg).
4. **Template expansion.** For the real feature the backend needs an
   assembler, not raw bytes: either the `cranelift-assembler-x64` DSL (with
   `.custom(Encode)` for gaps) or a port of a real assembler (nasm/GAS
   frontends — see the glcc project's `glasm` for a working table-driven
   x86-64 assembler in Rust that could be reused here).
5. **Clobbers of flags and XMM**, `memory`/`cc` clobber keywords, mapping
   onto `PRegSet` (flags need a virtual clobber or scheduler ordering
   barrier).
6. **Other ISAs**: aarch64 (fixed regs x0–x17, no flags), riscv64, s390x
   backends each need a lowering rule; the IR-level design above is
   backend-agnostic.
7. **Interpreter support**: execute templates by invoking them as generated
   code is not possible in the safe interpreter; either document as
   unimplemented (current) or evaluate templates with an emulator.

## Why raw bytes for the prototype

Everything above the machine-code layer is the hard part: the meta DSL
plumbing, ISLE lowering, regalloc integration, the reader, and the frontend
builder API all had to exist before any of the interesting constraint work
could land. The 8-byte-payload form exercises that entire pipeline with zero
design risk, and the shape of the eventual change (side table + operands +
constraint parser) is strictly additive to it.
