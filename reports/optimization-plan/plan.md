# Cranelift Missing Optimizations — Investigation & Implementation Plan

Date: 2026-08-17
Repo: bytecodealliance/wasmtime @ bc2f967927
Focus: Wasmtime runtime performance

## Investigation summary

Cranelift's optimizer consists of a single fused egraph mid-end pass
(`cranelift/codegen/src/egraph/mod.rs`) doing GVN, ISLE `simplify` rule
saturation (`cranelift/codegen/src/opts/*.isle`), alias-analysis-driven memory
opts (`alias_analysis.rs`), cost-based extraction/elaboration with built-in
LICM, plus pre-passes (`remove_constant_phis`, `eliminate_unreachable_code`).
Backends lower via ISLE, then regalloc + emit (only MachBuffer branch
peepholes).

### Missing (identified)

Analyses:
- No known-bits / value-range analysis (only a cheap single-mask `uextend`
  trick, `opts/extends.isle:15-25`).
- No SCCP / path-sensitive constant propagation; block params are opaque
  eclasses (`egraph/mod.rs:950-954`).
- No jump threading / empty-block fusion (`inline.rs:1127-1130`; issue #6154).

Mid-end rules/mechanisms:
- Multi-result ops excluded from the egraph (`inst_predicates.rs:62-70`);
  only `uadd_overflow_trap` has rules (`opts/skeleton.isle:12-18`).
- No `imul` strength reduction for non-pow2 constants
  (`opts/arithmetic.isle:172-182` covers only 2/pow2).
- Float const-fold missing for fcmp/fma/demote/promote/to-int
  (`opts/cprop.isle:397`).
- Extend-load reuse: RLE/store-forwarding bails on differing extension mode
  (`alias_analysis.rs:561-563`).
- Cost model documents loop-nest x1024 scaling but does not implement it
  (`egraph/cost.rs:16-19`).
- Rematerialization is single-level only (#7313, `egraph/elaborate.rs:424`).

Loop/CFG: no memory LICM, no loop unrolling, no IV strength reduction.
Backend: no post-lowering peephole/scheduling pass.

## Implementation plan

Ordered by impact on Wasmtime runtime perf, then by risk.

### Phase 1 — well-scoped

1. **1.1 Jump threading / empty-block fusion** (`src/branch_folding.rs`)
   - [x] Implemented `do_branch_folding`: fuses blocks with exactly one
     predecessor containing only a `jump` terminator, threading block args.
     Guards: not entry block, no self-loop / no back-edge-into-pred fold, jump
     target still in layout (same-sweep cascade).
   - [x] Wired into `context.rs::optimize()` before `egraph_pass()`
     (recomputes cfg+domtree after folding; runs only at opt_level != None).
   - [x] New `timing::branch_folding` entry.
   - [x] Tests: new `filetests/cfg/branch-folding.clif`; blessed 10
     `precise-output` optimize tests + fixed `egraph/load-hoist.clif`
     filecheck (all diffs are genuine empty-block folds).
   - [x] Verified: all 1283 `clif-util test` pass, `cargo test -p
     cranelift-codegen` pass, `cargo clippy -p cranelift-codegen` clean.
   - Not done: `brif (icmp ne x 0)` -> `brif x` canonicalization (already
     handled by egraph `skeleton.isle` truthy rules).

2. **1.2 Cost-model loop scaling** (`src/egraph/cost.rs`, `elaborate.rs`)
   - [x] Implemented `Cost::scale_by_loop_level`: multiplies a cost by
     1024 per loop-nest level of the block in which the value is
     computed, capped at level two (u32). Documented behavior in
     `cost.rs` is now real.
   - [x] `BestEntry` now carries `LoopLevel`; `compute_best_values`
     assigns loop levels: params/skeleton values get their block's
     level, pure ops get the max over operands (a value cannot be
     hoisted above its operands; a fully loop-invariant value carries
     level 0 and is unscaled).
   - [x] Unit test `loop_nest_scaling` for the arithmetic
     (incl. capping and saturation). All 1283 filetests pass unchanged
     (scaling only biases eclass choice toward hoistable reps; it does
     not alter existing rule-driven decisions).
   - Status: done.

3. **1.3 Extend-load store-to-load forwarding** (`src/alias_analysis.rs`)
   - [x] When a known memory value's recorded `extending_opcode` differs from
     the load's (e.g. `istore8` then `uload8`, or a wider store/load then a
     narrower extending load), synthesize the missing truncate+extend instead
     of bailing, and alias the load result to it.
   - [x] New `MemExtend` (None/Zero/Sign) + `mem_extend`/`mem_access_width`
     helpers; `OptResult::AliasedLoadExtend` variant; `try_forward_extending_load`
     scan (extending loads only, matching same memory version/address/offset,
     width >= load width, domination check).
   - [x] Soundness: same-width forwards are endianness-independent (store and
     load access the same bytes in the same order); truncating a wider value
     requires both accesses explicitly little-endian (`is_explicit_little_endian`).
   - [x] Synthesis: standalone path inserts `ireduce`+`uextend`/`sextend` via
     `FuncCursor`; egraph path builds the chain through `insert_pure_enode` so
     it is GVN'd (the residual `uextend(ireduce(...))` is not further folded
     by ISLE, matching what the plain egraph does for that pattern).
   - [x] Tests: new `filetests/egraph/extends-forwarding.clif` (8 optimize
     cases incl. negative native-endianness) + new
     `filetests/runtests/extends-forwarding.clif` (runtime correctness across
     aarch64/x86_64/riscv64 for all forwarding modes); updated
     `filetests/alias/extends.clif` (same-width cross-mode forward is now
     sound). All 1285 filetests pass, `cargo test -p cranelift-codegen` pass,
     clippy clean.
   - Status: done.

### Phase 2 — analysis-driven

4. **2.1 Known-bits analysis** (`src/known_bits.rs`, `prelude_opt.isle`,
   `opts/bitops.isle`, `opts/icmp.isle`)
   - Per-value known-zero/known-one masks, computed in a forward pass over the
     egraph traversal (`insert_pure_enode`), merged (intersection) into egraph
     eclasses, and exposed to ISLE `simplify` rules.
   - Analysis covers scalar ints: constants (`iconst`), `uextend`/`sextend`/
     `ireduce`/`bnot`, `band`/`bor`/`bxor`, `ishl`/`ushr`/`sshr` (const shift
     amounts, taken modulo bitwidth), and `icmp` (result is 0-or-1).
   - ISLE externs: `known_bits_zero_mask`, `known_bits_one_mask`,
     `known_bits_sign_zero`, `known_bits_sign_one`; rules added:
     `band x, mask -> const` when the known bits cover `mask`,
     `slt`/`sge x, 0` when the sign bit is known, `eq`/`ne x, 0` when a
     one-bit is known or `x` is fully known zero.
   - Tests: `filetests/egraph/known-bits.clif` (incl. negative case: `band` of
     an `icmp` result must not fold to 1). All 1286 filetests pass, `cargo
     test -p cranelift-codegen` (194 unit tests) pass, clippy clean.
   - Status: done.
   - Known-bits-driven *narrowing* (`opts/extends.isle`, `opts/bitops.isle`,
     `filetests/egraph/narrowing.clif`): added
     `uextend(ireduce x) -> x` when the reduced value's high bits are known
     zero (extend elimination), and `band.i64 (band|bor|bxor.i64 x y)
     0xffff_ffff -> uextend.i64 (op.i32 (ireduce x) (ireduce y))` (carry-free
     ops are done at 32 bits when re-masked; `iadd`/`isub`/`imul` are
     deliberately NOT narrowed without a wrap since the carry can propagate).
     All 1287 filetests pass, unit tests pass, clippy clean.

5. **2.2 Extend egraph to multi-result ops** (`inst_predicates.rs`,
    `egraph/mod.rs`, new `opts/overflow.isle`)
    - [x] Multi-result pure ops (e.g. `uadd_overflow`, `sadd_overflow`,
      `iadd_carry`) are now mergeable by the egraph: `is_mergeable_for_egraph`
      no longer excludes them (multi-result ops with side-effects, even
      idempotent ones, are still excluded), and the GVN merge path maps *all*
      results of the duplicate instruction onto the original's results rather
      than just the first.
    - [x] Tests: `filetests/egraph/multi-result-merge.clif` (merge with both
      results, merge with one result used, negative case with different
      operands) and `filetests/runtests/multi-result-merge.clif` (runtime +
      interpreter correctness of a merged `uadd_overflow`). All 1293 filetests
      pass, 195 unit tests pass, clippy clean.
    - Partial: making multi-result ops full "pure" egraph enodes (so ISLE
      `simplify` rules can constant-fold/canonicalize carry/overflow ops) is a
      larger change: `insert_pure_enode` and the mid-end ISLE `simplify`
      interface return a single `Value` today, so producing/consuming
      (result, overflow) tuples needs new infrastructure. Deferred; the
      mergeability half is the low-risk win.

6. **2.3 ISLE rule expansion**
   - `imul` non-pow2 strength reduction (aarch64/riscv64).
     - Deferred / low priority per project direction: backend-specific; the
       target-independent egraph/mid-end work is the focus (x86 primary).
   - Float const-fold for fcmp/fma/to-int (`opts/cprop.isle`,
     `opts.rs`).
     - [x] `fcmp` of constant floats folds to `iconst.i8 0/1` (F16/F32/F64).
       The result is an integer so there is no NaN-bit-pattern concern; all
       14 FloatCC conditions are covered and NaN inputs evaluate per IEEE
       (ordered comparisons false, `ne` true). Scalar `fcmp` results are
       always `i8`; the wasm `uextend.i32` of it folds away too.
     - [x] `fma` of constant floats folds using a correctly-rounded fused
       multiply-add (`libm::fmaf`/`libm::fma`), guarded against NaN results
       (preserves NaN bit patterns). Test pins a fused-vs-double-rounded
       difference (`fma(0x1.000002p0, 0x1.000002p0, 0x1.000000p-24)` ->
       0x1.000006p0, not 0x1.000004p0).
     - [x] `fdemote`/`fpromote` of non-NaN constants fold (F64->F32,
       F32->F64); NaN inputs are not folded since demoting can change the
       NaN bit pattern.
     - [x] `fcvt_to_{s,u}int_sat` of constants always fold (never trap,
       NaN -> 0, out-of-range saturates), for F32/F64 to I32/I64. The
       trapping `fcvt_to_{s,u}int` variants are intentionally not folded
       (they are `Inst`-typed and may trap).
     - Tests: `filetests/egraph/float-const-fold.clif` (incl. negative
       NaN cases). All 1288 filetests pass, 194 unit tests pass, clippy
       clean.
     - Status: done.
   - `fma` fusion behind a flag (wasm NaN semantics).
     - [x] New shared setting `enable_fma` (default `false`); the egraph
       `simplify` rules consult it via an ISLE extern (`fma_allowed`) plumbed
       through `EgraphPass`/`OptimizeCtx`.
     - [x] When enabled, `fadd(fmul(x,y), z)`, `fadd(z, fmul(x,y))` and
       `fsub(fmul(x,y), z)` fuse into a single `fma`. The fused result is
       rounded once instead of twice, so it is more precise but may differ
       from the non-fused sequence; NaN propagation differs too, hence the
       opt-in flag. `x*y - z` emits `fma(x, y, fneg(z))` and backends fold the
       negation into the fused instruction (`vfnmadd`/`fnmsub`).
     - [x] Flag plumbing: `cranelift-codegen-meta` shared setting,
       `settings.rs` default-settings test, `context.rs::egraph_pass`,
       `EgraphPass`, `OptimizeCtx`, `opts.rs` extern. Guard is written
       `(if-let true (fma_allowed))` since ISLE's bare `(if e)` form desugars
       to a wildcard if-let and DCEs the pure call.
     - [x] Tests: `filetests/egraph/fma-fusion.clif` (fusion + constant
       operands still fold to a constant), `filetests/egraph/fma-no-fusion.clif`
       (flag off), `filetests/runtests/fma-fusion.clif` (runtime: fused result
       equals the correctly-rounded value, differing from the double-rounded
       non-fused result for chosen inputs). All 1291 filetests pass, 195 unit
       tests pass, clippy clean.
     - Status: done.

### Testing

- `filetests/filetests/egraph/*.clif` + runtests, differential via
  `clif-util`.
- `cargo test -p cranelift-codegen`; `cargo run -p cranelift-tools test`.
- fuzzgen for rule interplay.