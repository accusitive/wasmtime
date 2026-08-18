//! Branch folding: elimination of trivial empty blocks.
//!
//! This pass fuses blocks that contain nothing but an unconditional `jump`
//! into their unique predecessor, threading block arguments as necessary.
//!
//! ```clif
//! block0(v0: i64):
//!     jump block1(v0)
//!
//! block1(v1: i64):
//!     jump block2(v1)
//!
//! block2(v2: i64):
//!     return v2
//! ```
//!
//! becomes
//!
//! ```clif
//! block0(v0: i64):
//!     jump block2(v0)
//!
//! block2(v2: i64):
//!     return v2
//! ```
//!
//! Running before the egraph mid-end, this simplifies the CFG that the
//! egraph and the backends' instruction selectors see. Frontends frequently
//! generate such empty blocks, and leaving them in place can both inhibit
//! pattern-matching during lowering (since the egraph moves pure
//! instructions across artificial block boundaries, see bytecodealliance/
//! wasmtime#6154) and impede other mid-end optimizations.

use crate::flowgraph::ControlFlowGraph;
use crate::ir::{self, Block, BlockArg, Function, Value};
use crate::timing;
use crate::trace;
use smallvec::SmallVec;

/// Determine whether `block` can be folded into its single predecessor.
///
/// A block can be folded when it is not the entry block, it has exactly one
/// predecessor, it contains nothing but an unconditional `jump`, and folding
/// it would not turn the predecessor's jump into a self-loop.
fn is_foldable(func: &Function, cfg: &ControlFlowGraph, block: Block) -> bool {
    // The block must contain exactly one instruction: an unconditional jump.
    if !func.layout.block_contains_exactly_one_inst(block) {
        return false;
    }
    let inst = func.layout.last_inst(block).unwrap();
    let target = match func.dfg.insts[inst] {
        ir::InstructionData::Jump { destination, .. } => {
            destination.block(&func.dfg.value_lists)
        }
        _ => return false,
    };

    // The block must have exactly one predecessor.
    let mut preds = cfg.pred_iter(block);
    let pred = match (preds.next(), preds.next()) {
        (Some(pred), None) => pred,
        _ => return false,
    };

    // Avoid folding a block that jumps to itself, and avoid turning the
    // predecessor's branch into a self-loop.
    if target == block || target == pred.block {
        return false;
    }

    // The target must still be in the layout: a previous fold in the same
    // sweep may have removed it.
    if !func.layout.is_block_inserted(target) {
        return false;
    }

    true
}

/// Fold `block` into its single predecessor.
///
/// `block` must satisfy `is_foldable`; this function does not re-check. The
/// predecessor's terminator is rewritten to jump directly to the destination
/// of `block`'s `jump`, and `block` is removed from the layout. The CFG is
/// updated incrementally.
fn fold_block(func: &mut Function, cfg: &mut ControlFlowGraph, block: Block) {
    let pred = cfg.pred_iter(block).next().unwrap();

    // Gather the destination and arguments of `block`'s terminating jump.
    let jump_inst = func.layout.last_inst(block).unwrap();
    let (target, target_args) = {
        let dfg = &func.dfg;
        let destination = match dfg.insts[jump_inst] {
            ir::InstructionData::Jump { destination, .. } => destination,
            _ => unreachable!(),
        };
        let target = destination.block(&dfg.value_lists);
        let args: SmallVec<[BlockArg; 8]> = destination.args(&dfg.value_lists).collect();
        (target, args)
    };

    // Gather the arguments that the predecessor currently passes to `block`.
    // These are the values that stand in for `block`'s parameters wherever
    // those parameters are used in `target`'s arguments.
    let mut pred_args: SmallVec<[BlockArg; 8]> = SmallVec::new();
    {
        let dfg = &func.dfg;
        let dests = dfg.insts[pred.inst]
            .branch_destination(&dfg.jump_tables, &dfg.exception_tables);
        for d in dests {
            if d.block(&dfg.value_lists) == block {
                pred_args.extend(d.args(&dfg.value_lists));
            }
        }
    }
    debug_assert!(!pred_args.is_empty() || func.dfg.num_block_params(block) == 0);

    // Rewrite the predecessor's branch to jump directly to `target`, with the
    // block arguments threaded through.
    {
        let dfg = &mut func.dfg;
        let block_params: SmallVec<[Value; 8]> = dfg.block_params(block).iter().copied().collect();
        let dests = dfg.insts[pred.inst]
            .branch_destination_mut(&mut dfg.jump_tables, &mut dfg.exception_tables);
        for d in dests {
            if d.block(&dfg.value_lists) != block {
                continue;
            }
            let new_args = target_args.iter().map(|arg| match *arg {
                BlockArg::Value(v) => match block_params.iter().position(|&p| p == v) {
                    Some(ix) => pred_args[ix],
                    None => *arg,
                },
                other => other,
            });
            d.set_block(target, &mut dfg.value_lists);
            d.clear(&mut dfg.value_lists);
            d.extend(new_args, &mut dfg.value_lists);
        }
    }

    // Remove `block` and its jump from the layout, and update the CFG:
    // `pred` gains `target` as a successor, and `block` is dropped entirely.
    func.layout.remove_inst(jump_inst);
    cfg.recompute_block(func, pred.block);
    cfg.recompute_block(func, block);
    func.layout.remove_block(block);
}

/// Fuse chains of empty blocks into their predecessors, threading block
/// arguments as necessary.
///
/// Returns the number of blocks folded.
pub fn do_branch_folding(func: &mut Function, cfg: &mut ControlFlowGraph) -> usize {
    let _tt = timing::branch_folding();
    let entry = match func.layout.entry_block() {
        Some(e) => e,
        None => return 0,
    };

    let mut folded = 0;

    // Sweep to a fixed point. Folding a block can make its successor
    // foldable in turn (a chain of empty blocks), so a single sweep over the
    // layout is not always sufficient. Each sweep removes at least one block,
    // so this terminates.
    loop {
        let mut folded_any = false;
        let blocks: SmallVec<[Block; 16]> = func.layout.blocks().collect();
        for &block in &blocks {
            if block == entry {
                continue;
            }
            if !is_foldable(func, cfg, block) {
                continue;
            }
            fold_block(func, cfg, block);
            folded += 1;
            folded_any = true;
        }
        if !folded_any {
            break;
        }
    }

    if folded > 0 {
        trace!("Folded {folded} empty blocks");
    }
    folded
}
