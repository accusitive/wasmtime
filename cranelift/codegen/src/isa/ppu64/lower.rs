//! Lowering rules for ppu64.
use crate::ir::Inst as IRInst;
use crate::isa::ppu64::Ppu64Backend;
use crate::isa::ppu64::inst::*;
use crate::machinst::*;

pub mod isle;

//=============================================================================
// Lowering-backend trait implementation.

impl LowerBackend for Ppu64Backend {
    type MInst = Inst;

    fn lower(&self, ctx: &mut Lower<Inst>, ir_inst: IRInst) -> Option<InstOutput> {
        isle::lower(ctx, self, ir_inst)
    }

    fn lower_branch(
        &self,
        ctx: &mut Lower<Inst>,
        ir_inst: IRInst,
        targets: &[MachLabel],
    ) -> Option<()> {
        isle::lower_branch(ctx, self, ir_inst, targets)
    }

    fn maybe_pinned_reg(&self) -> Option<Reg> {
        None
    }
}
