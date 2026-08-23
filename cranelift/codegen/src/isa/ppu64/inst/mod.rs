//! This module defines ppu64-specific machine instruction types.

use crate::CodegenResult;
use crate::binemit::{Addend, CodeOffset, Reloc};
use crate::ir::condcodes::IntCC;
use crate::ir::types::{F32, F64, I8, I16, I32, I64, I8X16};
pub use crate::ir::{ExternalName, MemFlagsData, Type};
use crate::isa::FunctionAlignment;
use crate::machinst::*;

pub mod args;
pub use self::args::*;
pub mod emit;
pub use self::emit::*;
pub mod imms;
pub use self::imms::*;
pub mod regs;
pub use self::regs::*;

use crate::isa::ppu64::abi::Ppu64MachineDeps;

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::{Display, Formatter};
use regalloc2::RegClass;
use smallvec::smallvec;

pub use crate::isa::ppu64::lower::isle::generated_code::{
    AluOPRR, AluOPRRI, AluOPRRR, FpuOPRR, FpuOPRRR, FpuOPRRRR, LoadOP, MInst as Inst, StoreOP,
    VecALUOp,
};

/// A conditional branch target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CondBrTarget {
    /// An unresolved reference to a Label, as passed into
    /// `lower_branch_group()`.
    Label(MachLabel),
    /// No jump; fall through to the next instruction.
    Fallthrough,
}

impl CondBrTarget {
    /// Return the target's label, if it is a label-based target.
    pub(crate) fn as_label(self) -> Option<MachLabel> {
        match self {
            CondBrTarget::Label(l) => Some(l),
            _ => None,
        }
    }
}

impl Display for CondBrTarget {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            CondBrTarget::Label(l) => write!(f, "{}", l.to_string()),
            CondBrTarget::Fallthrough => write!(f, "0"),
        }
    }
}

/// A register-register comparison together with the comparison kind and
/// operand width, used by conditional branches and traps. The compare is
/// emitted against CR field 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntegerCompare {
    pub kind: IntCC,
    pub rs1: Reg,
    pub rs2: Reg,
    /// True when operands are 64-bit (`cmpd`/`cmpld`); false for 32-bit
    /// (`cmpw`/`cmplw`).
    pub is64: bool,
}

/// Invert an integer comparison condition.
pub(crate) fn invert_intcc(cc: IntCC) -> IntCC {
    match cc {
        IntCC::Equal => IntCC::NotEqual,
        IntCC::NotEqual => IntCC::Equal,
        IntCC::SignedLessThan => IntCC::SignedGreaterThanOrEqual,
        IntCC::SignedGreaterThanOrEqual => IntCC::SignedLessThan,
        IntCC::SignedGreaterThan => IntCC::SignedLessThanOrEqual,
        IntCC::SignedLessThanOrEqual => IntCC::SignedGreaterThan,
        IntCC::UnsignedLessThan => IntCC::UnsignedGreaterThanOrEqual,
        IntCC::UnsignedGreaterThanOrEqual => IntCC::UnsignedLessThan,
        IntCC::UnsignedGreaterThan => IntCC::UnsignedLessThanOrEqual,
        IntCC::UnsignedLessThanOrEqual => IntCC::UnsignedGreaterThan,
    }
}

impl IntegerCompare {
    /// Return the same comparison with the condition inverted.
    pub fn inverse(self) -> IntegerCompare {
        IntegerCompare {
            kind: invert_intcc(self.kind),
            rs1: self.rs1,
            rs2: self.rs2,
            is64: self.is64,
        }
    }
}

fn riscv_get_operands(inst: &mut Inst, collector: &mut impl OperandVisitor) {
    match inst {
        Inst::Nop0 | Inst::Nop4 => {}
        Inst::Mov { rd, rm, .. } => {
            collector.reg_use(rm);
            collector.reg_def(rd);
        }
        Inst::AluRRR { rd, rs1, rs2, .. } => {
            collector.reg_use(rs1);
            collector.reg_use(rs2);
            collector.reg_def(rd);
        }
        Inst::AluRR { rd, rs, .. } => {
            collector.reg_use(rs);
            collector.reg_def(rd);
        }
        Inst::AluRRSimm16 { rd, ra, .. } => {
            collector.reg_use(ra);
            collector.reg_def(rd);
        }
        Inst::LoadConst64 { rd, .. } => {
            collector.reg_def(rd);
        }
        Inst::Li32 { rd, .. } => {
            collector.reg_def(rd);
        }
        Inst::Load { rd, from, .. } => {
            from.get_operands(collector);
            collector.reg_def(rd);
        }
        Inst::Store { to, src, .. } => {
            to.get_operands(collector);
            collector.reg_use(src);
        }
        Inst::LoadAddr { rd, mem } => {
            mem.get_operands(collector);
            collector.reg_early_def(rd);
        }
        Inst::Args { args } => {
            for ArgPair { vreg, preg } in args {
                collector.reg_fixed_def(vreg, *preg);
            }
        }
        Inst::Rets { rets } => {
            for RetPair { vreg, preg } in rets {
                collector.reg_fixed_use(vreg, *preg);
            }
        }
        Inst::Ret {} => {}
        Inst::Extend { rd, rn, .. } => {
            collector.reg_use(rn);
            collector.reg_def(rd);
        }
        Inst::Call { info, .. } => {
            let CallInfo { uses, defs, .. } = &mut **info;
            for CallArgPair { vreg, preg } in uses {
                collector.reg_fixed_use(vreg, *preg);
            }
            for CallRetPair { vreg, location } in defs {
                match location {
                    RetLocation::Reg(preg, ..) => collector.reg_fixed_def(vreg, *preg),
                    RetLocation::Stack(..) => collector.any_def(vreg),
                }
            }
            collector.reg_clobbers(info.clobbers);
        }
        Inst::CallInd { info } => {
            let CallInfo {
                dest, uses, defs, ..
            } = &mut **info;
            collector.reg_use(dest);
            for CallArgPair { vreg, preg } in uses {
                collector.reg_fixed_use(vreg, *preg);
            }
            for CallRetPair { vreg, location } in defs {
                match location {
                    RetLocation::Reg(preg, ..) => collector.reg_fixed_def(vreg, *preg),
                    RetLocation::Stack(..) => collector.any_def(vreg),
                }
            }
            collector.reg_clobbers(info.clobbers);
        }
        Inst::Jmp { .. } => {}
        Inst::CondBr {
            kind: IntegerCompare { rs1, rs2, .. },
            ..
        } => {
            collector.reg_use(rs1);
            collector.reg_use(rs2);
        }
        Inst::BrTable {
            index, tmp1, tmp2, ..
        } => {
            collector.reg_use(index);
            collector.reg_early_def(tmp1);
            collector.reg_early_def(tmp2);
        }
        Inst::TrapIf { cmp, .. } => {
            collector.reg_use(&mut cmp.rs1);
            collector.reg_use(&mut cmp.rs2);
        }
        Inst::SetCC { rd, cmp } => {
            collector.reg_late_use(&mut cmp.rs1);
            collector.reg_late_use(&mut cmp.rs2);
            collector.reg_def(rd);
        }
        Inst::Udf { .. } => {}
        Inst::MovFromPReg { rd, rm: _ } => {
            collector.reg_def(rd);
        }
        Inst::Mfspr { rd, .. } => {
            collector.reg_def(rd);
        }
        Inst::Mtspr { rs, .. } => {
            // The SPR (LR/CTR) is not modelled as an allocatable register.
            let _ = rs;
        }
        Inst::FpuRRR { rd, rs1, rs2, .. } => {
            collector.reg_use(rs1);
            collector.reg_use(rs2);
            collector.reg_def(rd);
        }
        Inst::FpuRRRR { rd, ra, rb, rc, .. } => {
            collector.reg_use(ra);
            collector.reg_use(rb);
            collector.reg_use(rc);
            collector.reg_def(rd);
        }
        Inst::FpuRR { rd, rs, .. } => {
            collector.reg_use(rs);
            collector.reg_def(rd);
        }
        Inst::FCmpSetCC { rd, a, b, .. } => {
            collector.reg_late_use(a);
            collector.reg_late_use(b);
            collector.reg_def(rd);
        }
        Inst::MoveGprToFpr { rd, rs } => {
            collector.reg_use(rs);
            collector.reg_def(rd);
        }
        Inst::MoveGprToFpr32 { rd, rs } => {
            collector.reg_use(rs);
            collector.reg_def(rd);
        }
        Inst::MoveFprBitsToGpr32 { rd, rs } => {
            collector.reg_use(rs);
            collector.reg_def(rd);
        }
        Inst::MoveFprToGprI32 { rd, rs } => {
            collector.reg_use(rs);
            collector.reg_def(rd);
        }
        Inst::MoveFprToGpr { rd, rs } => {
            collector.reg_use(rs);
            collector.reg_def(rd);
        }
        Inst::PromoteF32ToF64 { rd, rs } => {
            collector.reg_use(rs);
            collector.reg_def(rd);
        }
        Inst::Clz { rd, rn, .. } => {
            collector.reg_use(rn);
            collector.reg_def(rd);
        }
        Inst::Rlwinm { rd, rn, .. } => {
            collector.reg_use(rn);
            collector.reg_def(rd);
        }
        Inst::VecALU { rd, ra, rb, .. } => {
            collector.reg_use(ra);
            collector.reg_use(rb);
            collector.reg_def(rd);
        }
        Inst::VecLoad { rd, mem, .. } => {
            mem.get_operands(collector);
            collector.reg_def(rd);
        }
        Inst::VecStore { rs, mem, .. } => {
            mem.get_operands(collector);
            collector.reg_use(rs);
        }
        Inst::Unwind { .. } => {}
        Inst::DummyUse { reg } => {
            collector.reg_use(reg);
        }
    }
}

impl MachInst for Inst {
    type LabelUse = LabelUse;
    type ABIMachineSpec = Ppu64MachineDeps;

    // All zeroes is an illegal instruction on PowerPC.
    const TRAP_OPCODE: &'static [u8] = &[0; 4];

    fn gen_dummy_use(reg: Reg) -> Self {
        Inst::DummyUse { reg }
    }

    fn canonical_type_for_rc(rc: RegClass) -> Type {
        match rc {
            regalloc2::RegClass::Int => I64,
            regalloc2::RegClass::Float => F64,
            regalloc2::RegClass::Vector => I8X16,
        }
    }

    fn is_safepoint(&self) -> bool {
        match self {
            Inst::Call { .. } | Inst::CallInd { .. } => true,
            _ => false,
        }
    }

    fn get_operands(&mut self, collector: &mut impl OperandVisitor) {
        riscv_get_operands(self, collector);
    }

    fn is_move(&self) -> Option<(Writable<Reg>, Reg)> {
        match self {
            Inst::Mov { rd, rm, .. } => Some((*rd, *rm)),
            _ => None,
        }
    }

    fn is_included_in_clobbers(&self) -> bool {
        match self {
            &Inst::Args { .. } => false,
            _ => true,
        }
    }

    fn is_trap(&self) -> bool {
        match self {
            Self::Udf { .. } => true,
            _ => false,
        }
    }

    fn is_args(&self) -> bool {
        match self {
            Self::Args { .. } => true,
            _ => false,
        }
    }

    fn call_type(&self) -> CallType {
        match self {
            Inst::Call { .. } | Inst::CallInd { .. } => CallType::Regular,
            _ => CallType::None,
        }
    }

    fn is_term(&self) -> MachTerminator {
        match self {
            &Inst::Jmp { .. } => MachTerminator::Branch,
            &Inst::CondBr { .. } => MachTerminator::Branch,
            &Inst::BrTable { .. } => MachTerminator::Branch,
            &Inst::Rets { .. } => MachTerminator::Ret,
            &Inst::Ret {} => MachTerminator::Ret,
            _ => MachTerminator::None,
        }
    }

    fn is_mem_access(&self) -> bool {
        matches!(
            self,
            Inst::Load { .. } | Inst::Store { .. } | Inst::VecLoad { .. } | Inst::VecStore { .. }
        )
    }

    fn gen_move(to_reg: Writable<Reg>, from_reg: Reg, ty: Type) -> Inst {
        Inst::Mov {
            rd: to_reg,
            rm: from_reg,
            ty,
        }
    }

    fn gen_nop(preferred_size: usize) -> Inst {
        if preferred_size == 0 {
            return Inst::Nop0;
        }
        assert!(preferred_size >= 4);
        Inst::Nop4
    }

    fn gen_nop_units() -> Vec<Vec<u8>> {
        // ori r0,r0,0
        vec![vec![0x60, 0x00, 0x00, 0x00]]
    }

    fn rc_for_type(ty: &Type) -> CodegenResult<(&[RegClass], &[Type])> {
        if ty.is_vector() {
            return Ok((&[RegClass::Vector], core::slice::from_ref(ty)));
        }
        match *ty {
            I8 | I16 | I32 | I64 => Ok((&[RegClass::Int], core::slice::from_ref(ty))),
            F32 | F64 => Ok((&[RegClass::Float], core::slice::from_ref(ty))),
            _ => Err(crate::CodegenError::Unsupported(format!(
                "Unexpected SSA-value type: {ty}"
            ))),
        }
    }

    fn gen_jump(target: MachLabel) -> Inst {
        Inst::Jmp { label: target }
    }

    fn worst_case_size() -> CodeOffset {
        // BrTable can emit up to 8 + 4*targets instructions.
        64
    }

    fn worst_case_island_growth() -> CodeOffset {
        512
    }

    fn function_alignment() -> FunctionAlignment {
        FunctionAlignment {
            minimum: 4,
            preferred: 16,
        }
    }
}

//=============================================================================
// Label-use kind: the I-form `b`/`bl` with its signed 24-bit word offset.

/// A label-use for direct branches (`b`/`bl`): bits 6..29 hold a signed
/// word offset relative to the branch instruction itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LabelUse;

impl MachInstLabelUse for LabelUse {
    const ALIGN: CodeOffset = 4;

    fn max_pos_range(self) -> CodeOffset {
        // Signed 24-bit immediate of instruction words.
        (1 << 25) - 4
    }

    fn max_neg_range(self) -> CodeOffset {
        1 << 25
    }

    fn patch_size(self) -> CodeOffset {
        4
    }

    fn patch(self, buffer: &mut [u8], use_offset: CodeOffset, label_offset: CodeOffset) {
        let delta = label_offset as i64 - use_offset as i64;
        debug_assert!(delta % 4 == 0);
        let li = ((delta >> 2) as i32 as u32) & 0x00ff_ffff;
        let mut word = u32::from_be_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]);
        word &= !0x03ff_fffc;
        word |= li << 2;
        buffer.copy_from_slice(&word.to_be_bytes());
    }

    fn supports_veneer(self) -> bool {
        true
    }

    fn veneer_size(self) -> CodeOffset {
        4
    }

    fn worst_case_veneer_size() -> CodeOffset {
        4
    }

    fn generate_veneer(self, buffer: &mut [u8], veneer_offset: CodeOffset) -> (CodeOffset, Self) {
        // The veneer is itself a plain unconditional `b`; the caller
        // registers the returned label-use so that it is patched with the
        // real target offset.
        let _ = veneer_offset;
        let word = 0x4800_0000u32.to_be_bytes();
        buffer.copy_from_slice(&word);
        (veneer_offset, LabelUse)
    }

    fn from_reloc(reloc: Reloc, _addend: Addend) -> Option<Self> {
        match reloc {
            Reloc::Ppc64Call => Some(LabelUse),
            _ => None,
        }
    }
}

//=============================================================================
// Pretty-printing of instructions.

pub fn reg_name(reg: Reg) -> String {
    match reg.to_real_reg() {
        Some(real) => match real.class() {
            RegClass::Int => format!("r{}", real.hw_enc()),
            RegClass::Float => format!("f{}", real.hw_enc()),
            RegClass::Vector => format!("v{}", real.hw_enc()),
        },
        None => {
            format!("{reg:?}")
        }
    }
}

//=============================================================================
// Helpers for common instruction sequences.

/// Move from the link register into `rd`.
pub(crate) fn mflr(rd: Writable<Reg>) -> Inst {
    Inst::Mfspr { rd, spr: 8 }
}

/// Move `rs` into the link register.
pub(crate) fn mtlr(rs: Writable<Reg>) -> Inst {
    Inst::Mtspr {
        rs: rs.to_reg(),
        spr: 8,
    }
}

impl Inst {
    /// Load a 32-bit constant into a register.
    pub(crate) fn load_constant_u32(rd: Writable<Reg>, value: u64) -> SmallInstVec<Inst> {
        smallvec![Inst::Li32 {
            rd,
            imm: value as u32
        }]
    }

    /// Load a 64-bit constant into a register.
    pub(crate) fn load_constant_u64(rd: Writable<Reg>, value: u64) -> SmallInstVec<Inst> {
        smallvec![Inst::LoadConst64 { rd, imm: value }]
    }

    /// Generic constructor for a load (zero-extending where appropriate).
    pub fn gen_load(into_reg: Writable<Reg>, mem: AMode, ty: Type, flags: MemFlagsData) -> Inst {
        Inst::Load {
            op: LoadOP::from_type(ty),
            rd: into_reg,
            from: mem,
            flags,
        }
    }

    /// Generic constructor for a store.
    pub fn gen_store(mem: AMode, from_reg: Reg, ty: Type, flags: MemFlagsData) -> Inst {
        Inst::Store {
            op: StoreOP::from_type(ty),
            src: from_reg,
            to: mem,
            flags,
        }
    }
}
