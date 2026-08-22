//! ISLE integration glue code for ppu64 lowering.

// Pull in the ISLE generated code.
pub mod generated_code;
use generated_code::MInst;

// Types that the generated ISLE code uses via `use super::*`.
use crate::ir::condcodes::{FloatCC, IntCC};
use crate::isa::ppu64::Ppu64Backend;
use crate::machinst::{isle::*, CallInfo, MachInst, Reg};
use crate::machinst::{VCodeConstant, VCodeConstantData};
use crate::{
    ir::{
        immediates::*, types::*, BlockCall, ExternalName, Inst, InstructionData, MemFlagsData,
        Opcode, TrapCode, Value, ValueList,
    },
    isa::ppu64::inst::*,
    machinst::{ArgPair, CallArgList, CallRetList, InstOutput, RetPair},
};
use alloc::boxed::Box;
use alloc::vec::Vec;
use regalloc2::PReg;

type BoxCallInfo = Box<CallInfo<ExternalName>>;
type BoxCallIndInfo = Box<CallInfo<Reg>>;
type VecArgPair = Vec<ArgPair>;
type VecRetPair = Vec<RetPair>;

pub(crate) struct Ppu64IsleContext<'a, 'b>
where
    'b: 'a,
{
    pub lower_ctx: &'a mut Lower<'b, MInst>,
    pub backend: &'a Ppu64Backend,
}

impl<'a, 'b> Ppu64IsleContext<'a, 'b> {
    fn new(lower_ctx: &'a mut Lower<'b, MInst>, backend: &'a Ppu64Backend) -> Self {
        Self { lower_ctx, backend }
    }
}

impl Ppu64IsleContext<'_, '_> {
    pub(crate) fn dfg(&self) -> &crate::ir::DataFlowGraph {
        &self.lower_ctx.f.dfg
    }
}

impl generated_code::Context for Ppu64IsleContext<'_, '_> {
    isle_lower_prelude_methods!();

    #[inline]
    fn emit(&mut self, arg0: &MInst) -> Unit {
        self.lower_ctx.emit(arg0.clone());
    }

    fn xreg_new(&mut self, r: Reg) -> XReg {
        XReg::new(r).unwrap()
    }
    fn writable_xreg_new(&mut self, r: WritableReg) -> WritableXReg {
        r.map(|wr| XReg::new(wr).unwrap())
    }
    fn writable_xreg_to_xreg(&mut self, arg0: WritableXReg) -> XReg {
        arg0.to_reg()
    }
    fn writable_xreg_to_writable_reg(&mut self, arg0: WritableXReg) -> WritableReg {
        arg0.map(|xr| xr.to_reg())
    }
    fn xreg_to_reg(&mut self, arg0: XReg) -> Reg {
        *arg0
    }
    fn freg_new(&mut self, r: Reg) -> FReg {
        FReg::new(r).unwrap()
    }
    fn writable_freg_new(&mut self, r: WritableReg) -> WritableFReg {
        r.map(|wr| FReg::new(wr).unwrap())
    }
    fn writable_freg_to_writable_reg(&mut self, arg0: WritableFReg) -> WritableReg {
        arg0.map(|fr| fr.to_reg())
    }
    fn freg_to_reg(&mut self, arg0: FReg) -> Reg {
        *arg0
    }

    fn gen_call_info(
        &mut self,
        sig: Sig,
        dest: ExternalName,
        uses: CallArgList,
        defs: CallRetList,
        try_call_info: OptionTryCallInfo,
        patchable: bool,
    ) -> BoxCallInfo {
        let stack_ret_space = self.lower_ctx.sigs()[sig].sized_stack_ret_space();
        let stack_arg_space = self.lower_ctx.sigs()[sig].sized_stack_arg_space();
        self.lower_ctx
            .abi_mut()
            .accumulate_outgoing_args_size(stack_ret_space + stack_arg_space);
        Box::new(
            self.lower_ctx
                .gen_call_info(sig, dest, uses, defs, try_call_info, patchable),
        )
    }

    fn gen_call_ind_info(
        &mut self,
        sig: Sig,
        dest: Reg,
        uses: CallArgList,
        defs: CallRetList,
        try_call_info: OptionTryCallInfo,
    ) -> BoxCallIndInfo {
        let stack_ret_space = self.lower_ctx.sigs()[sig].sized_stack_ret_space();
        let stack_arg_space = self.lower_ctx.sigs()[sig].sized_stack_arg_space();
        self.lower_ctx
            .abi_mut()
            .accumulate_outgoing_args_size(stack_ret_space + stack_arg_space);
        Box::new(
            self.lower_ctx
                .gen_call_info(sig, dest, uses, defs, try_call_info, false),
        )
    }

    fn gen_li(&mut self, ty: Type, n: u64) -> Reg {
        let rd = self.temp_writable_reg(ty);
        let insts = if ty.bits() <= 32 {
            MInst::load_constant_u32(rd, n)
        } else {
            MInst::load_constant_u64(rd, n)
        };
        for inst in insts {
            self.lower_ctx.emit(inst);
        }
        rd.to_reg()
    }

    fn memflags_trusted(&mut self) -> MemFlagsData {
        MemFlagsData::trusted()
    }

    fn amode_from_ptr(&mut self, p: Value, offset: Offset32) -> AMode {
        let reg = self.put_in_reg(p);
        AMode::RegOffset(reg, i64::from(offset))
    }

    fn gen_store_default(&mut self, val: Value, p: Value, offset: Offset32) -> InstOutput {
        let ty = self.lower_ctx.dfg().value_type(val);
        let op = StoreOP::from_type(ty);
        let src = self.put_in_reg(val);
        let addr = self.put_in_reg(p);
        let mem = AMode::RegOffset(addr, i64::from(offset));
        let inst = MInst::Store {
            op,
            src,
            to: mem,
            flags: MemFlagsData::trusted(),
        };
        self.emit(&inst);
        smallvec::smallvec![]
    }

    fn label_to_br_target(&mut self, label: MachLabel) -> CondBrTarget {
        CondBrTarget::Label(label)
    }

    fn int_compare_from_values(&mut self, kind: &IntCC, x: Value, y: Value) -> IntegerCompare {
        let rs1 = self.put_in_reg(x);
        let rs2 = self.put_in_reg(y);
        let ty = self.lower_ctx.dfg().value_type(x);
        IntegerCompare {
            kind: *kind,
            rs1,
            rs2,
            is64: ty.bits() > 32,
        }
    }

    fn int_compare_nonzero(&mut self, val: Value) -> IntegerCompare {
        let rs1 = self.put_in_reg(val);
        let ty = self.lower_ctx.dfg().value_type(val);
        // r0 is never allocated, so it can serve as a zero operand here.
        IntegerCompare {
            kind: IntCC::NotEqual,
            rs1,
            rs2: scratch_reg(),
            is64: ty.bits() > 32,
        }
    }

    fn int_compare_inverse(&mut self, c: IntegerCompare) -> IntegerCompare {
        c.inverse()
    }

    fn gen_icmp(&mut self, cc: &IntCC, x: Value, y: Value) -> Reg {
        let cmp = self.int_compare_from_values(cc, x, y);
        let rd = self.temp_writable_reg(I64);
        self.emit(&MInst::SetCC { rd, cmp });
        rd.to_reg()
    }
}

/// The main entry point for lowering with ISLE.
pub(crate) fn lower(
    lower_ctx: &mut Lower<MInst>,
    backend: &Ppu64Backend,
    inst: Inst,
) -> Option<InstOutput> {
    let mut isle_ctx = Ppu64IsleContext::new(lower_ctx, backend);
    generated_code::constructor_lower(&mut isle_ctx, inst)
}

/// The main entry point for branch lowering with ISLE.
pub(crate) fn lower_branch(
    lower_ctx: &mut Lower<MInst>,
    backend: &Ppu64Backend,
    branch: Inst,
    targets: &[MachLabel],
) -> Option<()> {
    let mut isle_ctx = Ppu64IsleContext::new(lower_ctx, backend);
    generated_code::constructor_lower_branch(&mut isle_ctx, branch, targets)
}
