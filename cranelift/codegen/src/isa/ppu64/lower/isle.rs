//! ISLE integration glue code for ppu64 lowering.

// Pull in the ISLE generated code.
pub mod generated_code;
use generated_code::MInst;

// Types that the generated ISLE code uses via `use super::*`.
use crate::ir::condcodes::{FloatCC, IntCC};
use crate::isa::ppu64::Ppu64Backend;
use crate::machinst::{CallInfo, MachInst, Reg, isle::*};
use crate::machinst::{VCodeConstant, VCodeConstantData};
use crate::{
    ir::{
        BlockCall, ExternalName, Inst, InstructionData, MemFlagsData, Opcode, TrapCode, Value,
        ValueList, immediates::*, types::*,
    },
    isa::ppu64::inst::*,
    machinst::{ArgPair, CallArgList, CallRetList, InstOutput, RetPair},
};
use alloc::boxed::Box;
use alloc::vec::Vec;
use smallvec::smallvec;

use generated_code::Context as _;
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

    fn put_in_freg(&mut self, val: Value) -> Reg {
        self.lower_ctx.put_value_in_regs(val).only_reg().unwrap()
    }

    /// Convert an integer in a GPR to a float. `signed` selects between
    /// `fcfid` and the unsigned hi/lo-split sequence.
    fn int_bits_to_float(&mut self, res_ty: Type, signed: bool, src_ty_bits: u8, reg: Reg) -> Reg {
        let fd = self.temp_writable_reg(F64);

        if signed || src_ty_bits <= 32 {
            // Sign-extended (or known-small non-negative) values convert
            // directly with fcfid.
            let tmp = self.temp_writable_reg(F64);
            self.emit(&MInst::MoveGprToFpr { rd: tmp, rs: reg });
            self.emit(&MInst::FpuRR {
                alu_op: generated_code::FpuOPRR::Fcfid,
                rd: fd,
                rs: tmp.to_reg(),
            });
        } else {
            // Unsigned 64-bit: value = hi32 * 2^32 + lo32.
            // c = 32
            let c = self.temp_writable_reg(I64);
            self.emit(&MInst::Li32 { rd: c, imm: 32 });
            // hi = x >>a 32
            let hi = self.temp_writable_reg(I64);
            self.emit(&MInst::AluRRR {
                alu_op: AluOPRRR::Srd,
                rd: hi,
                rs1: reg,
                rs2: c.to_reg(),
            });
            // lo = zext32(x)
            let lo = self.temp_writable_reg(I64);
            self.emit(&MInst::Extend {
                rd: lo,
                rn: reg,
                signed: false,
                from_bits: 32,
            });
            // ftwo = the bit pattern of the double 2^32
            let two32bits = self.temp_writable_reg(I64);
            self.emit(&MInst::LoadConst64 {
                rd: two32bits,
                imm: 0x41f0_0000_0000_0000,
            });
            let ftwo = self.temp_writable_reg(F64);
            self.emit(&MInst::MoveGprToFpr {
                rd: ftwo,
                rs: two32bits.to_reg(),
            });
            // fh = fcfid(hi) * 2^32
            let fh0 = self.temp_writable_reg(F64);
            self.emit(&MInst::MoveGprToFpr {
                rd: fh0,
                rs: hi.to_reg(),
            });
            let fh1 = self.temp_writable_reg(F64);
            self.emit(&MInst::FpuRR {
                alu_op: generated_code::FpuOPRR::Fcfid,
                rd: fh1,
                rs: fh0.to_reg(),
            });
            let fh = self.temp_writable_reg(F64);
            self.emit(&MInst::FpuRRR {
                alu_op: generated_code::FpuOPRRR::Fmul,
                rd: fh,
                rs1: fh1.to_reg(),
                rs2: ftwo.to_reg(),
            });
            // fl = fcfid(lo)
            let fl0 = self.temp_writable_reg(F64);
            self.emit(&MInst::MoveGprToFpr {
                rd: fl0,
                rs: lo.to_reg(),
            });
            let fl = self.temp_writable_reg(F64);
            self.emit(&MInst::FpuRR {
                alu_op: generated_code::FpuOPRR::Fcfid,
                rd: fl,
                rs: fl0.to_reg(),
            });
            // fd = fh + fl
            self.emit(&MInst::FpuRRR {
                alu_op: generated_code::FpuOPRRR::Fadd,
                rd: fd,
                rs1: fh.to_reg(),
                rs2: fl.to_reg(),
            });
        }

        if res_ty == F32 {
            let r2 = self.temp_writable_reg(F64);
            self.emit(&MInst::FpuRR {
                alu_op: generated_code::FpuOPRR::Frsp,
                rd: r2,
                rs: fd.to_reg(),
            });
            return r2.to_reg();
        }
        fd.to_reg()
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

    //=====================================================================
    // Floating point.

    fn gen_fbinop(
        &mut self,
        op: &generated_code::FpuOPRRR,
        ty: Type,
        x: Value,
        y: Value,
    ) -> InstOutput {
        let xa = self.put_in_freg(x);
        let ya = self.put_in_freg(y);
        let rd = self.temp_writable_reg(F64);
        self.emit(&MInst::FpuRRR {
            alu_op: *op,
            rd,
            rs1: xa,
            rs2: ya,
        });
        if ty == F32 {
            // Round the double-precision result to single precision.
            let r2 = self.temp_writable_reg(F64);
            self.emit(&MInst::FpuRR {
                alu_op: generated_code::FpuOPRR::Frsp,
                rd: r2,
                rs: rd.to_reg(),
            });
            return smallvec![ValueRegs::one(r2.to_reg())];
        }
        smallvec![ValueRegs::one(rd.to_reg())]
    }

    fn gen_funop(&mut self, op: &generated_code::FpuOPRR, x: Value) -> InstOutput {
        let xs = self.put_in_freg(x);
        let rd = self.temp_writable_reg(F64);
        self.emit(&MInst::FpuRR {
            alu_op: *op,
            rd,
            rs: xs,
        });
        smallvec![ValueRegs::one(rd.to_reg())]
    }

    fn gen_fcvt_from_sint(&mut self, res_ty: Type, src_ty: Type, v: Value) -> InstOutput {
        let src_bits = src_ty.bits() as u8;
        let x = self.put_in_reg(v);
        let x2 = if src_ty == I64 {
            x
        } else {
            // Sign-extend sub-64-bit sources to the full register.
            let t = self.temp_writable_reg(I64);
            self.emit(&MInst::Extend {
                rd: t,
                rn: x,
                signed: true,
                from_bits: src_bits,
            });
            t.to_reg()
        };
        let reg = self.int_bits_to_float(res_ty, true, src_bits, x2);
        smallvec![ValueRegs::one(reg)]
    }

    fn gen_fcvt_from_uint(&mut self, res_ty: Type, src_ty: Type, v: Value) -> InstOutput {
        let src_bits = src_ty.bits() as u8;
        let x = self.put_in_reg(v);
        // Zero-extend sub-64-bit sources first; the result is then a
        // non-negative 64-bit value that fcfid handles directly.
        let (x2, known_nonneg) = if src_ty == I64 {
            (x, false)
        } else {
            let t = self.temp_writable_reg(I64);
            self.emit(&MInst::Extend {
                rd: t,
                rn: x,
                signed: false,
                from_bits: src_bits,
            });
            (t.to_reg(), true)
        };
        let reg = self.int_bits_to_float(res_ty, known_nonneg, src_bits, x2);
        smallvec![ValueRegs::one(reg)]
    }

    fn gen_promote(&mut self, v: Value) -> InstOutput {
        let xs = self.put_in_freg(v);
        let rd = self.temp_writable_reg(F64);
        self.emit(&MInst::PromoteF32ToF64 { rd, rs: xs });
        smallvec![ValueRegs::one(rd.to_reg())]
    }

    fn gen_bitcast_i64_f64(&mut self, v: Value) -> InstOutput {
        let xs = self.put_in_reg(v);
        let rd = self.temp_writable_reg(F64);
        self.emit(&MInst::MoveGprToFpr { rd, rs: xs });
        smallvec![ValueRegs::one(rd.to_reg())]
    }

    fn gen_bitcast_f64_i64(&mut self, v: Value) -> InstOutput {
        let xs = self.put_in_freg(v);
        let rd = self.temp_writable_reg(I64);
        self.emit(&MInst::MoveFprToGpr { rd, rs: xs });
        smallvec![ValueRegs::one(rd.to_reg())]
    }

    fn gen_fcmp(&mut self, cc: &FloatCC, x: Value, y: Value) -> Reg {
        let a = self.put_in_freg(x);
        let b = self.put_in_freg(y);
        let rd = self.temp_writable_reg(I64);
        self.emit(&MInst::FCmpSetCC { rd, cc: *cc, a, b });
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
