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

    fn gen_stack_addr(&mut self, ss: StackSlot, offset: Offset32) -> Reg {
        let result = self.temp_writable_reg(I64);
        let inst = self
            .lower_ctx
            .abi()
            .sized_stackslot_addr(ss, i32::from(offset) as u32, result);
        self.emit(&inst);
        result.to_reg()
    }



    /// 32-bit byte swap using four `rlwinm` extractions and ORs.
    fn bswap32(&mut self, x: Reg) -> Reg {
                // rotl24 puts [b3,b0,b1,b2]; rotl8 puts [b1,b2,b3,b0].
        let terms: [(u8, u8, u8); 4] = [
            (24, 0, 7),   // b3 -> bits 31..24
            (8, 8, 15),   // b2 -> bits 23..16
            (24, 16, 23), // b1 -> bits 15..8
            (8, 24, 31),  // b0 -> bits 7..0
        ];
        let mut acc: Option<Reg> = None;
        for &(sh, mb, me) in &terms {
            let t = self.temp_writable_reg(I64);
            self.emit(&MInst::Rlwinm {
                rd: t,
                rn: x,
                sh,
                mb,
                me,
            });
            let d = self.temp_writable_reg(I64);
            let (a, b) = match acc {
                None => {
                    acc = Some(t.to_reg());
                    continue;
                }
                Some(prev) => (prev, t.to_reg()),
            };
            self.emit(&MInst::AluRRR {
                alu_op: AluOPRRR::Or,
                rd: d,
                rs1: a,
                rs2: b,
            });
            acc = Some(d.to_reg());
        }
        acc.unwrap()
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

    fn gen_f64const(&mut self, bits: u64) -> InstOutput {
        let tmp = self.temp_writable_reg(I64);
        self.emit(&MInst::LoadConst64 { rd: tmp, imm: bits });
        let rd = self.temp_writable_reg(F64);
        self.emit(&MInst::MoveGprToFpr {
            rd,
            rs: tmp.to_reg(),
        });
        smallvec![ValueRegs::one(rd.to_reg())]
    }

    fn gen_f32const(&mut self, bits: u32) -> InstOutput {
        let tmp = self.temp_writable_reg(I64);
        self.emit(&MInst::Li32 {
            rd: tmp,
            imm: bits,
        });
        let rd = self.temp_writable_reg(F64);
        self.emit(&MInst::MoveGprToFpr32 {
            rd,
            rs: tmp.to_reg(),
        });
        smallvec![ValueRegs::one(rd.to_reg())]
    }

    fn gen_stack_addr(&mut self, ss: StackSlot, offset: Offset32) -> Reg {
        Ppu64IsleContext::gen_stack_addr(self, ss, offset)
    }

    fn gen_fcmp(&mut self, cc: &FloatCC, x: Value, y: Value) -> Reg {
        let a = self.put_in_freg(x);
        let b = self.put_in_freg(y);
        let rd = self.temp_writable_reg(I64);
        self.emit(&MInst::FCmpSetCC { rd, cc: *cc, a, b });
        rd.to_reg()
    }

    /// Prepare an operand for a 32-bit count operation: zero-extend
    /// 32-bit values so the upper bits are defined; pass 64-bit values
    /// through.
    fn zext_or_sext(&mut self, is64: bool, val: Value) -> Reg {
        if is64 {
            return self.put_in_reg(val);
        }
        let ty = self.lower_ctx.dfg().value_type(val);
        debug_assert!(ty.bits() <= 32);
        let src = self.put_in_reg(val);
        let t = self.temp_writable_reg(I64);
        self.emit(&MInst::Extend {
            rd: t,
            rn: src,
            signed: false,
            from_bits: ty.bits() as u8,
        });
        t.to_reg()
    }

    /// Population count via the classic SWAR sequence.
    fn gen_popcnt(&mut self, ty: Type, v: Value) -> InstOutput {
        let x = self.put_in_reg(v);
        let mut cur = self.temp_writable_reg(ty);
        self.emit(&MInst::gen_move(cur, x, ty));

        // For 32-bit inputs zero-extend first so 64-bit SWAR masks work.
        if ty.bits() <= 32 {
            self.emit(&MInst::Extend {
                rd: cur,
                rn: cur.to_reg(),
                signed: false,
                from_bits: ty.bits() as u8,
            });
        }

        // cur -= (cur >> shift) & mask, for each field width.
        let steps: [u64; 4] = [
            0x5555_5555_5555_5555,
            0x3333_3333_3333_3333,
            0x0f0f_0f0f_0f0f_0f0f,
            0x00ff_00ff_00ff_00ff,
        ];
        let shifts = [1u32, 2, 4, 8];
        for (mask, shift) in steps.iter().zip(shifts.iter()) {
            let mask = *mask;
            let shift = *shift;
            let c = self.temp_writable_reg(I64);
            self.emit(&MInst::Li32 { rd: c, imm: shift });
            let shifted = self.temp_writable_reg(I64);
            self.emit(&MInst::AluRRR {
                alu_op: AluOPRRR::Srd,
                rd: shifted,
                rs1: cur.to_reg(),
                rs2: c.to_reg(),
            });
            let m = self.temp_writable_reg(I64);
            self.emit(&MInst::LoadConst64 { rd: m, imm: mask });
            let masked = self.temp_writable_reg(I64);
            self.emit(&MInst::AluRRR {
                alu_op: AluOPRRR::And,
                rd: masked,
                rs1: shifted.to_reg(),
                rs2: m.to_reg(),
            });
            // subf computes rb - ra: cur - masked.
            let next = self.temp_writable_reg(I64);
            self.emit(&MInst::AluRRR {
                alu_op: AluOPRRR::Subf,
                rd: next,
                rs1: masked.to_reg(),
                rs2: cur.to_reg(),
            });
            let newcur = self.temp_writable_reg(I64);
            self.emit(&MInst::gen_move(newcur, next.to_reg(), I64));
            cur = newcur;
        }

        // Fold bytes: cur * 0x0101...01 >> 56.
        let m = self.temp_writable_reg(I64);
        self.emit(&MInst::LoadConst64 {
            rd: m,
            imm: 0x0101_0101_0101_0101,
        });
        let prod = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::Mulld,
            rd: prod,
            rs1: cur.to_reg(),
            rs2: m.to_reg(),
        });
        let c56 = self.temp_writable_reg(I64);
        self.emit(&MInst::Li32 { rd: c56, imm: 56 });
        let result = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::Srd,
            rd: result,
            rs1: prod.to_reg(),
            rs2: c56.to_reg(),
        });
        smallvec![ValueRegs::one(result.to_reg())]
    }

    /// Byte swap built from rlwinm field extractions.
    fn gen_bswap(&mut self, ty: Type, v: Value) -> InstOutput {
        let x = self.put_in_reg(v);

        match ty.bits() {
            16 => {
                // Swap bytes within the low 16 bits:
                // ((x & 0xff) <<a 8) | (x >>a 8) & 0x00ff.
                let sh = self.temp_writable_reg(I64);
                self.emit(&MInst::Li32 { rd: sh, imm: 8 });
                let lo_byte = self.temp_writable_reg(I64);
                self.emit(&MInst::Rlwinm {
                    rd: lo_byte,
                    rn: x,
                    sh: 24,
                    mb: 24,
                    me: 31,
                }); // x & 0xff moved to bits 31..24
                // wait: rlwinm sh=24 puts low byte at top of the 32-bit
                // word; we want it in bits 15..8 instead. Do it with two
                // extractions and an or instead.
                let b0 = self.temp_writable_reg(I64);
                self.emit(&MInst::Rlwinm {
                    rd: b0,
                    rn: lo_byte.to_reg(),
                    sh: 0,
                    mb: 16,
                    me: 23,
                }); // place byte into bits 23..16 -> i.e. value << 8 within low half? no.
                let _ = b0;
                // Simpler: use register shifts.
                let shl = self.temp_writable_reg(I64);
                self.emit(&MInst::AluRRR {
                    alu_op: AluOPRRR::Slw,
                    rd: shl,
                    rs1: x,
                    rs2: sh.to_reg(),
                }); // x <<a 8 (mod 2^32)
                let hi_part = self.temp_writable_reg(I64);
                self.emit(&MInst::Rlwinm {
                    rd: hi_part,
                    rn: shl.to_reg(),
                    sh: 0,
                    mb: 16,
                    me: 23,
                }); // keep bits 23..16 => (x<<8)&0xff00
                let shr = self.temp_writable_reg(I64);
                self.emit(&MInst::AluRRR {
                    alu_op: AluOPRRR::Srw,
                    rd: shr,
                    rs1: x,
                    rs2: sh.to_reg(),
                }); // x >>a 8
                let lo_part = self.temp_writable_reg(I64);
                self.emit(&MInst::Rlwinm {
                    rd: lo_part,
                    rn: shr.to_reg(),
                    sh: 0,
                    mb: 24,
                    me: 31,
                }); // & 0x00ff
                let dst = self.temp_writable_reg(I64);
                self.emit(&MInst::AluRRR {
                    alu_op: AluOPRRR::Or,
                    rd: dst,
                    rs1: hi_part.to_reg(),
                    rs2: lo_part.to_reg(),
                });
                smallvec![ValueRegs::one(dst.to_reg())]
            }
            32 => {
                let d = self.bswap32(x);
                smallvec![ValueRegs::one(d)]
            }
            _ => {
                // 64-bit: swap each 32-bit half and reassemble.
                let hi = self.temp_writable_reg(I64);
                self.emit(&MInst::Extend {
                    rd: hi,
                    rn: x,
                    signed: false,
                    from_bits: 32,
                });
                let c32 = self.temp_writable_reg(I64);
                self.emit(&MInst::Li32 { rd: c32, imm: 32 });
                let lo = self.temp_writable_reg(I64);
                self.emit(&MInst::AluRRR {
                    alu_op: AluOPRRR::Srd,
                    rd: lo,
                    rs1: x,
                    rs2: c32.to_reg(),
                });

                let hi_swapped = self.bswap32(hi.to_reg());
                let lo_swapped = self.bswap32(lo.to_reg());

                // result = (lo_swapped << 32) | hi_swapped
                let shifted = self.temp_writable_reg(I64);
                self.emit(&MInst::AluRRR {
                    alu_op: AluOPRRR::Sld,
                    rd: shifted,
                    rs1: lo_swapped,
                    rs2: c32.to_reg(),
                });
                let dst = self.temp_writable_reg(I64);
                self.emit(&MInst::AluRRR {
                    alu_op: AluOPRRR::Or,
                    rd: dst,
                    rs1: shifted.to_reg(),
                    rs2: hi_swapped,
                });
                smallvec![ValueRegs::one(dst.to_reg())]
            }
        }
    }

    /// Integer select: cond != 0 ? x : y via sign-mask AND/OR.
    fn gen_select_int(&mut self, c: Value, x: Value, y: Value) -> InstOutput {
        let xr = self.put_in_reg(x);
        let yr = self.put_in_reg(y);

        // flag = setcc(cond != 0)
        let cmp = self.int_compare_nonzero(c);
        let flag = self.temp_writable_reg(I64);
        self.emit(&MInst::SetCC { rd: flag, cmp });
        // mask = -(flag)
        let mask = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRR {
            alu_op: AluOPRR::Neg,
            rd: mask,
            rs: flag.to_reg(),
        });
        // notmask = ~mask
        let notmask = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::Nor,
            rd: notmask,
            rs1: mask.to_reg(),
            rs2: mask.to_reg(),
        });
        // tx = x & mask ; ty = y & ~mask
        let tx = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::And,
            rd: tx,
            rs1: xr,
            rs2: mask.to_reg(),
        });
        let ty2 = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::And,
            rd: ty2,
            rs1: yr,
            rs2: notmask.to_reg(),
        });
        let dst = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::Or,
            rd: dst,
            rs1: tx.to_reg(),
            rs2: ty2.to_reg(),
        });
        smallvec![ValueRegs::one(dst.to_reg())]
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
