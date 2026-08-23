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

    /// fctiwz + extract the i32 result to a GPR.
    fn fpr_to_i32(&mut self, f: Reg) -> Reg {
        let t = self.fpu_unop(generated_code::FpuOPRR::Fctiwz, f);
        let rd = self.temp_writable_reg(I64);
        self.emit(&MInst::MoveFprToGprI32 { rd, rs: t });
        rd.to_reg()
    }

    /// Sign-extend a 32-bit GPR value and convert it to a double.
    fn i32_to_fpr(&mut self, g: Reg) -> Reg {
        let wide = self.temp_writable_reg(I64);
        self.emit(&MInst::Extend {
            rd: wide,
            rn: g,
            signed: true,
            from_bits: 32,
        });
        let asbits = self.temp_writable_reg(F64);
        self.emit(&MInst::MoveGprToFpr {
            rd: asbits,
            rs: wide.to_reg(),
        });
        self.fpu_unop(generated_code::FpuOPRR::Fcfid, asbits.to_reg())
    }

    /// fcvt_to_sint_sat.i32: clamp to [-2^31, 2^31-1], NaN -> 0.
    fn gen_fcvt_to_sint_sat_i32(&mut self, v: Value) -> InstOutput {
        let d = self.put_in_freg(v);
        let zero_d = self.fpr_zero();

        // conv = (i32) d, wraparound semantics via fctiwz.
        let conv_i = self.fpr_to_i32(d);

        // diff = d - (double)conv ; positive iff overflow above INT32_MAX.
        let rconv = self.i32_to_fpr(conv_i);
        let neg_two31 = self.fpr_const(0xc1e0_0000_0000_0000); // -2^31
        let diff_clamp = self.fpu_binop(generated_code::FpuOPRRR::Fsub, d, neg_two31);
        let d_clamped = self.fsel(diff_clamp, d, neg_two31); // max(d, -2^31)
        let diff = self.fpu_binop(generated_code::FpuOPRRR::Fsub, d_clamped, rconv);

        let flag = self.temp_writable_reg(I64);
        self.emit(&MInst::FCmpSetCC {
            rd: flag,
            cc: FloatCC::GreaterThan,
            a: diff,
            b: zero_d,
        });
        let mask = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRR {
            alu_op: AluOPRR::Neg,
            rd: mask,
            rs: flag.to_reg(),
        });
        let res = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::Xor,
            rd: res,
            rs1: conv_i,
            rs2: mask.to_reg(),
        });

        // NaN -> 0.
        let nanflag = self.isnan_flag(d);
        let zero_i = self.temp_writable_reg(I64);
        self.emit(&MInst::Li32 { rd: zero_i, imm: 0 });
        let final_res = self.select_on_flag(nanflag, zero_i.to_reg(), res.to_reg());
        smallvec![ValueRegs::one(final_res)]
    }

    /// fcvt_to_uint_sat.i32: clamp negatives to 0, wrap [2^31, 2^32),
    /// saturate >= 2^32 to 0xFFFFFFFF, NaN -> 0.
    fn gen_fcvt_to_uint_sat_i32(&mut self, v: Value) -> InstOutput {
        let d = self.put_in_freg(v);
        let zero_d = self.fpr_zero();
        let one_d = self.fpr_const(0x3ff0_0000_0000_0000);

        // t = max(d, 0): fsel picks C (=d) when A (=d) >= 0 or NaN.
        let t = self.fsel(d, d, zero_d);

        // conv = (i32) t, wrapping for values >= 2^31.
        let conv_i = self.fpr_to_i32(t);

        // diff1 = t - (double)conv ; > 0 iff wrapped past 2^31.
        let rconv = self.i32_to_fpr(conv_i);
        let diff1 = self.fpu_binop(generated_code::FpuOPRRR::Fsub, t, rconv);

        let flag1 = self.temp_writable_reg(I64);
        self.emit(&MInst::FCmpSetCC {
            rd: flag1,
            cc: FloatCC::GreaterThan,
            a: diff1,
            b: zero_d,
        });
        let c31 = self.temp_writable_reg(I64);
        self.emit(&MInst::Li32 { rd: c31, imm: 31 });
        let signA = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::Sld,
            rd: signA,
            rs1: flag1.to_reg(),
            rs2: c31.to_reg(),
        });
        let part = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::Xor,
            rd: part,
            rs1: conv_i,
            rs2: signA.to_reg(),
        });

        // corrB: t >= 2^32 -> force all-ones.
        let twop32 = self.fpr_const(0x41f0_0000_0000_0000); // 2^32
        let flag2 = self.temp_writable_reg(I64);
        self.emit(&MInst::FCmpSetCC {
            rd: flag2,
            cc: FloatCC::GreaterThanOrEqual,
            a: t,
            b: twop32,
        });
        let all_ones = self.temp_writable_reg(I64);
        self.emit(&MInst::LoadConst64 {
            rd: all_ones,
            imm: 0xffff_ffff,
        });
        let part2_res = self.select_on_flag(flag2.to_reg(), all_ones.to_reg(), part.to_reg());

        // NaN -> 0.
        let nanflag = self.isnan_flag(d);
        let zero_i = self.temp_writable_reg(I64);
        self.emit(&MInst::Li32 { rd: zero_i, imm: 0 });
        let final_res = self.select_on_flag(nanflag, zero_i.to_reg(), part2_res);
        smallvec![ValueRegs::one(final_res)]
    }

    /// fnearest via the magic-number trick: adding/subtracting 2^52
    /// forces the FPU's own round-to-nearest-even at integer precision
    /// for |x| < 2^52; bigger magnitudes are already integers, and NaN
    /// passes through unchanged.
    fn gen_fnearest(&mut self, v: Value) -> InstOutput {
        use generated_code::FpuOPRR as O;
        let x = self.put_in_freg(v);
        let zero_d = self.fpr_zero();
        let one_d = self.fpr_const(0x3ff0_0000_0000_0000);
        let two52 = self.fpr_const(0x4330_0000_0000_0000); // 2^52
        let mtwo52 = self.fpr_const(0xc330_0000_0000_0000); // -2^52

        // s = +2^52 when x >= 0, else -2^52 (fsel takes C on NaN).
        let s = self.fsel(x, two52, mtwo52);

        // r = |round(x)| computed in the magnitude domain.
        let ax = self.fpu_unop(O::Fabs, x);
        let y = self.fpu_binop(generated_code::FpuOPRRR::Fadd, ax, two52);
        let rmag = self.fpu_binop(generated_code::FpuOPRRR::Fsub, y, two52);

        // Reapply the sign bit from x.
        let rmag_bits = self.temp_writable_reg(I64);
        self.emit(&MInst::MoveFprToGpr {
            rd: rmag_bits,
            rs: rmag,
        });
        let x_bits = self.temp_writable_reg(I64);
        self.emit(&MInst::MoveFprToGpr { rd: x_bits, rs: x });

        let sign_mask_c = self.temp_writable_reg(I64);
        self.emit(&MInst::LoadConst64 {
            rd: sign_mask_c,
            imm: 0x8000_0000_0000_0000,
        });
        let sign = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::And,
            rd: sign,
            rs1: x_bits.to_reg(),
            rs2: sign_mask_c.to_reg(),
        });
        let mag_signed = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::Or,
            rd: mag_signed,
            rs1: rmag_bits.to_reg(),
            rs2: sign.to_reg(),
        });

        // Guard: only trust the trick when |x| < 2^52.
        let sub52 = self.fpu_binop(generated_code::FpuOPRRR::Fsub, ax, two52);
        let small_i = self.temp_writable_reg(I64);
        self.emit(&MInst::FCmpSetCC {
            rd: small_i,
            cc: FloatCC::LessThan,
            a: sub52,
            b: zero_d,
        });

        // Normalize small flag to exactly 0/1 by comparing its bits...
        // simpler: nonzero means small; select_on_flag treats any
        // nonzero as "one" branch since mask = -(flag != 0).
        let near_bits =
            self.select_on_flag(small_i.to_reg(), mag_signed.to_reg(), x_bits.to_reg());

        // NaN -> x (still NaN).
        let nanflag = self.isnan_flag(x);
        let res = self.select_on_flag(nanflag, x_bits.to_reg(), near_bits);
        let out = self.temp_writable_reg(F64);
        self.emit(&MInst::MoveGprToFpr { rd: out, rs: res });
        smallvec![ValueRegs::one(out.to_reg())]
    }

    fn gen_fcmp(&mut self, cc: &FloatCC, x: Value, y: Value) -> Reg {
        let a = self.put_in_freg(x);
        let b = self.put_in_freg(y);
        let rd = self.temp_writable_reg(I64);
        self.emit(&MInst::FCmpSetCC { rd, cc: *cc, a, b });
        rd.to_reg()
    }


    /// Materialize a double constant into an FPR.
    fn fpr_const(&mut self, bits: u64) -> Reg {
        let tmp = self.temp_writable_reg(I64);
        self.emit(&MInst::LoadConst64 { rd: tmp, imm: bits });
        let rd = self.temp_writable_reg(F64);
        self.emit(&MInst::MoveGprToFpr {
            rd,
            rs: tmp.to_reg(),
        });
        rd.to_reg()
    }

    fn fpr_zero(&mut self) -> Reg {
        let tmp = self.temp_writable_reg(I64);
        self.emit(&MInst::Li32 { rd: tmp, imm: 0 });
        let rd = self.temp_writable_reg(F64);
        self.emit(&MInst::MoveGprToFpr32 {
            rd,
            rs: tmp.to_reg(),
        });
        rd.to_reg()
    }

    fn gen_fma(&mut self, x: Value, y: Value, z: Value) -> InstOutput {
        let xa = self.put_in_freg(x);
        let ya = self.put_in_freg(y);
        let za = self.put_in_freg(z);
        let rd = self.temp_writable_reg(F64);
        self.emit(&MInst::FpuRRRR {
            alu_op: generated_code::FpuOPRRRR::Fmadd,
            rd,
            ra: xa,
            rb: za,
            rc: ya,
        });
        smallvec![ValueRegs::one(rd.to_reg())]
    }

    /// fsel: rd = (a >= 0 or NaN) ? c : b
    fn fsel(&mut self, a: Reg, c: Reg, b: Reg) -> Reg {
        let rd = self.temp_writable_reg(F64);
        self.emit(&MInst::FpuRRRR {
            alu_op: generated_code::FpuOPRRRR::Fsel,
            rd,
            ra: a,
            rb: b,
            rc: c,
        });
        rd.to_reg()
    }

    fn fpu_unop(&mut self, op: generated_code::FpuOPRR, x: Reg) -> Reg {
        let rd = self.temp_writable_reg(F64);
        self.emit(&MInst::FpuRR { alu_op: op, rd, rs: x });
        rd.to_reg()
    }

    fn fpu_binop(&mut self, op: generated_code::FpuOPRRR, a: Reg, b: Reg) -> Reg {
        let rd = self.temp_writable_reg(F64);
        self.emit(&MInst::FpuRRR {
            alu_op: op,
            rd,
            rs1: a,
            rs2: b,
        });
        rd.to_reg()
    }

    /// fctidz + move to GPR.
    fn fpr_to_i64_bits(&mut self, f: Reg) -> Reg {
        let t = self.fpu_unop(generated_code::FpuOPRR::Fctidz, f);
        let rd = self.temp_writable_reg(I64);
        self.emit(&MInst::MoveFprToGpr { rd, rs: t });
        rd.to_reg()
    }

    /// Select between two GPR values on a 0/1 flag.
    fn select_on_flag(&mut self, flag: Reg, if_one: Reg, if_zero: Reg) -> Reg {
        // mask = -(flag)
        let mask = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRR {
            alu_op: AluOPRR::Neg,
            rd: mask,
            rs: flag,
        });
        // notmask = ~mask
        let notmask = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::Nor,
            rd: notmask,
            rs1: mask.to_reg(),
            rs2: mask.to_reg(),
        });
        let a = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::And,
            rd: a,
            rs1: if_one,
            rs2: mask.to_reg(),
        });
        let b = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::And,
            rd: b,
            rs1: if_zero,
            rs2: notmask.to_reg(),
        });
        let dst = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::Or,
            rd: dst,
            rs1: a.to_reg(),
            rs2: b.to_reg(),
        });
        dst.to_reg()
    }

    fn isnan_flag(&mut self, x: Reg) -> Reg {
        let rd = self.temp_writable_reg(I64);
        self.emit(&MInst::FCmpSetCC {
            rd,
            cc: FloatCC::Unordered,
            a: x,
            b: x,
        });
        rd.to_reg()
    }

    fn gen_fcvt_to_sint_sat(&mut self, v: Value) -> InstOutput {
        let d = self.put_in_freg(v);
        let two63 = self.fpr_const(0x43e0_0000_0000_0000);
        let neg_two63 = self.fpr_const(0xc3e0_0000_0000_0000);
        let flag_high = self.temp_writable_reg(I64);
        self.emit(&MInst::FCmpSetCC {
            rd: flag_high,
            cc: FloatCC::GreaterThanOrEqual,
            a: d,
            b: two63,
        });
        let flag_low = self.temp_writable_reg(I64);
        self.emit(&MInst::FCmpSetCC {
            rd: flag_low,
            cc: FloatCC::LessThan,
            a: d,
            b: neg_two63,
        });
        let conv_f = self.fpu_unop(generated_code::FpuOPRR::Fctidz, d);
        let conv_i = self.temp_writable_reg(I64);
        self.emit(&MInst::MoveFprToGpr {
            rd: conv_i,
            rs: conv_f,
        });
        let res_max = self.temp_writable_reg(I64);
        self.emit(&MInst::LoadConst64 {
            rd: res_max,
            imm: 0x7fff_ffff_ffff_ffff,
        });
        let res_min = self.temp_writable_reg(I64);
        self.emit(&MInst::LoadConst64 {
            rd: res_min,
            imm: 0x8000_0000_0000_0000,
        });
        let tmp = self.select_on_flag(flag_high.to_reg(), res_max.to_reg(), conv_i.to_reg());
        let res = self.select_on_flag(flag_low.to_reg(), res_min.to_reg(), tmp);

        // NaN -> 0.
        let nanflag = self.isnan_flag(d);
        let zero_i = self.temp_writable_reg(I64);
        self.emit(&MInst::Li32 { rd: zero_i, imm: 0 });
        let final_res = self.select_on_flag(nanflag, zero_i.to_reg(), res);
        smallvec![ValueRegs::one(final_res)]
    }

    fn gen_fcvt_to_uint_sat(&mut self, v: Value) -> InstOutput {
        let d = self.put_in_freg(v);
        let zero_d = self.fpr_zero();
        let one_d = self.fpr_const(0x3ff0_0000_0000_0000); // 1.0

        // Clamp negatives away: t = max(d, 0). fsel picks C (=d) when
        // A (=d) >= 0; negatives take B (=0).
        let t = self.fsel(d, d, zero_d);

        let two63 = self.fpr_const(0x43e0_0000_0000_0000);
        let two64 = self.fpr_const(0x43f0_0000_0000_0000);
        // flag_ge64 = t >= 2^64
        let flag_ge64 = self.temp_writable_reg(I64);
        self.emit(&MInst::FCmpSetCC {
            rd: flag_ge64,
            cc: FloatCC::GreaterThanOrEqual,
            a: t,
            b: two64,
        });
        // flag_ge63 = t >= 2^63
        let flag_ge63 = self.temp_writable_reg(I64);
        self.emit(&MInst::FCmpSetCC {
            rd: flag_ge63,
            cc: FloatCC::GreaterThanOrEqual,
            a: t,
            b: two63,
        });
        // conv for t < 2^63
        let conv_f = self.fpu_unop(generated_code::FpuOPRR::Fctidz, t);
        let conv_i = self.temp_writable_reg(I64);
        self.emit(&MInst::MoveFprToGpr {
            rd: conv_i,
            rs: conv_f,
        });
        // conv for t >= 2^63 : (t - 2^63) + 0x8000...
        let t_sub63 = self.fpu_binop(generated_code::FpuOPRRR::Fsub, t, two63);
        let conv_sub_f = self.fpu_unop(generated_code::FpuOPRR::Fctidz, t_sub63);
        let conv_sub_i = self.temp_writable_reg(I64);
        self.emit(&MInst::MoveFprToGpr {
            rd: conv_sub_i,
            rs: conv_sub_f,
        });
        let high_bit = self.temp_writable_reg(I64);
        self.emit(&MInst::LoadConst64 {
            rd: high_bit,
            imm: 0x8000_0000_0000_0000,
        });
        let res_ge63 = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::Or,
            rd: res_ge63,
            rs1: conv_sub_i.to_reg(),
            rs2: high_bit.to_reg(),
        });
        let res_mid = self.select_on_flag(flag_ge63.to_reg(), res_ge63.to_reg(), conv_i.to_reg());
        let all_ones = self.temp_writable_reg(I64);
        self.emit(&MInst::LoadConst64 {
            rd: all_ones,
            imm: 0xffff_ffff_ffff_ffff,
        });
        let res = self.select_on_flag(flag_ge64.to_reg(), all_ones.to_reg(), res_mid);

        // NaN -> 0.
        let nanflag = self.isnan_flag(d);
        let zero_i = self.temp_writable_reg(I64);
        self.emit(&MInst::Li32 { rd: zero_i, imm: 0 });
        let final_res = self.select_on_flag(nanflag, zero_i.to_reg(), res);
        smallvec![ValueRegs::one(final_res)]
    }

    /// Synthesized integral rounding: trunc/floor/ceil. The PPU has no
    /// fri* instructions, so round-trip through fctidz and correct by
    /// the fractional part, guarding big/NaN inputs at the bit level.
    fn gen_fround(&mut self, kind: u8, v: Value) -> InstOutput {
        use generated_code::FpuOPRR as O;
        let x = self.put_in_freg(v);

        // t = trunc(x), valid for |x| < 2^63.
        let conv = self.fpu_unop(O::Fctidz, x);
        let t = self.fpu_unop(O::Fcfid, conv);

        // frac = x - t.
        let frac = self.fpu_binop(generated_code::FpuOPRRR::Fsub, x, t);

        let computed = match kind {
            0 => t,
            // floor: frac < 0 -> t - 1.
            1 => {
                let nfrac = self.fpu_unop(O::Fneg, frac);
                let zero_d = self.fpr_zero();
                let minus_one = self.fpr_const(0xbff0_0000_0000_0000); // -1.0
                // fsel picks C when A >= 0: nfrac >= 0 (frac <= 0) -> 0;
                // nfrac < 0 (frac > 0 impossible for floor)... we need
                // frac < 0 -> -1: fsel(frac, ZERO, MINUSONE).
                let corr = self.fsel(frac, zero_d, minus_one);
                self.fpu_binop(generated_code::FpuOPRRR::Fadd, t, corr)
            }
            // ceil: frac > 0 -> t + 1.
            _ => {
                let zero_d = self.fpr_zero();
                let one_d = self.fpr_const(0x3ff0_0000_0000_0000);
                let nfrac = self.fpu_unop(O::Fneg, frac);
                // nfrac < 0 iff frac > 0 -> pick ONE.
                let corr = self.fsel(nfrac, zero_d, one_d);
                self.fpu_binop(generated_code::FpuOPRRR::Fadd, t, corr)
            }
        };

        let comp_bits = self.temp_writable_reg(I64);
        self.emit(&MInst::MoveFprToGpr {
            rd: comp_bits,
            rs: computed,
        });

        // Guard: |x| >= 2^63 or NaN must return x unchanged.
        let axm = self.fpu_unop(O::Fabs, x);
        let two63 = self.fpr_const(0x43e0_0000_0000_0000);
        let guard_i = self.temp_writable_reg(I64);
        self.emit(&MInst::FCmpSetCC {
            rd: guard_i,
            cc: FloatCC::UnorderedOrGreaterThanOrEqual,
            a: axm,
            b: two63,
        });

        let x_bits = self.temp_writable_reg(I64);
        self.emit(&MInst::MoveFprToGpr { rd: x_bits, rs: x });

        let res = self.select_on_flag(guard_i.to_reg(), x_bits.to_reg(), comp_bits.to_reg());
        let out = self.temp_writable_reg(F64);
        self.emit(&MInst::MoveGprToFpr { rd: out, rs: res });
        smallvec![ValueRegs::one(out.to_reg())]
    }

    fn gen_fminmax(&mut self, is_max: bool, x: Value, y: Value) -> InstOutput {
        let xa = self.put_in_freg(x);
        let ya = self.put_in_freg(y);
        let cc = if is_max {
            FloatCC::GreaterThan
        } else {
            FloatCC::LessThan
        };
        let flag = self.temp_writable_reg(I64);
        self.emit(&MInst::FCmpSetCC {
            rd: flag,
            cc,
            a: xa,
            b: ya,
        });
        let xb = self.temp_writable_reg(I64);
        self.emit(&MInst::MoveFprToGpr { rd: xb, rs: xa });
        let yb = self.temp_writable_reg(I64);
        self.emit(&MInst::MoveFprToGpr { rd: yb, rs: ya });
        let res = self.select_on_flag(flag.to_reg(), xb.to_reg(), yb.to_reg());
        let out = self.temp_writable_reg(F64);
        self.emit(&MInst::MoveGprToFpr { rd: out, rs: res });
        smallvec![ValueRegs::one(out.to_reg())]
    }

    fn gen_bitcast_f32_i32(&mut self, v: Value) -> InstOutput {
        let xs = self.put_in_reg(v);
        let rd = self.temp_writable_reg(F64);
        self.emit(&MInst::MoveGprToFpr32 { rd, rs: xs });
        smallvec![ValueRegs::one(rd.to_reg())]
    }

    fn gen_bitcast_i32_f32(&mut self, v: Value) -> InstOutput {
        let xs = self.put_in_freg(v);
        let rd = self.temp_writable_reg(I64);
        self.emit(&MInst::MoveFprBitsToGpr32 { rd, rs: xs });
        smallvec![ValueRegs::one(rd.to_reg())]
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

    fn gen_bmask(&mut self, ty: Type, x: Value) -> InstOutput {
        let flag = self.temp_writable_reg(I64);
        let cmp = self.int_compare_nonzero(x);
        self.emit(&MInst::SetCC { rd: flag, cmp });
        let mask = self.temp_writable_reg(ty);
        self.emit(&MInst::AluRR {
            alu_op: AluOPRR::Neg,
            rd: mask,
            rs: flag.to_reg(),
        });
        smallvec![ValueRegs::one(mask.to_reg())]
    }

    fn gen_bitrev(&mut self, ty: Type, x: Value) -> InstOutput {
        let bits = ty.bits() as u8;
        let src = self.put_in_reg(x);
        let mut cur = self.temp_writable_reg(ty);
        self.emit(&MInst::gen_move(cur, src, ty));
        let masks = [
            (0x5555555555555555u64, 1),
            (0x3333333333333333u64, 2),
            (0x0f0f0f0f0f0f0f0f_u64, 4),
            (0x00ff00ff00ff00ff_u64, 8),
        ];
        for (m, s) in masks.iter() {
            let mask = self.temp_writable_reg(I64);
            self.emit(&MInst::LoadConst64 { rd: mask, imm: *m });
            let sh = self.temp_writable_reg(I64);
            self.emit(&MInst::Li32 { rd: sh, imm: *s as u32 });
            let t = self.temp_writable_reg(I64);
            self.emit(&MInst::AluRRR {
                alu_op: AluOPRRR::Srd,
                rd: t,
                rs1: cur.to_reg(),
                rs2: sh.to_reg(),
            });
            let a = self.temp_writable_reg(I64);
            self.emit(&MInst::AluRRR {
                alu_op: AluOPRRR::And,
                rd: a,
                rs1: t.to_reg(),
                rs2: mask.to_reg(),
            });
            let b = self.temp_writable_reg(I64);
            self.emit(&MInst::AluRRR {
                alu_op: AluOPRRR::And,
                rd: b,
                rs1: cur.to_reg(),
                rs2: mask.to_reg(),
            });
            let sh2 = self.temp_writable_reg(I64);
            self.emit(&MInst::Li32 { rd: sh2, imm: *s as u32 });
            let b_sh = self.temp_writable_reg(I64);
            self.emit(&MInst::AluRRR {
                alu_op: AluOPRRR::Sld,
                rd: b_sh,
                rs1: b.to_reg(),
                rs2: sh2.to_reg(),
            });
            let res = self.temp_writable_reg(I64);
            self.emit(&MInst::AluRRR {
                alu_op: AluOPRRR::Or,
                rd: res,
                rs1: a.to_reg(),
                rs2: b_sh.to_reg(),
            });
            cur = res;
        }
        smallvec![ValueRegs::one(cur.to_reg())]
    }

    fn gen_sminmax(&mut self, ty: Type, x: Value, y: Value, is_signed: bool, is_max: bool) -> InstOutput {
        let cc = if is_signed {
            if is_max { IntCC::SignedGreaterThan } else { IntCC::SignedLessThan }
        } else {
            if is_max { IntCC::UnsignedGreaterThan } else { IntCC::UnsignedLessThan }
        };
        let flag = self.temp_writable_reg(I64);
        let cmp = self.int_compare_from_values(&cc, x, y);
        self.emit(&MInst::SetCC { rd: flag, cmp });
        let yr = self.put_in_reg(y);
        let xr = self.put_in_reg(x);
        let selected = self.select_on_flag(flag.to_reg(), yr, xr);
        smallvec![ValueRegs::one(selected)]
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
    fn vreg_new(&mut self, r: Reg) -> VReg {
        VReg::new(r).unwrap()
    }
    fn writable_vreg_new(&mut self, r: WritableReg) -> WritableVReg {
        r.map(|wr| VReg::new(wr).unwrap())
    }
    fn writable_vreg_to_writable_reg(&mut self, arg0: WritableVReg) -> WritableReg {
        arg0.map(|vr| vr.to_reg())
    }
    fn vreg_to_reg(&mut self, arg0: VReg) -> Reg {
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
        let ty = self.lower_ctx.dfg().value_type(p);
        let reg = if ty.bits() == 32 {
            let src = self.put_in_reg(p);
            let tmp = self.temp_writable_reg(I64);
            self.emit(&MInst::Extend {
                rd: tmp,
                rn: src,
                signed: false,
                from_bits: 32,
            });
            tmp.to_reg()
        } else {
            self.put_in_reg(p)
        };
        AMode::RegOffset(reg, i64::from(offset))
    }

    fn gen_store_default(&mut self, val: Value, p: Value, offset: Offset32) -> InstOutput {
        let ty = self.lower_ctx.dfg().value_type(val);
        let op = StoreOP::from_type(ty);
        let src = self.put_in_reg(val);
        let p_ty = self.lower_ctx.dfg().value_type(p);
        let addr = if p_ty.bits() == 32 {
            let src_p = self.put_in_reg(p);
            let tmp = self.temp_writable_reg(I64);
            self.emit(&MInst::Extend {
                rd: tmp,
                rn: src_p,
                signed: false,
                from_bits: 32,
            });
            tmp.to_reg()
        } else {
            self.put_in_reg(p)
        };
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
        // PowerPC has no hardwired zero register readable as a general
        // operand -- materialize a literal zero instead.
        let zero_v = self.temp_writable_reg(I64);
        self.emit(&MInst::Li32 { rd: zero_v, imm: 0 });
        IntegerCompare {
            kind: IntCC::NotEqual,
            rs1,
            rs2: zero_v.to_reg(),
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

    fn gen_fcvt_to_sint_sat(&mut self, v: Value) -> InstOutput {
        Ppu64IsleContext::gen_fcvt_to_sint_sat(self, v)
    }

    fn gen_fcvt_to_uint_sat(&mut self, v: Value) -> InstOutput {
        Ppu64IsleContext::gen_fcvt_to_uint_sat(self, v)
    }

    fn gen_fround(&mut self, kind: u8, v: Value) -> InstOutput {
        Ppu64IsleContext::gen_fround(self, kind, v)
    }

    fn gen_fnearest(&mut self, v: Value) -> InstOutput {
        Ppu64IsleContext::gen_fnearest(self, v)
    }

    fn gen_fma(&mut self, x: Value, y: Value, z: Value) -> InstOutput {
        Ppu64IsleContext::gen_fma(self, x, y, z)
    }

    fn gen_fminmax(&mut self, is_max: bool, x: Value, y: Value) -> InstOutput {
        Ppu64IsleContext::gen_fminmax(self, is_max, x, y)
    }

    fn gen_bitcast_f32_i32(&mut self, v: Value) -> InstOutput {
        Ppu64IsleContext::gen_bitcast_f32_i32(self, v)
    }

    fn gen_bitcast_i32_f32(&mut self, v: Value) -> InstOutput {
        Ppu64IsleContext::gen_bitcast_i32_f32(self, v)
    }

    fn gen_fcmp(&mut self, cc: &FloatCC, x: Value, y: Value) -> Reg {
        Ppu64IsleContext::gen_fcmp(self, cc, x, y)
    }

    fn gen_fcvt_to_sint_sat_i32(&mut self, v: Value) -> InstOutput {
        Ppu64IsleContext::gen_fcvt_to_sint_sat_i32(self, v)
    }

    fn gen_fcvt_to_uint_sat_i32(&mut self, v: Value) -> InstOutput {
        Ppu64IsleContext::gen_fcvt_to_uint_sat_i32(self, v)
    }

    fn gen_bmask(&mut self, ty: Type, x: Value) -> InstOutput {
        Ppu64IsleContext::gen_bmask(self, ty, x)
    }

    fn gen_bitrev(&mut self, ty: Type, x: Value) -> InstOutput {
        Ppu64IsleContext::gen_bitrev(self, ty, x)
    }

    fn gen_sminmax(&mut self, ty: Type, x: Value, y: Value, is_signed: bool, is_max: bool) -> InstOutput {
        Ppu64IsleContext::gen_sminmax(self, ty, x, y, is_signed, is_max)
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

    /// Population count via the canonical SWAR sequence (Hacker's Delight):
    ///   v = x - ((x>>1) & m1)
    ///   v = (v & m2) + ((v>>2) & m2)
    ///   v = (v + (v>>4)) & m3
    ///   return (v * 0x0101010101010101) >> 56
    fn gen_popcnt(&mut self, ty: Type, v: Value) -> InstOutput {
        let x = self.put_in_reg(v);
        let mut cur = self.temp_writable_reg(ty);
        self.emit(&MInst::gen_move(cur, x, ty));

        if ty.bits() <= 32 {
            self.emit(&MInst::Extend {
                rd: cur,
                rn: cur.to_reg(),
                signed: false,
                from_bits: ty.bits() as u8,
            });
        }

        // Step 1: cur -= (cur >> 1) & 0x5555...
        let m1 = self.temp_writable_reg(I64);
        self.emit(&MInst::LoadConst64 { rd: m1, imm: 0x5555_5555_5555_5555 });
        let c1 = self.temp_writable_reg(I64);
        self.emit(&MInst::Li32 { rd: c1, imm: 1 });
        let sh1 = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::Srd, rd: sh1, rs1: cur.to_reg(), rs2: c1.to_reg(),
        });
        let ms1 = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::And, rd: ms1, rs1: sh1.to_reg(), rs2: m1.to_reg(),
        });
        let s1 = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::Subf, rd: s1, rs1: ms1.to_reg(), rs2: cur.to_reg(),
        });

        // Step 2: (s1 & m2) + ((s1 >> 2) & m2)
        let m2 = self.temp_writable_reg(I64);
        self.emit(&MInst::LoadConst64 { rd: m2, imm: 0x3333_3333_3333_3333 });
        let c2 = self.temp_writable_reg(I64);
        self.emit(&MInst::Li32 { rd: c2, imm: 2 });
        let sh2 = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::Srd, rd: sh2, rs1: s1.to_reg(), rs2: c2.to_reg(),
        });
        let lo2 = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::And, rd: lo2, rs1: s1.to_reg(), rs2: m2.to_reg(),
        });
        let hi2 = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::And, rd: hi2, rs1: sh2.to_reg(), rs2: m2.to_reg(),
        });
        let s2 = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::Add, rd: s2, rs1: lo2.to_reg(), rs2: hi2.to_reg(),
        });

        // Step 3: (s2 + (s2 >> 4)) & 0x0F0F...
        let m3 = self.temp_writable_reg(I64);
        self.emit(&MInst::LoadConst64 { rd: m3, imm: 0x0f0f_0f0f_0f0f_0f0f });
        let c4 = self.temp_writable_reg(I64);
        self.emit(&MInst::Li32 { rd: c4, imm: 4 });
        let sh4 = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::Srd, rd: sh4, rs1: s2.to_reg(), rs2: c4.to_reg(),
        });
        let sum4 = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::Add, rd: sum4, rs1: s2.to_reg(), rs2: sh4.to_reg(),
        });
        let s3 = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::And, rd: s3, rs1: sum4.to_reg(), rs2: m3.to_reg(),
        });

        // Fold: (s3 * 0x0101010101010101) >> 56
        let mf = self.temp_writable_reg(I64);
        self.emit(&MInst::LoadConst64 { rd: mf, imm: 0x0101_0101_0101_0101 });
        let prod = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::Mulld, rd: prod, rs1: s3.to_reg(), rs2: mf.to_reg(),
        });
        let c56 = self.temp_writable_reg(I64);
        self.emit(&MInst::Li32 { rd: c56, imm: 56 });
        let result = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::Srd, rd: result, rs1: prod.to_reg(), rs2: c56.to_reg(),
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
