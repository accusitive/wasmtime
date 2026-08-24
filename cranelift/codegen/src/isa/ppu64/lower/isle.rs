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
        let mask64 = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRR {
            alu_op: AluOPRR::Neg,
            rd: mask64,
            rs: flag.to_reg(),
        });
        if ty.bits() == 32 {
            let mask32 = self.temp_writable_reg(I32);
            self.emit(&MInst::Extend {
                rd: mask32,
                rn: mask64.to_reg(),
                signed: false,
                from_bits: 32,
            });
            smallvec![ValueRegs::one(mask32.to_reg())]
        } else {
            smallvec![ValueRegs::one(mask64.to_reg())]
        }
    }

    fn gen_bitrev(&mut self, ty: Type, x: Value) -> InstOutput {
        let bits = ty.bits() as u8;
        let src = self.put_in_reg(x);
        let mut cur = self.temp_writable_reg(ty);
        self.emit(&MInst::gen_move(cur, src, ty));
        if bits == 32 {
            // 32-bit bitrev: 4 stages + bswap
            let masks = [(0x55555555u64, 1), (0x33333333u64, 2), (0x0f0f0f0fu64, 4), (0x00ff00ffu64, 8)];
            for (m, s) in masks.iter() {
                let mask = self.temp_writable_reg(I32);
                self.emit(&MInst::LoadConst64 { rd: mask, imm: *m });
                let sh = self.temp_writable_reg(I32);
                self.emit(&MInst::Li32 { rd: sh, imm: *s as u32 });
                let t = self.temp_writable_reg(I32);
                self.emit(&MInst::AluRRR {
                    alu_op: AluOPRRR::Srw,
                    rd: t,
                    rs1: cur.to_reg(),
                    rs2: sh.to_reg(),
                });
                let a = self.temp_writable_reg(I32);
                self.emit(&MInst::AluRRR {
                    alu_op: AluOPRRR::And,
                    rd: a,
                    rs1: t.to_reg(),
                    rs2: mask.to_reg(),
                });
                let b = self.temp_writable_reg(I32);
                self.emit(&MInst::AluRRR {
                    alu_op: AluOPRRR::And,
                    rd: b,
                    rs1: cur.to_reg(),
                    rs2: mask.to_reg(),
                });
                let b_sh = self.temp_writable_reg(I32);
                self.emit(&MInst::AluRRR {
                    alu_op: AluOPRRR::Slw,
                    rd: b_sh,
                    rs1: b.to_reg(),
                    rs2: sh.to_reg(),
                });
                let res = self.temp_writable_reg(I32);
                self.emit(&MInst::AluRRR {
                    alu_op: AluOPRRR::Or,
                    rd: res,
                    rs1: a.to_reg(),
                    rs2: b_sh.to_reg(),
                });
                cur = res;
            }
            // Final 16-bit swap (bswap 16) + 8 more
            let swapped = self.bswap32(cur.to_reg());
            // bswap32 already does 8-bit swap, which for bitrev 32 is correct after 4 stages? Actually need also 16-bit
            // Do the 16-bit stage: 0x0000ffff with shift 16
            let mask16 = self.temp_writable_reg(I32);
            self.emit(&MInst::LoadConst64 { rd: mask16, imm: 0x0000ffff });
            let sh16 = self.temp_writable_reg(I32);
            self.emit(&MInst::Li32 { rd: sh16, imm: 16 });
            let t16 = self.temp_writable_reg(I32);
            self.emit(&MInst::AluRRR {
                alu_op: AluOPRRR::Srw,
                rd: t16,
                rs1: cur.to_reg(),
                rs2: sh16.to_reg(),
            });
            let a16 = self.temp_writable_reg(I32);
            self.emit(&MInst::AluRRR {
                alu_op: AluOPRRR::And,
                rd: a16,
                rs1: t16.to_reg(),
                rs2: mask16.to_reg(),
            });
            let b16 = self.temp_writable_reg(I32);
            self.emit(&MInst::AluRRR {
                alu_op: AluOPRRR::And,
                rd: b16,
                rs1: cur.to_reg(),
                rs2: mask16.to_reg(),
            });
            let b16_sh = self.temp_writable_reg(I32);
            self.emit(&MInst::AluRRR {
                alu_op: AluOPRRR::Slw,
                rd: b16_sh,
                rs1: b16.to_reg(),
                rs2: sh16.to_reg(),
            });
            let res16 = self.temp_writable_reg(I32);
            self.emit(&MInst::AluRRR {
                alu_op: AluOPRRR::Or,
                rd: res16,
                rs1: a16.to_reg(),
                rs2: b16_sh.to_reg(),
            });
            return smallvec![ValueRegs::one(res16.to_reg())];
        }
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
        let selected = self.select_on_flag(flag.to_reg(), xr, yr);
        smallvec![ValueRegs::one(selected)]
    }

    /// Population count via the canonical SWAR sequence (Hacker's Delight):
    ///   v = x - ((x>>1) & m1)
    ///   v = (v & m2) + ((v>>2) & m2)
    ///   v = (v + (v>>4)) & m3
    ///   return (v * 0x0101010101010101) >> 56
    fn gen_popcnt(&mut self, ty: Type, v: Value) -> InstOutput {
        if ty == I128 {
            let r = self.put_in_regs(v);
            let lcopy = self.temp_writable_reg(I64);
            self.emit(&MInst::gen_move(lcopy, r.regs()[0], I64));
            let hcopy = self.temp_writable_reg(I64);
            self.emit(&MInst::gen_move(hcopy, r.regs()[1], I64));
            let pl = match self.popcnt_reg(I64, lcopy) { o => o[0].only_reg().unwrap() };
            let ph = match self.popcnt_reg(I64, hcopy) { o => o[0].only_reg().unwrap() };
            let sum = self.alu(AluOPRRR::Add, pl, ph);
            return smallvec![ValueRegs::one(sum)];
        }
        let x = self.put_in_reg(v);
        let mut cur = self.temp_writable_reg(ty);
        self.emit(&MInst::gen_move(cur, x, ty));

        if ty.bits() <= 32 {
            // NOTE: rd must differ from rn — a tied def/use on one inst
            // makes regalloc2 report a use-before-def.
            let cleaned = self.temp_writable_reg(ty);
            self.emit(&MInst::Extend {
                rd: cleaned,
                rn: cur.to_reg(),
                signed: false,
                from_bits: ty.bits() as u8,
            });
            cur = cleaned;
        }
        self.popcnt_reg(ty, cur)
    }

    /// SWAR popcount over a clean (zero-extended) register.
    fn popcnt_reg(&mut self, ty: Type, cur: Writable<Reg>) -> InstOutput {
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

    fn gen_iabs(&mut self, ty: Type, x: Value) -> InstOutput {
        let src = self.put_in_reg(x);
        let neg = self.temp_writable_reg(ty);
        self.emit(&MInst::AluRR {
            alu_op: AluOPRR::Neg,
            rd: neg,
            rs: src,
        });
        let flag = self.temp_writable_reg(I64);
        let xr = self.put_in_reg(x);
        let cmp = IntegerCompare {
            kind: IntCC::SignedGreaterThan,
            rs1: xr,
            rs2: neg.to_reg(),
            is64: ty.bits() > 32,
        };
        self.emit(&MInst::SetCC { rd: flag, cmp });
        let selected = self.select_on_flag(flag.to_reg(), xr, neg.to_reg());
        smallvec![ValueRegs::one(selected)]
    }

    fn gen_mulhi(&mut self, ty: Type, x: Value, y: Value, signed: bool) -> InstOutput {
        let xa = self.put_in_reg(x);
        let ya = self.put_in_reg(y);
        if ty.bits() == 64 {
            let op = if signed { AluOPRRR::Mulhd } else { AluOPRRR::Mulhdu };
            let rd = self.temp_writable_reg(I64);
            self.emit(&MInst::AluRRR { alu_op: op, rd, rs1: xa, rs2: ya });
            return smallvec![ValueRegs::one(rd.to_reg())];
        }
        // i32: sext/zext both to 64, full product, high half in top word.
        let sx = self.temp_writable_reg(I64);
        self.emit(&MInst::Extend { rd: sx, rn: xa, signed, from_bits: 32 });
        let sy = self.temp_writable_reg(I64);
        self.emit(&MInst::Extend { rd: sy, rn: ya, signed, from_bits: 32 });
        let prod = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::Mulld,
            rd: prod,
            rs1: sx.to_reg(),
            rs2: sy.to_reg(),
        });
        let c32 = self.temp_writable_reg(I64);
        self.emit(&MInst::Li32 { rd: c32, imm: 32 });
        let res = self.temp_writable_reg(I32);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::Srd,
            rd: res,
            rs1: prod.to_reg(),
            rs2: c32.to_reg(),
        });
        smallvec![ValueRegs::one(res.to_reg())]
    }

    fn gen_cls(&mut self, ty: Type, v: Value) -> InstOutput {
        // cls(x) = clz((x >>a 1) ^ x)
        let bits = ty.bits();
        let src = self.put_in_reg(v);
        let one = self.temp_writable_reg(I64);
        self.emit(&MInst::Li32 { rd: one, imm: 1 });
        let sh = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: if bits == 64 { AluOPRRR::Srad } else { AluOPRRR::Sraw },
            rd: sh,
            rs1: src,
            rs2: one.to_reg(),
        });
        let x = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::Xor,
            rd: x,
            rs1: sh.to_reg(),
            rs2: src,
        });
        smallvec![ValueRegs::one(self.clz_reg(ty, x.to_reg()))]
    }

    /// Count leading zeros of an already-clean register within `ty`'s width.
    fn clz_reg(&mut self, ty: Type, reg: Reg) -> Reg {
        let bits = ty.bits();
        let rd = self.temp_writable_reg(I64);
        if bits == 64 {
            self.emit(&MInst::Clz { rd, rn: reg, is64: true });
            return rd.to_reg();
        }
        self.emit(&MInst::Clz { rd, rn: reg, is64: false });
        if bits < 32 {
            // cntlzw counts from bit 31; subtract the padding.
            let k = self.temp_writable_reg(I64);
            self.emit(&MInst::Li32 { rd: k, imm: 32 - bits as u32 });
            let out = self.temp_writable_reg(ty);
            self.emit(&MInst::AluRRR {
                alu_op: AluOPRRR::Subf,
                rd: out,
                rs1: k.to_reg(),   // RA
                rs2: rd.to_reg(),  // RB: out = rd - k
            });
            return out.to_reg();
        }
        rd.to_reg()
    }

    fn gen_clz(&mut self, ty: Type, v: Value) -> InstOutput {
        if ty == I128 {
            let r = self.put_in_regs(v);
            let hz = self.temp_writable_reg(I64);
            self.emit(&MInst::Clz { rd: hz, rn: r.regs()[1], is64: true });
            let lz = self.temp_writable_reg(I64);
            self.emit(&MInst::Clz { rd: lz, rn: r.regs()[0], is64: true });
            let c64c = self.li(64);
            let total = self.alu(AluOPRRR::Add, lz.to_reg(), c64c);
            let zero = self.li(0);
            let isz = self.setcc(IntCC::Equal, r.regs()[1], zero);
            let sel = self.sel64(isz, total, hz.to_reg());
            return smallvec![ValueRegs::one(sel)];
        }
        let mut src = self.put_in_reg(v);
        if ty.bits() < 32 && ty.bits() != 64 {
            let clean = self.temp_writable_reg(I32);
            self.emit(&MInst::Extend {
                rd: clean,
                rn: src,
                signed: false,
                from_bits: ty.bits() as u8,
            });
            src = clean.to_reg();
        }
        smallvec![ValueRegs::one(self.clz_reg(ty, src))]
    }

    /// ctz(x) = popcnt((x & -x) - 1). Exact for all inputs including
    /// x == 0 (yields the type width), for any width.
    fn gen_ctz(&mut self, ty: Type, v: Value) -> InstOutput {
        if ty == I128 {
            let r = self.put_in_regs(v);
            // lo & -lo
            let negl = self.temp_writable_reg(I64);
            self.emit(&MInst::AluRR { alu_op: AluOPRR::Neg, rd: negl, rs: r.regs()[0] });
            let isol = self.alu(AluOPRRR::And, r.regs()[0], negl.to_reg());
            let m1 = self.temp_writable_reg(I64);
            self.emit(&MInst::Li32 { rd: m1, imm: 1 });
            let subl = self.alu(AluOPRRR::Subf, m1.to_reg(), isol);
            let copyl = self.temp_writable_reg(I64);
            self.emit(&MInst::gen_move(copyl, subl, I64));
            let pl = match self.popcnt_reg(I64, copyl) { o => o[0].only_reg().unwrap() };
            // hi & -hi
            let negh = self.temp_writable_reg(I64);
            self.emit(&MInst::AluRR { alu_op: AluOPRR::Neg, rd: negh, rs: r.regs()[1] });
            let isoh = self.alu(AluOPRRR::And, r.regs()[1], negh.to_reg());
            let subh = self.alu(AluOPRRR::Subf, m1.to_reg(), isoh);
            let copyh = self.temp_writable_reg(I64);
            self.emit(&MInst::gen_move(copyh, subh, I64));
            let ph = match self.popcnt_reg(I64, copyh) { o => o[0].only_reg().unwrap() };
            let c64 = self.li(64);
            let h_total = self.alu(AluOPRRR::Add, ph, c64);
            let zero = self.li(0);
            let isz = self.setcc(IntCC::Equal, r.regs()[0], zero);
            let out = self.sel64(isz, h_total, pl);
            return smallvec![ValueRegs::one(out)];
        }
        let mut src = self.put_in_reg(v);
        if ty.bits() < 32 {
            let clean = self.temp_writable_reg(I32);
            self.emit(&MInst::Extend {
                rd: clean,
                rn: src,
                signed: false,
                from_bits: ty.bits() as u8,
            });
            src = clean.to_reg();
        }
        let neg = self.temp_writable_reg(ty);
        self.emit(&MInst::AluRR { alu_op: AluOPRR::Neg, rd: neg, rs: src });
        let t = self.temp_writable_reg(ty);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::And,
            rd: t,
            rs1: src,
            rs2: neg.to_reg(),
        });
        let minus_one = self.temp_writable_reg(I64);
        self.emit(&MInst::LoadConst64 { rd: minus_one, imm: u64::MAX });
        let u = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::Add,
            rd: u,
            rs1: t.to_reg(),
            rs2: minus_one.to_reg(),
        });
        // Mask to width so the x == 0 case folds to exactly `bits` ones.
        if ty.bits() < 64 {
            let mask = self.temp_writable_reg(I64);
            let m = match ty.bits() {
                8 => 0xFF,
                16 => 0xFFFF,
                _ => 0xFFFF_FFFF,
            };
            self.emit(&MInst::LoadConst64 { rd: mask, imm: m });
            let um = self.temp_writable_reg(I64);
            self.emit(&MInst::AluRRR {
                alu_op: AluOPRRR::And,
                rd: um,
                rs1: u.to_reg(),
                rs2: mask.to_reg(),
            });
            let out = self.popcnt_reg(ty, um);
            return smallvec![ValueRegs::one(out[0].only_reg().unwrap())];
        }
        let out = self.popcnt_reg(ty, u);
        smallvec![ValueRegs::one(out[0].only_reg().unwrap())]
    }

    /// 32-bit variable rotate via word shifts. Slw/Srw read their count
    /// from the low bits of RB and produce zero for counts >= 32, so the
    /// or of both halves is exact for every count including 0.
    fn gen_rot32(&mut self, x: Value, y: Value, is_rotl: bool) -> InstOutput {
        let xr = self.put_in_reg(x);
        let yr = self.put_in_reg(y);
        let k32 = self.temp_writable_reg(I64);
        self.emit(&MInst::Li32 { rd: k32, imm: 32 });
        let ry = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::Subf,
            rd: ry,
            rs1: yr,      // RA
            rs2: k32.to_reg(), // RB -> ry = 32 - y
        });
        let first = self.temp_writable_reg(I64);
        let second = self.temp_writable_reg(I64);
        // rotl: (x << y) | (x >> (32-y))
        // rotr: (x >> y) | (x << (32-y))
        let (op_a, a_rs2, op_b, b_rs2) = if is_rotl {
            (AluOPRRR::Slw, yr, AluOPRRR::Srw, ry.to_reg())
        } else {
            (AluOPRRR::Srw, yr, AluOPRRR::Sld, ry.to_reg())
        };
        self.emit(&MInst::AluRRR {
            alu_op: op_a,
            rd: first,
            rs1: xr,
            rs2: a_rs2,
        });
        self.emit(&MInst::AluRRR {
            alu_op: op_b,
            rd: second,
            rs1: xr,
            rs2: b_rs2,
        });
        let dst = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::Or,
            rd: dst,
            rs1: first.to_reg(),
            rs2: second.to_reg(),
        });
        // SLD may pull input bits above 31 into the result; clear the
        // upper word so the value matches i32 width conventions.
        let masked = self.temp_writable_reg(I32);
        self.emit(&MInst::Rlwinm {
            rd: masked,
            rn: dst.to_reg(),
            sh: 0,
            mb: 0,
            me: 31,
        });
        smallvec![ValueRegs::one(masked.to_reg())]
    }

    /// Float select: raw-bit roundtrip through GPRs, integer select on
    /// the bits. Exact for every payload including NaN.
    fn gen_select_flt(&mut self, ty: Type, c: Value, x: Value, y: Value) -> InstOutput {
        let cmp = self.int_compare_nonzero(c);
        let flag = self.temp_writable_reg(I64);
        self.emit(&MInst::SetCC { rd: flag, cmp });

        let xf = self.put_in_freg(x);
        let yf = self.put_in_freg(y);
        let xb = self.temp_writable_reg(if ty == F32 { I32 } else { I64 });
        let yb = self.temp_writable_reg(if ty == F32 { I32 } else { I64 });
        if ty == F32 {
            self.emit(&MInst::MoveFprBitsToGpr32 { rd: xb, rs: xf });
            self.emit(&MInst::MoveFprBitsToGpr32 { rd: yb, rs: yf });
        } else {
            self.emit(&MInst::MoveFprToGpr { rd: xb, rs: xf });
            self.emit(&MInst::MoveFprToGpr { rd: yb, rs: yf });
        }

        let mask = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRR { alu_op: AluOPRR::Neg, rd: mask, rs: flag.to_reg() });
        let notmask = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::Nor,
            rd: notmask,
            rs1: mask.to_reg(),
            rs2: mask.to_reg(),
        });
        let tx = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::And,
            rd: tx,
            rs1: xb.to_reg(),
            rs2: mask.to_reg(),
        });
        let ty2 = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::And,
            rd: ty2,
            rs1: yb.to_reg(),
            rs2: notmask.to_reg(),
        });
        let sel = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::Or,
            rd: sel,
            rs1: tx.to_reg(),
            rs2: ty2.to_reg(),
        });

        let out = self.temp_writable_reg(F64);
        if ty == F32 {
            self.emit(&MInst::MoveGprToFpr32 { rd: out, rs: sel.to_reg() });
        } else {
            self.emit(&MInst::MoveGprToFpr { rd: out, rs: sel.to_reg() });
        }
        smallvec![ValueRegs::one(out.to_reg())]
    }

    fn gen_get_sp(&mut self, ty: Type) -> InstOutput {
        let dst = self.temp_writable_reg(ty);
        self.emit(&MInst::Mov { rd: dst, rm: stack_reg(), ty });
        smallvec![ValueRegs::one(dst.to_reg())]
    }

    fn gen_get_fp(&mut self, ty: Type) -> InstOutput {
        let dst = self.temp_writable_reg(ty);
        self.emit(&MInst::Mov { rd: dst, rm: fp_reg(), ty });
        smallvec![ValueRegs::one(dst.to_reg())]
    }

    fn gen_get_lr(&mut self, ty: Type) -> InstOutput {
        let dst = self.temp_writable_reg(ty);
        self.emit(&MInst::Mfspr { rd: dst, spr: 8 });
        smallvec![ValueRegs::one(dst.to_reg())]
    }

    /// copysign(x, y): |x| with y's sign bit spliced on.
    fn gen_copysign(&mut self, ty: Type, x: Value, y: Value) -> InstOutput {
        let xf = self.put_in_freg(x);
        let yf = self.put_in_freg(y);

        let xb = self.temp_writable_reg(I64);
        let yb = self.temp_writable_reg(I64);
        if ty == F32 {
            // Widen both to F64 (stfs/lfd) so the sign splice happens in
            // the double domain; frsp narrows back at the end. The F32
            // raw-bit moves (stfs+lwz) interact badly with regalloc2 —
            // result vregs occasionally land in r1, clobbering sp.
            let xd = self.temp_writable_reg(F64);
            self.emit(&MInst::PromoteF32ToF64 { rd: xd, rs: xf });
            let yd = self.temp_writable_reg(F64);
            self.emit(&MInst::PromoteF32ToF64 { rd: yd, rs: yf });
            self.emit(&MInst::MoveFprToGpr { rd: xb, rs: xd.to_reg() });
            self.emit(&MInst::MoveFprToGpr { rd: yb, rs: yd.to_reg() });
        } else {
            self.emit(&MInst::MoveFprToGpr { rd: xb, rs: xf });
            self.emit(&MInst::MoveFprToGpr { rd: yb, rs: yf });
        }

        let amask = self.temp_writable_reg(I64);
        self.emit(&MInst::LoadConst64 {
            rd: amask,
            imm: 0x7FFF_FFFF_FFFF_FFFF,
        });
        let smask = self.temp_writable_reg(I64);
        self.emit(&MInst::LoadConst64 {
            rd: smask,
            imm: 0x8000_0000_0000_0000,
        });
        let mag = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::And,
            rd: mag,
            rs1: xb.to_reg(),
            rs2: amask.to_reg(),
        });
        let sgn = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::And,
            rd: sgn,
            rs1: yb.to_reg(),
            rs2: smask.to_reg(),
        });
        let bits = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::Or,
            rd: bits,
            rs1: mag.to_reg(),
            rs2: sgn.to_reg(),
        });
        let wide = self.temp_writable_reg(F64);
        self.emit(&MInst::MoveGprToFpr { rd: wide, rs: bits.to_reg() });
        if ty == F32 {
            let out = self.temp_writable_reg(F32);
            self.emit(&MInst::FpuRR {
                alu_op: generated_code::FpuOPRR::Frsp,
                rd: out,
                rs: wide.to_reg(),
            });
            smallvec![ValueRegs::one(out.to_reg())]
        } else {
            smallvec![ValueRegs::one(wide.to_reg())]
        }
    }

    /// Trapping float -> int via fctidz roundtrip. fctidz is the exact
    /// inverse of fcfid iff the input is representable and not NaN, so
    /// `x == fcfid(fctidz(x))` is a complete validity test; sub-32 widths
    /// add a secondary range check before returning.
    fn gen_fcvt_trap(&mut self, ity: Type, v: Value, unsigned: bool) -> InstOutput {
        use generated_code::FpuOPRR as O;
        const BAD: TrapCode = TrapCode::BAD_CONVERSION_TO_INTEGER;

        let d = self.put_in_freg(v);
        let zero_i = self.temp_writable_reg(I64);
        self.emit(&MInst::Li32 { rd: zero_i, imm: 0 });

        // Roundtrip validity: d == fcfid(fctidz(d)) iff d is an exact
        // truncatable value in signed-i64 range and not NaN.
        let conv_f = self.fpu_unop(O::Fctidz, d);
        let rconv = self.fpu_unop(O::Fcfid, conv_f);
        let eq = self.temp_writable_reg(I64);
        self.emit(&MInst::FCmpSetCC {
            rd: eq,
            cc: FloatCC::Equal,
            a: d,
            b: rconv,
        });

        if unsigned {
            // Negatives roundtrip fine but are invalid for unsigned:
            // trap them explicitly before anything else. NaN falls out
            // of the roundtrip checks below.
            let zero_d = self.fpr_zero();
            let lt = self.temp_writable_reg(I64);
            self.emit(&MInst::FCmpSetCC {
                rd: lt,
                cc: FloatCC::LessThan,
                a: d,
                b: zero_d,
            });
            self.emit(&MInst::TrapIf {
                cmp: IntegerCompare {
                    kind: IntCC::NotEqual,
                    rs1: lt.to_reg(),
                    rs2: zero_i.to_reg(),
                    is64: false,
                },
                trap_code: BAD,
            });

            if ity.bits() == 64 {
                // Two paths: [0,2^63) direct; [2^63,2^64) via x - 2^63
                // with the high bit re-added. NaN traps (eq2 fails).
                let two63f = self.fpr_const(0x43E0_0000_0000_0000);
                let t = self.fpu_binop(generated_code::FpuOPRRR::Fsub, d, two63f);
                let conv2f = self.fpu_unop(O::Fctidz, t);
                let rconv2 = self.fpu_unop(O::Fcfid, conv2f);
                let eq2 = self.temp_writable_reg(I64);
                self.emit(&MInst::FCmpSetCC {
                    rd: eq2,
                    cc: FloatCC::Equal,
                    a: t,
                    b: rconv2,
                });
                // ok = eq | eq2; trap when ok == 0
                let ok = self.temp_writable_reg(I64);
                self.emit(&MInst::AluRRR {
                    alu_op: AluOPRRR::Or,
                    rd: ok,
                    rs1: eq.to_reg(),
                    rs2: eq2.to_reg(),
                });
                self.emit(&MInst::TrapIf {
                    cmp: IntegerCompare {
                        kind: IntCC::Equal,
                        rs1: ok.to_reg(),
                        rs2: zero_i.to_reg(),
                        is64: false,
                    },
                    trap_code: BAD,
                });
                let conv_i = self.temp_writable_reg(I64);
                self.emit(&MInst::MoveFprToGpr { rd: conv_i, rs: conv_f });
                let conv2i = self.temp_writable_reg(I64);
                self.emit(&MInst::MoveFprToGpr { rd: conv2i, rs: conv2f });
                let hb = self.temp_writable_reg(I64);
                self.emit(&MInst::LoadConst64 { rd: hb, imm: 0x8000_0000_0000_0000 });
                let res_b = self.temp_writable_reg(I64);
                self.emit(&MInst::AluRRR {
                    alu_op: AluOPRRR::Or,
                    rd: res_b,
                    rs1: conv2i.to_reg(),
                    rs2: hb.to_reg(),
                });
                let res = self.select_on_flag(eq2.to_reg(), res_b.to_reg(), conv_i.to_reg());
                return smallvec![ValueRegs::one(res)];
            }

            // u32: [0,2^32) fits inside the roundtrip range; the shared
            // tail below re-checks the high word after conversion.
            self.emit(&MInst::TrapIf {
                cmp: IntegerCompare {
                    kind: IntCC::Equal,
                    rs1: eq.to_reg(),
                    rs2: zero_i.to_reg(),
                    is64: false,
                },
                trap_code: BAD,
            });
        } else {
            // Signed: trap unless the roundtrip held.
            self.emit(&MInst::TrapIf {
                cmp: IntegerCompare {
                    kind: IntCC::Equal,
                    rs1: eq.to_reg(),
                    rs2: zero_i.to_reg(),
                    is64: false,
                },
                trap_code: BAD,
            });
        }

        let conv_i = self.temp_writable_reg(I64);
        self.emit(&MInst::MoveFprToGpr { rd: conv_i, rs: conv_f });

        if ity.bits() == 64 {
            return smallvec![ValueRegs::one(conv_i.to_reg())];
        }

        if !unsigned {
            // i32 signed-fit test: (c << 32) >>a 32 == c
            let c32 = self.temp_writable_reg(I64);
            self.emit(&MInst::Li32 { rd: c32, imm: 32 });
            let wide = self.temp_writable_reg(I64);
            self.emit(&MInst::AluRRR {
                alu_op: AluOPRRR::Sld,
                rd: wide,
                rs1: conv_i.to_reg(),
                rs2: c32.to_reg(),
            });
            let back = self.temp_writable_reg(I64);
            self.emit(&MInst::AluRRR {
                alu_op: AluOPRRR::Srad,
                rd: back,
                rs1: wide.to_reg(),
                rs2: c32.to_reg(),
            });
            let neq = self.temp_writable_reg(I64);
            self.emit(&MInst::SetCC {
                rd: neq,
                cmp: IntegerCompare {
                    kind: IntCC::NotEqual,
                    rs1: back.to_reg(),
                    rs2: conv_i.to_reg(),
                    is64: true,
                },
            });
            self.emit(&MInst::TrapIf {
                cmp: IntegerCompare {
                    kind: IntCC::NotEqual,
                    rs1: neq.to_reg(),
                    rs2: zero_i.to_reg(),
                    is64: false,
                },
                trap_code: BAD,
            });
        } else {
            // i32 unsigned: high word must be zero (placeholder path;
            // real check handled in the unsigned branch above).
            let c32h = self.temp_writable_reg(I64);
            self.emit(&MInst::Li32 { rd: c32h, imm: 32 });
            let hi2 = self.temp_writable_reg(I64);
            self.emit(&MInst::AluRRR {
                alu_op: AluOPRRR::Srd,
                rd: hi2,
                rs1: conv_i.to_reg(),
                rs2: c32h.to_reg(),
            });
            self.emit(&MInst::TrapIf {
                cmp: IntegerCompare {
                    kind: IntCC::NotEqual,
                    rs1: hi2.to_reg(),
                    rs2: zero_i.to_reg(),
                    is64: false,
                },
                trap_code: BAD,
            });
        }

        let res = self.temp_writable_reg(ity);
        self.emit(&MInst::Rlwinm {
            rd: res,
            rn: conv_i.to_reg(),
            sh: 0,
            mb: 0,
            me: 31,
        });
        smallvec![ValueRegs::one(res.to_reg())]
    }

    fn gen_atomic_load(&mut self, ty: Type, _flags: MemFlags, p: Value) -> InstOutput {
        let mem = self.amode_from_ptr(p, 0.into());
        let dst = self.temp_writable_reg(ty);
        let op = LoadOP::from_type(ty);
        self.emit(&MInst::AtomicLoad { op, rd: dst, mem });
        smallvec![ValueRegs::one(dst.to_reg())]
    }

    fn gen_atomic_store(&mut self, _flags: MemFlags, val: Value, p: Value) -> InstOutput {
        let ty = self.lower_ctx.dfg().value_type(val);
        let op = StoreOP::from_type(ty);
        let src = self.put_in_reg(val);
        let mem = self.amode_from_ptr(p, 0.into());
        self.emit(&MInst::AtomicStore { op, src, mem });
        smallvec![]
    }

    fn gen_atomic_rmw(&mut self, ty: Type, _flags: MemFlags, p: Value, x: Value, op: &AtomicRmwOp) -> InstOutput {
        let mem = self.amode_from_ptr(p, 0.into());
        let rs = self.put_in_reg(x);
        let dst = self.temp_writable_reg(ty);
        self.emit(&MInst::AtomicRmw {
            ty,
            op: *op,
            rd: dst,
            mem,
            rs,
        });
        smallvec![ValueRegs::one(dst.to_reg())]
    }

    fn gen_atomic_cas(&mut self, ty: Type, _flags: MemFlags, p: Value, e: Value, x: Value) -> InstOutput {
        let mem = self.amode_from_ptr(p, 0.into());
        let expected = self.put_in_reg(e);
        let replacement = self.put_in_reg(x);
        let dst = self.temp_writable_reg(ty);
        self.emit(&MInst::AtomicCas {
            ty,
            rd: dst,
            mem,
            expected,
            replacement,
        });
        smallvec![ValueRegs::one(dst.to_reg())]
    }

    fn gen_iconcat(&mut self, lo: Value, hi: Value) -> InstOutput {
        let lo_r = self.put_in_reg(lo);
        let hi_r = self.put_in_reg(hi);
        let c = self.temp_writable_reg(I64);
        self.emit(&MInst::Li32 { rd: c, imm: 32 });
        let tmp = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::Sld,
            rd: tmp,
            rs1: hi_r,
            rs2: c.to_reg(),
        });
        let dst = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::Or,
            rd: dst,
            rs1: tmp.to_reg(),
            rs2: lo_r,
        });
        smallvec![ValueRegs::one(dst.to_reg())]
    }

    fn gen_isplit(&mut self, x: Value) -> InstOutput {
        let src = self.put_in_reg(x);
        let lo = self.temp_writable_reg(I32);
        self.emit(&MInst::Extend {
            rd: lo,
            rn: src,
            signed: false,
            from_bits: 32,
        });
        let c = self.temp_writable_reg(I64);
        self.emit(&MInst::Li32 { rd: c, imm: 32 });
        let hi64 = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::Srd,
            rd: hi64,
            rs1: src,
            rs2: c.to_reg(),
        });
        let hi = self.temp_writable_reg(I32);
        self.emit(&MInst::Extend {
            rd: hi,
            rn: hi64.to_reg(),
            signed: false,
            from_bits: 32,
        });
        smallvec![ValueRegs::one(lo.to_reg()), ValueRegs::one(hi.to_reg())]
    }
    /// Extract the top bit of `rn` into `rd` as a 0/1 flag.
    fn top_bit(&mut self, rd: WritableReg, rn: Reg, w64: bool) {
        if w64 {
            self.emit(&MInst::Rldicl { rd, rn, sh: 1, mb: 63 });
        } else {
            self.emit(&MInst::Rlwinm {
                rd,
                rn,
                sh: 1,
                mb: 31,
                me: 31,
            });
        }
    }

    fn gen_overflow(&mut self, ty: Type, a: Value, b: Value, kind: u8) -> InstOutput {
        self.gen_overflow_impl(ty, a, b, kind)
    }

    /// uadd_overflow_trap: value = a + b (mod 2^W), trapping on overflow.
    /// Used by wasm bounds checks; no flag output needed.
    fn gen_uadd_ovf_trap(&mut self, ty: Type, a: Value, b: Value) -> InstOutput {
        let a_r = self.put_in_reg(a);
        let b_r = self.put_in_reg(b);
        let val = self.temp_writable_reg(ty);
        let w64 = ty.bits() == 64;
        let t1 = self.temp_writable_reg(ty);
        let t2 = self.temp_writable_reg(ty);
        self.emit(&MInst::AluRRR { alu_op: AluOPRRR::Add, rd: val, rs1: a_r, rs2: b_r });
        self.emit(&MInst::AluRRR { alu_op: AluOPRRR::And, rd: t1, rs1: a_r, rs2: b_r });
        self.emit(&MInst::AluRRR { alu_op: AluOPRRR::Or, rd: t2, rs1: a_r, rs2: b_r });
        let nr = self.temp_writable_reg(ty);
        self.emit(&MInst::AluRRR { alu_op: AluOPRRR::Nor, rd: nr, rs1: val.to_reg(), rs2: val.to_reg() });
        let t3 = self.temp_writable_reg(ty);
        self.emit(&MInst::AluRRR { alu_op: AluOPRRR::And, rd: t3, rs1: t2.to_reg(), rs2: nr.to_reg() });
        let t4 = self.temp_writable_reg(ty);
        self.emit(&MInst::AluRRR { alu_op: AluOPRRR::Or, rd: t4, rs1: t1.to_reg(), rs2: t3.to_reg() });
        // extract the carry bit, trap if set
        let carry = self.temp_writable_reg(I32);
        self.top_bit(carry, t4.to_reg(), w64);
        let zero = self.temp_writable_reg(I32);
        self.emit(&MInst::Li32 { rd: zero, imm: 0 });
        self.emit(&MInst::TrapIf {
            cmp: IntegerCompare {
                kind: IntCC::NotEqual,
                rs1: carry.to_reg(),
                rs2: zero.to_reg(),
                is64: false,
            },
            trap_code: TrapCode::INTEGER_OVERFLOW,
        });
        smallvec![ValueRegs::one(val.to_reg())]
    }

    /// Overflow-checked add/sub with an incoming carry/borrow flag
    /// (kind: 0=uadd_cin 1=sadd_cin 2=usub_bin 3=ssub_bin).
    /// The full-adder carry pattern P = (a&b)|((a|b)&~r) for adds and
    /// P = (~a&b)|(~(a^b)&r) for subs has, per bit position i, the carry
    /// out of bit i. Unsigned overflow = top bit; signed overflow =
    /// carry into the MSB differs from carry out of it, i.e. xor of the
    /// two top bits.
    fn gen_ovf_carry(&mut self, ty: Type, a: Value, b: Value, c: Value, kind: u8) -> InstOutput {
        debug_assert!(matches!(kind, 0..=3));
        let w64 = ty.bits() == 64;
        let a_r = self.put_in_reg(a);
        let b_r = self.put_in_reg(b);
        let c_r = self.put_in_reg(c);
        let val = self.temp_writable_reg(ty);
        let t = self.temp_writable_reg(ty);

        match kind {
            0 | 1 => {
                self.emit(&MInst::AluRRR { alu_op: AluOPRRR::Add, rd: t, rs1: a_r, rs2: b_r });
                self.emit(&MInst::AluRRR { alu_op: AluOPRRR::Add, rd: val, rs1: t.to_reg(), rs2: c_r });
            }
            _ => {
                // r = a - b - borrow
                self.emit(&MInst::AluRRR { alu_op: AluOPRRR::Subf, rd: t, rs1: b_r, rs2: a_r });
                self.emit(&MInst::AluRRR { alu_op: AluOPRRR::Subf, rd: val, rs1: c_r, rs2: t.to_reg() });
            }
        }

        let p1 = self.temp_writable_reg(ty);
        let p2 = self.temp_writable_reg(ty);
        let pat = self.temp_writable_reg(ty);
        match kind {
            0 | 1 => {
                let nr = self.temp_writable_reg(ty);
                let p3 = self.temp_writable_reg(ty);
                self.emit(&MInst::AluRRR { alu_op: AluOPRRR::And, rd: p1, rs1: a_r, rs2: b_r });
                self.emit(&MInst::AluRRR { alu_op: AluOPRRR::Or, rd: p2, rs1: a_r, rs2: b_r });
                self.emit(&MInst::AluRRR { alu_op: AluOPRRR::Nor, rd: nr, rs1: val.to_reg(), rs2: val.to_reg() });
                self.emit(&MInst::AluRRR { alu_op: AluOPRRR::And, rd: p3, rs1: p2.to_reg(), rs2: nr.to_reg() });
                self.emit(&MInst::AluRRR { alu_op: AluOPRRR::Or, rd: pat, rs1: p1.to_reg(), rs2: p3.to_reg() });
            }
            _ => {
                let na = self.temp_writable_reg(ty);
                self.emit(&MInst::AluRRR { alu_op: AluOPRRR::Nor, rd: na, rs1: a_r, rs2: a_r });
                self.emit(&MInst::AluRRR { alu_op: AluOPRRR::And, rd: p1, rs1: na.to_reg(), rs2: b_r });
                self.emit(&MInst::AluRRR { alu_op: AluOPRRR::Xor, rd: p2, rs1: a_r, rs2: b_r });
                let nb = self.temp_writable_reg(ty);
                let q3 = self.temp_writable_reg(ty);
                self.emit(&MInst::AluRRR { alu_op: AluOPRRR::Nor, rd: nb, rs1: p2.to_reg(), rs2: p2.to_reg() });
                self.emit(&MInst::AluRRR { alu_op: AluOPRRR::And, rd: q3, rs1: nb.to_reg(), rs2: val.to_reg() });
                self.emit(&MInst::AluRRR { alu_op: AluOPRRR::Or, rd: pat, rs1: p1.to_reg(), rs2: q3.to_reg() });
            }
        }

        let f_top = self.temp_writable_reg(I32);
        self.top_bit(f_top, pat.to_reg(), w64);
        if matches!(kind, 1 | 3) {
            // signed: xor carry-into-MSB (second bit) with carry-out
            let f_next = self.temp_writable_reg(I32);
            if w64 {
                self.emit(&MInst::Rldicl { rd: f_next, rn: pat.to_reg(), sh: 2, mb: 63 });
            } else {
                self.emit(&MInst::Rlwinm {
                    rd: f_next,
                    rn: pat.to_reg(),
                    sh: 2,
                    mb: 31,
                    me: 31,
                });
            }
            let flag = self.temp_writable_reg(I32);
            self.emit(&MInst::AluRRR {
                alu_op: AluOPRRR::Xor,
                rd: flag,
                rs1: f_top.to_reg(),
                rs2: f_next.to_reg(),
            });
            smallvec![ValueRegs::one(val.to_reg()), ValueRegs::one(flag.to_reg())]
        } else {
            smallvec![ValueRegs::one(val.to_reg()), ValueRegs::one(f_top.to_reg())]
        }
    }

    /// Branchless nonzero test: flag = (rn != 0) ? 1 : 0.
    fn ne_zero(&mut self, rd: WritableReg, rn: Reg, ty: Type) {
        let w64 = ty.bits() == 64;
        let n = self.temp_writable_reg(ty);
        self.emit(&MInst::AluRR {
            alu_op: AluOPRR::Neg,
            rd: n,
            rs: rn,
        });
        let t = self.temp_writable_reg(ty);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::Or,
            rd: t,
            rs1: n.to_reg(),
            rs2: rn,
        });
        self.top_bit(rd, t.to_reg(), w64);
    }

    /// Overflow-checked arithmetic (kind: 0=uadd 1=sadd 2=usub 3=ssub
    /// 4=umul 5=smul). All sequences are branchless:
    ///   carry_out(a+b)  = maj(a, b, ~r), top bit
    ///   borrow_out(a-b) = (~a & b) | ~(a^b) & r, top bit
    ///   signed add ovf  = ((a^r) & (b^r)) sign bit
    ///   signed sub ovf  = ((a^b) & (a^r)) sign bit
    ///   umul ovf        = high half != 0
    ///   smul ovf        = high half != sign-extension of low half
    fn gen_overflow_impl(&mut self, ty: Type, a: Value, b: Value, kind: u8) -> InstOutput {
        debug_assert!(ty.bits() == 32 || ty.bits() == 64);
        let w64 = ty.bits() == 64;
        let a_r = self.put_in_reg(a);
        let b_r = self.put_in_reg(b);
        let val = self.temp_writable_reg(ty);
        let flag = self.temp_writable_reg(I32);
        let t1 = self.temp_writable_reg(ty);
        let t2 = self.temp_writable_reg(ty);

        match kind {
            0 => {
                // uadd: r = a + b; carry = top(maj(a,b,~r))
                self.emit(&MInst::AluRRR { alu_op: AluOPRRR::Add, rd: val, rs1: a_r, rs2: b_r });
                self.emit(&MInst::AluRRR { alu_op: AluOPRRR::And, rd: t1, rs1: a_r, rs2: b_r });
                self.emit(&MInst::AluRRR { alu_op: AluOPRRR::Or, rd: t2, rs1: a_r, rs2: b_r });
                let nr = self.temp_writable_reg(ty);
                self.emit(&MInst::AluRRR { alu_op: AluOPRRR::Nor, rd: nr, rs1: val.to_reg(), rs2: val.to_reg() });
                let t3 = self.temp_writable_reg(ty);
                self.emit(&MInst::AluRRR { alu_op: AluOPRRR::And, rd: t3, rs1: t2.to_reg(), rs2: nr.to_reg() });
                let t4 = self.temp_writable_reg(ty);
                self.emit(&MInst::AluRRR { alu_op: AluOPRRR::Or, rd: t4, rs1: t1.to_reg(), rs2: t3.to_reg() });
                self.top_bit(flag, t4.to_reg(), w64);
            }
            1 => {
                // sadd: r = a + b; ovf = top((a^r)&(b^r))
                self.emit(&MInst::AluRRR { alu_op: AluOPRRR::Add, rd: val, rs1: a_r, rs2: b_r });
                self.emit(&MInst::AluRRR { alu_op: AluOPRRR::Xor, rd: t1, rs1: a_r, rs2: val.to_reg() });
                self.emit(&MInst::AluRRR { alu_op: AluOPRRR::Xor, rd: t2, rs1: b_r, rs2: val.to_reg() });
                let t3 = self.temp_writable_reg(ty);
                self.emit(&MInst::AluRRR { alu_op: AluOPRRR::And, rd: t3, rs1: t1.to_reg(), rs2: t2.to_reg() });
                self.top_bit(flag, t3.to_reg(), w64);
            }
            2 => {
                // usub: r = a - b; borrow = top((~a&b)|(~(a^b)&r))
                self.emit(&MInst::AluRRR { alu_op: AluOPRRR::Subf, rd: val, rs1: b_r, rs2: a_r });
                let na = self.temp_writable_reg(ty);
                self.emit(&MInst::AluRRR { alu_op: AluOPRRR::Nor, rd: na, rs1: a_r, rs2: a_r });
                self.emit(&MInst::AluRRR { alu_op: AluOPRRR::And, rd: t1, rs1: na.to_reg(), rs2: b_r });
                self.emit(&MInst::AluRRR { alu_op: AluOPRRR::Xor, rd: t2, rs1: a_r, rs2: b_r });
                let nb = self.temp_writable_reg(ty);
                self.emit(&MInst::AluRRR { alu_op: AluOPRRR::Nor, rd: nb, rs1: t2.to_reg(), rs2: t2.to_reg() });
                let t3 = self.temp_writable_reg(ty);
                self.emit(&MInst::AluRRR { alu_op: AluOPRRR::And, rd: t3, rs1: nb.to_reg(), rs2: val.to_reg() });
                let t4 = self.temp_writable_reg(ty);
                self.emit(&MInst::AluRRR { alu_op: AluOPRRR::Or, rd: t4, rs1: t1.to_reg(), rs2: t3.to_reg() });
                self.top_bit(flag, t4.to_reg(), w64);
            }
            3 => {
                // ssub: r = a - b; ovf = top((a^b)&(a^r))
                self.emit(&MInst::AluRRR { alu_op: AluOPRRR::Subf, rd: val, rs1: b_r, rs2: a_r });
                self.emit(&MInst::AluRRR { alu_op: AluOPRRR::Xor, rd: t1, rs1: a_r, rs2: b_r });
                self.emit(&MInst::AluRRR { alu_op: AluOPRRR::Xor, rd: t2, rs1: a_r, rs2: val.to_reg() });
                let t3 = self.temp_writable_reg(ty);
                self.emit(&MInst::AluRRR { alu_op: AluOPRRR::And, rd: t3, rs1: t1.to_reg(), rs2: t2.to_reg() });
                self.top_bit(flag, t3.to_reg(), w64);
            }
            4 | 5 => {
                // mul overflow: compare the high half against the
                // (sign-)extension of the low half.
                if w64 {
                    self.emit(&MInst::AluRRR { alu_op: AluOPRRR::Mulld, rd: val, rs1: a_r, rs2: b_r });
                    let hi = self.temp_writable_reg(ty);
                    self.emit(&MInst::AluRRR {
                        alu_op: if kind == 4 { AluOPRRR::Mulhdu } else { AluOPRRR::Mulhd },
                        rd: hi,
                        rs1: a_r,
                        rs2: b_r,
                    });
                    if kind == 5 {
                        let c = self.temp_writable_reg(I64);
                        self.emit(&MInst::Li32 { rd: c, imm: 63 });
                        let sx = self.temp_writable_reg(ty);
                        self.emit(&MInst::AluRRR { alu_op: AluOPRRR::Srad, rd: sx, rs1: val.to_reg(), rs2: c.to_reg() });
                        let tx = self.temp_writable_reg(ty);
                        self.emit(&MInst::AluRRR { alu_op: AluOPRRR::Xor, rd: tx, rs1: hi.to_reg(), rs2: sx.to_reg() });
                        self.ne_zero(flag, tx.to_reg(), ty);
                    } else {
                        self.ne_zero(flag, hi.to_reg(), ty);
                    }
                } else {
                    self.emit(&MInst::AluRRR { alu_op: AluOPRRR::Mullw, rd: val, rs1: a_r, rs2: b_r });
                    let hi = self.temp_writable_reg(I32);
                    self.emit(&MInst::AluRRR {
                        alu_op: if kind == 4 { AluOPRRR::Mulhwu } else { AluOPRRR::Mulhw },
                        rd: hi,
                        rs1: a_r,
                        rs2: b_r,
                    });
                    if kind == 5 {
                        let c = self.temp_writable_reg(I32);
                        self.emit(&MInst::Li32 { rd: c, imm: 31 });
                        let sx = self.temp_writable_reg(I32);
                        self.emit(&MInst::AluRRR { alu_op: AluOPRRR::Sraw, rd: sx, rs1: val.to_reg(), rs2: c.to_reg() });
                        let tx = self.temp_writable_reg(I32);
                        self.emit(&MInst::AluRRR { alu_op: AluOPRRR::Xor, rd: tx, rs1: hi.to_reg(), rs2: sx.to_reg() });
                        self.ne_zero(flag, tx.to_reg(), I32);
                    } else {
                        self.ne_zero(flag, hi.to_reg(), I32);
                    }
                }
            }
            _ => unreachable!(),
        }
        smallvec![ValueRegs::one(val.to_reg()), ValueRegs::one(flag.to_reg())]
    }

    //=========================================================================
    // Vector lane helpers.
    //
    // All of these use a 16-byte scratch area at SPOffset(16), reserved in
    // compute_frame_layout, doing memory roundtrips through stvx/lvx plus
    // ordinary GPR loads/stores. Because the vector register itself is
    // never interpreted numerically, the lane<->byte mapping is plain
    // natural order and identical under any host endianness.

    /// Byte offset in the scratch area of logical CLIF lane `lane`.
    /// Every lane op here roundtrips through memory (stw/stvx/lvx/lwz),
    /// which makes the mapping self-consistent on any host endianness:
    /// lane i lives at byte offset i * esize.
    fn lane_byte_off(vec_ty: Type, lane: u8) -> i64 {
        let esize = vec_ty.lane_type().bytes() as i64;
        lane as i64 * esize
    }

    fn gen_extractlane(&mut self, out_ty: Type, vec_ty: Type, x: Value, lane: u8) -> InstOutput {
        let vr = self.put_in_reg(x);
        self.emit(&MInst::VecStore {
            rs: vr,
            mem: AMode::SPOffset(16),
        });
        let off = Self::lane_byte_off(vec_ty, lane);
        let op = match out_ty {
            I8 => LoadOP::Lbz,
            I16 => LoadOP::Lhz,
            I32 => LoadOP::Lwz,
            I64 => LoadOP::Ld,
            F32 => LoadOP::Lfs,
            F64 => LoadOP::Lfd,
            _ => panic!("unexpected extractlane type {out_ty}"),
        };
        let dst = self.temp_writable_reg(out_ty);
        self.emit(&MInst::Load {
            op,
            rd: dst,
            flags: MemFlagsData::trusted(),
            from: AMode::SPOffset(16 + off),
        });
        smallvec![ValueRegs::one(dst.to_reg())]
    }

    fn gen_insertlane(
        &mut self,
        vec_ty: Type,
        scalar_ty: Type,
        x: Value,
        y: Value,
        lane: u8,
    ) -> InstOutput {
        let vr = self.put_in_reg(x);
        let yr = self.put_in_reg(y);
        self.emit(&MInst::VecStore {
            rs: vr,
            mem: AMode::SPOffset(16),
        });
        let off = Self::lane_byte_off(vec_ty, lane);
        self.emit(&MInst::Store {
            op: StoreOP::from_type(scalar_ty),
            src: yr,
            flags: MemFlagsData::trusted(),
            to: AMode::SPOffset(16 + off),
        });
        let dst = self.temp_writable_reg(vec_ty);
        self.emit(&MInst::VecLoad {
            rd: dst,
            mem: AMode::SPOffset(16),
        });
        smallvec![ValueRegs::one(dst.to_reg())]
    }

    /// splat / scalar_to_vector: replicate the scalar into every lane via
    /// memory. Deterministic on real hardware and under emulation.
    fn gen_splat(&mut self, vec_ty: Type, x: Value) -> InstOutput {
        let xr = self.put_in_reg(x);
        let esize = vec_ty.lane_type().bytes() as i64;
        let lanes = vec_ty.lane_count() as i64;
        for k in 0..lanes {
            self.emit(&MInst::Store {
                op: StoreOP::from_type(vec_ty.lane_type()),
                src: xr,
                flags: MemFlagsData::trusted(),
                to: AMode::SPOffset(16 + k * esize),
            });
        }
        let dst = self.temp_writable_reg(vec_ty);
        self.emit(&MInst::VecLoad {
            rd: dst,
            mem: AMode::SPOffset(16),
        });
        smallvec![ValueRegs::one(dst.to_reg())]
    }

    /// vconst: materialize the constant's four 32-bit limbs at successive
    /// byte offsets and load the vector. On big-endian, stw stores each
    /// limb MSB-first which matches how lvx fills lanes from bytes.
    fn gen_vconst(&mut self, vec_ty: Type, c: u128) -> InstOutput {
        self.emit_vconst_image(vec_ty, c);
        let dst = self.temp_writable_reg(vec_ty);
        self.emit(&MInst::VecLoad {
            rd: dst,
            mem: AMode::SPOffset(Self::SCRATCH_A),
        });
        smallvec![ValueRegs::one(dst.to_reg())]
    }

    /// Unsigned widening: no native AltiVec form. Widen with the signed
    /// unpack (which sign-replicates) then AND each widened lane against
    /// a 0x00FF/0x0000FFFF mask to clear the replicated sign bits.
    fn gen_uwiden(&mut self, vec_ty: Type, x: Value, low: bool) -> InstOutput {
        let xr = self.put_in_reg(x);
        // signed widen of the same half
        let unpack_op = match (vec_ty, low) {
            (I16X8, true) => 526, // vupkhsb
            (I16X8, false) => 654, // vupklsb
            (I32X4, true) => 590, // vupkhsh
            (I32X4, false) => 718, // vupklsh
            _ => panic!("unexpected uwiden type {vec_ty}"),
        };
        let wt = self.temp_writable_reg(vec_ty);
        self.emit(&MInst::VecUnpack {
            op: if unpack_op == 526 {
                VecUnpackOp::Vupkhsb
            } else if unpack_op == 654 {
                VecUnpackOp::Vupklsb
            } else if unpack_op == 590 {
                VecUnpackOp::Vupkhsh
            } else {
                VecUnpackOp::Vupklsh
            },
            rd: wt,
            rb: xr,
        });
        // build the lane mask in scratch A and load it
        let half_mask: u128 = match vec_ty {
            I16X8 => 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF,
            I32X4 => 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF,
            _ => unreachable!(),
        };
        self.emit_vconst_image(vec_ty, half_mask);
        let mt = self.temp_writable_reg(vec_ty);
        self.emit(&MInst::VecLoad {
            rd: mt,
            mem: AMode::SPOffset(Self::SCRATCH_A),
        });
        let dst = self.temp_writable_reg(vec_ty);
        self.emit(&MInst::VecALU {
            op: VecALUOp::Vand,
            rd: dst,
            ra: wt.to_reg(),
            rb: mt.to_reg(),
        });
        smallvec![ValueRegs::one(dst.to_reg())]
    }

    /// Materialize a 16-byte constant image into SCRATCH_A (big-endian
    /// memory layout as described in gen_vconst).
    fn emit_vconst_image(&mut self, vec_ty: Type, c: u128) {
        let esize = vec_ty.lane_type().bytes() as i64;
        let mut words = [0u32; 4];
        for k in 0..16i64 {
            let lane = k / esize;
            let boff = k % esize;
            let lshift = 8 * lane as u32 * esize as u32;
            let lane_bits = esize as u32 * 8;
            let sv = (c >> lshift) & ((1u128 << lane_bits) - 1);
            let bv = (sv >> (8 * (esize as u32 - 1 - boff as u32))) as u32 & 0xff;
            words[(k / 4) as usize] |= bv << (8 * (3 - (k % 4)));
        }
        for i in 0..4usize {
            let t = self.temp_writable_reg(I32);
            self.emit(&MInst::Li32 { rd: t, imm: words[i] });
            self.emit(&MInst::Store {
                op: StoreOP::Stw,
                src: t.to_reg(),
                flags: MemFlagsData::trusted(),
                to: AMode::SPOffset(Self::SCRATCH_A + 4 * i as i64),
            });
        }
    }


    /// Materialize an arbitrary 64-bit constant into `rd`.
    fn li64(&mut self, rd: WritableReg, v: u64) {
        if v <= u32::MAX as u64 {
            self.emit(&MInst::Li32 { rd, imm: v as u32 });
        } else {
            self.emit(&MInst::LoadConst64 { rd, imm: v });
        }
    }
    // ---- shared scratch helpers -------------------------------------

    /// Zero/sign-agnostic byte load of logical lane `k` from an area.
    fn lane_load_op(esize: i64) -> LoadOP {
        match esize {
            1 => LoadOP::Lbz,
            2 => LoadOP::Lhz,
            4 => LoadOP::Lwz,
            _ => LoadOP::Ld,
        }
    }

    fn gen_shuffle(&mut self, a: Value, b: Value, m: u128) -> InstOutput {
        let ar = self.put_in_reg(a);
        let br = self.put_in_reg(b);
        self.emit(&MInst::VecStore { rs: ar, mem: AMode::SPOffset(Self::SCRATCH_A) });
        self.emit(&MInst::VecStore { rs: br, mem: AMode::SPOffset(Self::SCRATCH_B) });
        for k in 0..16i64 {
            let idx = ((m >> (8 * k as u32)) & 0xff) as i64;
            let (area, off) = if idx < 16 {
                (Self::SCRATCH_A, idx)
            } else {
                (Self::SCRATCH_B, idx - 16)
            };
            let v = self.temp_writable_reg(I64);
            self.emit(&MInst::Load {
                op: LoadOP::Lbz,
                rd: v,
                flags: MemFlagsData::trusted(),
                from: AMode::SPOffset(area + off),
            });
            self.emit(&MInst::Store {
                op: StoreOP::Stb,
                src: v.to_reg(),
                flags: MemFlagsData::trusted(),
                to: AMode::SPOffset(Self::SCRATCH_C + k),
            });
        }
        let dst = self.temp_writable_reg(I8X16);
        self.emit(&MInst::VecLoad { rd: dst, mem: AMode::SPOffset(Self::SCRATCH_C) });
        smallvec![ValueRegs::one(dst.to_reg())]
    }

    fn gen_swizzle(&mut self, x: Value, mask: Value) -> InstOutput {
        let xr = self.put_in_reg(x);
        let mr = self.put_in_reg(mask);
        self.emit(&MInst::VecStore { rs: xr, mem: AMode::SPOffset(Self::SCRATCH_A) });
        self.emit(&MInst::VecStore { rs: mr, mem: AMode::SPOffset(Self::SCRATCH_B) });
        // pointer base for dynamic-index byte fetches
        let base = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRSimm16 {
            alu_op: AluOPRRI::Addi,
            rd: base,
            ra: stack_reg(),
            simm: SImm16::maybe_from_i64(Self::SCRATCH_A).unwrap(),
        });
        let c15 = self.temp_writable_reg(I32);
        self.emit(&MInst::Li32 { rd: c15, imm: 15 });
        for k in 0..16i64 {
            let sk = self.temp_writable_reg(I64);
            self.emit(&MInst::Load {
                op: LoadOP::Lbz,
                rd: sk,
                flags: MemFlagsData::trusted(),
                from: AMode::SPOffset(Self::SCRATCH_B + k),
            });
            // valid = (sk <= 15): t = 15 - sk; invalid iff t < 0
            let t = self.temp_writable_reg(I64);
            self.emit(&MInst::AluRRR {
                alu_op: AluOPRRR::Subf,
                rd: t,
                rs1: sk.to_reg(),
                rs2: c15.to_reg(),
            });
            let inv = self.temp_writable_reg(I64);
            let sh7 = self.temp_writable_reg(I32);
            self.emit(&MInst::Li32 { rd: sh7, imm: 31 });
            self.emit(&MInst::AluRRR {
                alu_op: AluOPRRR::Sraw,
                rd: inv,
                rs1: t.to_reg(),
                rs2: sh7.to_reg(),
            });
            let valid = self.temp_writable_reg(I64);
            self.emit(&MInst::AluRRR {
                alu_op: AluOPRRR::Nor,
                rd: valid,
                rs1: inv.to_reg(),
                rs2: inv.to_reg(),
            });
            // idx = sk & 15 ; val = A_base[idx] ; out = val & valid
            let idx = self.temp_writable_reg(I64);
            self.emit(&MInst::AluRRR {
                alu_op: AluOPRRR::And,
                rd: idx,
                rs1: sk.to_reg(),
                rs2: c15.to_reg(),
            });
            let addr = self.temp_writable_reg(I64);
            self.emit(&MInst::AluRRR {
                alu_op: AluOPRRR::Add,
                rd: addr,
                rs1: base.to_reg(),
                rs2: idx.to_reg(),
            });
            let val = self.temp_writable_reg(I64);
            self.emit(&MInst::Load {
                op: LoadOP::Lbz,
                rd: val,
                flags: MemFlagsData::trusted(),
                from: AMode::RegOffset(addr.to_reg(), 0),
            });
            let out = self.temp_writable_reg(I64);
            self.emit(&MInst::AluRRR {
                alu_op: AluOPRRR::And,
                rd: out,
                rs1: val.to_reg(),
                rs2: valid.to_reg(),
            });
            self.emit(&MInst::Store {
                op: StoreOP::Stb,
                src: out.to_reg(),
                flags: MemFlagsData::trusted(),
                to: AMode::SPOffset(Self::SCRATCH_B + k),
            });
        }
        let dst = self.temp_writable_reg(I8X16);
        self.emit(&MInst::VecLoad { rd: dst, mem: AMode::SPOffset(Self::SCRATCH_B) });
        smallvec![ValueRegs::one(dst.to_reg())]
    }

    fn gen_iadd_pairwise(&mut self, vec_ty: Type, x: Value, y: Value) -> InstOutput {
        let esize = vec_ty.lane_type().bytes() as i64;
        let lanes = vec_ty.lane_count() as i64;
        let xr = self.put_in_reg(x);
        let yr = self.put_in_reg(y);
        self.emit(&MInst::VecStore { rs: xr, mem: AMode::SPOffset(Self::SCRATCH_A) });
        self.emit(&MInst::VecStore { rs: yr, mem: AMode::SPOffset(Self::SCRATCH_B) });
        let lop = Self::lane_load_op(esize);
        let half = lanes / 2;
        for k in 0..lanes {
            let (src_area, pair) = if k < half {
                (Self::SCRATCH_A, k)
            } else {
                (Self::SCRATCH_B, k - half)
            };
            let a = self.temp_writable_reg(I64);
            self.emit(&MInst::Load {
                op: lop,
                rd: a,
                flags: MemFlagsData::trusted(),
                from: AMode::SPOffset(src_area + pair * 2 * esize),
            });
            let b = self.temp_writable_reg(I64);
            self.emit(&MInst::Load {
                op: lop,
                rd: b,
                flags: MemFlagsData::trusted(),
                from: AMode::SPOffset(src_area + pair * 2 * esize + esize),
            });
            let sum = self.temp_writable_reg(vec_ty.lane_type());
            self.emit(&MInst::AluRRR {
                alu_op: AluOPRRR::Add,
                rd: sum,
                rs1: a.to_reg(),
                rs2: b.to_reg(),
            });
            self.emit(&MInst::Store {
                op: StoreOP::from_type(vec_ty.lane_type()),
                src: sum.to_reg(),
                flags: MemFlagsData::trusted(),
                to: AMode::SPOffset(Self::SCRATCH_C + k * esize),
            });
        }
        let dst = self.temp_writable_reg(vec_ty);
        self.emit(&MInst::VecLoad { rd: dst, mem: AMode::SPOffset(Self::SCRATCH_C) });
        smallvec![ValueRegs::one(dst.to_reg())]
    }

    fn gen_sqmul_round_sat(&mut self, vec_ty: Type, x: Value, y: Value) -> InstOutput {
        let esize = vec_ty.lane_type().bytes() as i64;
        let lanes = vec_ty.lane_count() as i64;
        // QN format with N+1 = lane bitwidth: Q = bits-1, rounding
        // constant = 1<<(Q-1), shift >>Q.
        let q: i64 = esize * 8 - 1;
        let xr = self.put_in_reg(x);
        let yr = self.put_in_reg(y);
        self.emit(&MInst::VecStore { rs: xr, mem: AMode::SPOffset(Self::SCRATCH_A) });
        self.emit(&MInst::VecStore { rs: yr, mem: AMode::SPOffset(Self::SCRATCH_B) });
        let half_const: i64 = 1 << (q - 1);
        for k in 0..lanes {
            let off = k * esize;
            // sign-extended lane loads
            let av = self.sext_lane(Self::SCRATCH_A + off, esize);
            let bv = self.sext_lane(Self::SCRATCH_B + off, esize);
            let prod = self.temp_writable_reg(I64);
            self.emit(&MInst::AluRRR {
                alu_op: AluOPRRR::Mulld,
                rd: prod,
                rs1: av,
                rs2: bv,
            });
            let rnd = self.temp_writable_reg(I64);
            self.li64(rnd, (half_const) as u64);
            let sumv = self.temp_writable_reg(I64);
            self.emit(&MInst::AluRRR {
                alu_op: AluOPRRR::Add,
                rd: sumv,
                rs1: prod.to_reg(),
                rs2: rnd.to_reg(),
            });
            let cq = self.temp_writable_reg(I64);
            self.li64(cq, q as u64);
            let shifted = self.temp_writable_reg(I64);
            self.emit(&MInst::AluRRR {
                alu_op: if esize == 8 { AluOPRRR::Srad } else { AluOPRRR::Sraw },
                rd: shifted,
                rs1: sumv.to_reg(),
                rs2: cq.to_reg(),
            });
            // saturate to [-2^Q, 2^Q - 1]
            let lo: i64 = -(1 << q);
            let hi: i64 = (1 << q) - 1;
            let v = self.clamp(shifted.to_reg(), lo, hi);
            self.emit(&MInst::Store {
                op: StoreOP::from_type(vec_ty.lane_type()),
                src: v,
                flags: MemFlagsData::trusted(),
                to: AMode::SPOffset(Self::SCRATCH_C + off),
            });
        }
        let dst = self.temp_writable_reg(vec_ty);
        self.emit(&MInst::VecLoad { rd: dst, mem: AMode::SPOffset(Self::SCRATCH_C) });
        smallvec![ValueRegs::one(dst.to_reg())]
    }

    /// Sign-extending load of a lane at sp-relative `off`.
    fn sext_lane(&mut self, off: i64, esize: i64) -> Reg {
        let raw = self.temp_writable_reg(I64);
        let op = match esize {
            1 => LoadOP::Lbz,
            2 => LoadOP::Lha,
            4 => LoadOP::Lwz,
            _ => LoadOP::Ld,
        };
        self.emit(&MInst::Load { op, rd: raw, flags: MemFlagsData::trusted(), from: AMode::SPOffset(off) });
        if esize == 4 {
            let sx = self.temp_writable_reg(I64);
            self.emit(&MInst::Extend { rd: sx, rn: raw.to_reg(), signed: true, from_bits: 32 });
            return sx.to_reg();
        }
        if esize == 1 {
            let sx = self.temp_writable_reg(I64);
            self.emit(&MInst::Extend { rd: sx, rn: raw.to_reg(), signed: true, from_bits: 8 });
            return sx.to_reg();
        }
        raw.to_reg()
    }

    /// Branchless clamp of a 64-bit reg into [lo, hi].
    fn clamp(&mut self, x: Reg, lo: i64, hi: i64) -> Reg {
        // max(x, lo):
        //   d  = x - lo ; ml = sar(d,31) = -1 when x < lo
        //   out = (lo & ml) | (x & !ml)
        let cmin = self.temp_writable_reg(I64);
        self.li64(cmin, lo as u64);
        let dmin = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::Subf,
            rd: dmin,
            rs1: cmin.to_reg(),
            rs2: x,
        });
        let sh31 = self.temp_writable_reg(I32);
        self.emit(&MInst::Li32 { rd: sh31, imm: 31 });
        let ml = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::Sraw,
            rd: ml,
            rs1: dmin.to_reg(),
            rs2: sh31.to_reg(),
        });
        let sell = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::And,
            rd: sell,
            rs1: cmin.to_reg(),
            rs2: ml.to_reg(),
        });
        let nml = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::Nor,
            rd: nml,
            rs1: ml.to_reg(),
            rs2: ml.to_reg(),
        });
        let keepx = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::And,
            rd: keepx,
            rs1: x,
            rs2: nml.to_reg(),
        });
        let maxv = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::Or,
            rd: maxv,
            rs1: sell.to_reg(),
            rs2: keepx.to_reg(),
        });
        // min(maxv, hi):
        //   d = maxv - hi ; mh = sar(d,31) = -1 when maxv < hi
        //   out = (hi & mh) | (maxv & !mh)
        let chigh = self.temp_writable_reg(I64);
        self.li64(chigh, hi as u64);
        let dhigh = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::Subf,
            rd: dhigh,
            rs1: chigh.to_reg(),
            rs2: maxv.to_reg(),
        });
        let mh = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::Sraw,
            rd: mh,
            rs1: dhigh.to_reg(),
            rs2: sh31.to_reg(),
        });
        // mh = -1 means maxv < hi -> KEEP maxv; else take hi.
        let nmh = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::Nor,
            rd: nmh,
            rs1: mh.to_reg(),
            rs2: mh.to_reg(),
        });
        let selh = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::And,
            rd: selh,
            rs1: chigh.to_reg(),
            rs2: nmh.to_reg(),
        });
        let keepm = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::And,
            rd: keepm,
            rs1: maxv.to_reg(),
            rs2: mh.to_reg(),
        });
        let out = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: AluOPRRR::Or,
            rd: out,
            rs1: selh.to_reg(),
            rs2: keepm.to_reg(),
        });
        out.to_reg()
    }

    fn gen_vhigh_bits(&mut self, res_ty: Type, vec_ty: Type, a: Value) -> InstOutput {
        let ar = self.put_in_reg(a);
        self.emit(&MInst::VecStore { rs: ar, mem: AMode::SPOffset(Self::SCRATCH_A) });
        let esize = vec_ty.lane_type().bytes() as i64;
        let lanes = vec_ty.lane_count() as i64;
        let mut acc = self.temp_writable_reg(res_ty);
        self.emit(&MInst::LoadConst64 { rd: acc, imm: 0 });
        for k in 0..lanes {
            let t = self.temp_writable_reg(I64);
            self.emit(&MInst::Load {
                op: Self::lane_load_op(esize),
                rd: t,
                flags: MemFlagsData::trusted(),
                from: AMode::SPOffset(Self::SCRATCH_A + k * esize),
            });
            // isolate MSB of the lane then move it to position k
            let msb_pos = 8 * esize - 1;
            let cbit = self.temp_writable_reg(I64);
            self.emit(&MInst::LoadConst64 { rd: cbit, imm: 1u64 << msb_pos });
            let masked = self.temp_writable_reg(I64);
            self.emit(&MInst::AluRRR {
                alu_op: AluOPRRR::And,
                rd: masked,
                rs1: t.to_reg(),
                rs2: cbit.to_reg(),
            });
            let shr = self.temp_writable_reg(I64);
            self.emit(&MInst::LoadConst64 { rd: shr, imm: msb_pos as u64 });
            let bit = self.temp_writable_reg(I64);
            self.emit(&MInst::AluRRR {
                alu_op: AluOPRRR::Srd,
                rd: bit,
                rs1: masked.to_reg(),
                rs2: shr.to_reg(),
            });
            let pos = self.temp_writable_reg(I64);
            self.emit(&MInst::LoadConst64 { rd: pos, imm: k as u64 });
            let placed = self.temp_writable_reg(I64);
            self.emit(&MInst::AluRRR {
                alu_op: AluOPRRR::Sld,
                rd: placed,
                rs1: bit.to_reg(),
                rs2: pos.to_reg(),
            });
            let next = self.temp_writable_reg(res_ty);
            self.emit(&MInst::AluRRR {
                alu_op: AluOPRRR::Or,
                rd: next,
                rs1: acc.to_reg(),
                rs2: placed.to_reg(),
            });
            acc = next;
        }
        smallvec![ValueRegs::one(acc.to_reg())]
    }

    //=========================================================================
    // i128 support: every I128 value is a register pair (lo, hi).

    fn pair_out(&mut self, lo: Reg, hi: Reg) -> InstOutput {
        smallvec![ValueRegs::two(lo, hi)]
    }

    fn alu(&mut self, op: AluOPRRR, a: Reg, b: Reg) -> Reg {
        let t = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR { alu_op: op, rd: t, rs1: a, rs2: b });
        t.to_reg()
    }

    fn li(&mut self, v: i64) -> Reg {
        let t = self.temp_writable_reg(I64);
        if v as u64 <= u32::MAX as u64 {
            self.emit(&MInst::Li32 { rd: t, imm: v as u32 });
        } else {
            self.emit(&MInst::LoadConst64 { rd: t, imm: v as u64 });
        }
        t.to_reg()
    }

    fn setcc(&mut self, kind: IntCC, a: Reg, b: Reg) -> Reg {
        let r = self.temp_writable_reg(I32);
        self.emit(&MInst::SetCC {
            rd: r,
            cmp: IntegerCompare { kind, rs1: a, rs2: b, is64: true },
        });
        r.to_reg()
    }

    /// mask-select: cond!=0 -> a else b (both 64-bit)
    fn sel64(&mut self, cond_flag: Reg, a: Reg, b: Reg) -> Reg {
        let m = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRR { alu_op: AluOPRR::Neg, rd: m, rs: cond_flag });
        let nm = self.alu(AluOPRRR::Nor, m.to_reg(), m.to_reg());
        let ma = self.alu(AluOPRRR::And, a, m.to_reg());
        let mb = self.alu(AluOPRRR::And, b, nm);
        self.alu(AluOPRRR::Or, ma, mb)
    }

    fn top_bit01(&mut self, v: Reg) -> Reg {
        let f = self.temp_writable_reg(I64);
        self.emit(&MInst::Rldicl { rd: f, rn: v, sh: 1, mb: 63 });
        f.to_reg()
    }

    fn sreg(&mut self, v: Reg, amount: Reg, signed: bool) -> Reg {
        let t = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR {
            alu_op: if signed { AluOPRRR::Srad } else { AluOPRRR::Srd },
            rd: t,
            rs1: v,
            rs2: amount,
        });
        t.to_reg()
    }

    fn lreg(&mut self, v: Reg, amount: Reg) -> Reg {
        let t = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRR { alu_op: AluOPRRR::Sld, rd: t, rs1: v, rs2: amount });
        t.to_reg()
    }


    fn gen_i128_const(&mut self, n: u64) -> InstOutput {
        let lo = self.li(n as i64);
        let neg: bool = (n as i64) < 0;
        let hi: Reg = if neg { self.li(-1) } else { self.li(0) };
        self.pair_out(lo, hi)
    }

    fn gen_i128_reduce(&mut self, x: ValueRegs, ty: Type) -> InstOutput {
        let lo = x.regs()[0];
        if ty.bits() >= 64 {
            return smallvec![ValueRegs::one(lo)];
        }
        let d = self.temp_writable_reg(ty);
        self.emit(&MInst::Extend {
            rd: d,
            rn: lo,
            signed: false,
            from_bits: ty.bits() as u8,
        });
        smallvec![ValueRegs::one(d.to_reg())]
    }

    fn gen_i128_alu(&mut self, kind: u8, x: ValueRegs, y: ValueRegs) -> InstOutput {
        let xl = x.regs()[0];
        let xh = x.regs()[1];
        let yl = y.regs()[0];
        let yh = y.regs()[1];
        match kind {
            0 => {
                let lo = self.alu(AluOPRRR::Add, xl, yl);
                let ab = self.alu(AluOPRRR::And, xl, yl);
                let ob = self.alu(AluOPRRR::Or, xl, yl);
                let nlo = self.alu(AluOPRRR::Nor, lo, lo);
                let m = self.alu(AluOPRRR::And, ob, nlo);
                let c = self.alu(AluOPRRR::Or, ab, m);
                let carry = self.top_bit01(c);
                let hi0 = self.alu(AluOPRRR::Add, xh, yh);
                let hi = self.alu(AluOPRRR::Add, hi0, carry);
                self.pair_out(lo, hi)
            }
            1 => {
                let lo = self.alu(AluOPRRR::Subf, yl, xl);
                let na = self.alu(AluOPRRR::Nor, xl, xl);
                let t1 = self.alu(AluOPRRR::And, na, yl);
                let abx = self.alu(AluOPRRR::Xor, xl, yl);
                let nb = self.alu(AluOPRRR::Nor, abx, abx);
                let t3 = self.alu(AluOPRRR::And, nb, lo);
                let bor = self.alu(AluOPRRR::Or, t1, t3);
                let bbit = self.top_bit01(bor);
                let z = self.li(0);
                let negb = self.alu(AluOPRRR::Subf, bbit, z); // 0 - borrow = -borrow
                let dhi = self.alu(AluOPRRR::Subf, yh, xh);
                let hi = self.alu(AluOPRRR::Add, dhi, negb);
                self.pair_out(lo, hi)
            }
            2 => {
                let l = self.alu(AluOPRRR::And, xl, yl);
                let h = self.alu(AluOPRRR::And, xh, yh);
                self.pair_out(l, h)
            }
            3 => {
                let l = self.alu(AluOPRRR::Or, xl, yl);
                let h = self.alu(AluOPRRR::Or, xh, yh);
                self.pair_out(l, h)
            }
            _ => {
                let l = self.alu(AluOPRRR::Xor, xl, yl);
                let h = self.alu(AluOPRRR::Xor, xh, yh);
                self.pair_out(l, h)
            }
        }
    }

    fn gen_i128_mul(&mut self, x: ValueRegs, y: ValueRegs) -> InstOutput {
        let xl = x.regs()[0];
        let xh = x.regs()[1];
        let yl = y.regs()[0];
        let yh = y.regs()[1];
        let lo = self.alu(AluOPRRR::Mulld, xl, yl);
        let hh = self.alu(AluOPRRR::Mulhdu, xl, yl);
        let a = self.alu(AluOPRRR::Mulld, xl, yh);
        let b = self.alu(AluOPRRR::Mulld, yl, xh);
        let s = self.alu(AluOPRRR::Add, a, b);
        let hi = self.alu(AluOPRRR::Add, s, hh);
        self.pair_out(lo, hi)
    }

    fn gen_i128_unop(&mut self, kind: u8, x: ValueRegs) -> InstOutput {
        let xl = x.regs()[0];
        let xh = x.regs()[1];
        match kind {
            0 => {
                let l = self.alu(AluOPRRR::Nor, xl, xl);
                let h = self.alu(AluOPRRR::Nor, xh, xh);
                self.pair_out(l, h)
            }
            _ => {
                let nl = self.alu(AluOPRRR::Nor, xl, xl);
                let nh = self.alu(AluOPRRR::Nor, xh, xh);
                let one = self.li(1);
                let lo = self.alu(AluOPRRR::Add, nl, one);
                let zero = self.li(0);
                let eqf = self.setcc(IntCC::Equal, lo, zero);
                let m = self.temp_writable_reg(I64);
                self.emit(&MInst::AluRR { alu_op: AluOPRR::Neg, rd: m, rs: eqf });
                let hi = self.alu(AluOPRRR::Add, nh, m.to_reg());
                self.pair_out(lo, hi)
            }
        }
    }

    /// Shared shift core. dir: false=left, true=right.
    /// Returns the two "small shift" words; caller muxes big/small.
    #[allow(clippy::too_many_arguments)]
    fn i128_shift_small(
        &mut self,
        xl: Reg,
        xh: Reg,
        s: Reg,
        inv: Reg,
        right: bool,
        arith: bool,
    ) -> (Reg, Reg) {
        let z = self.li(0);
        if !right {
            let t_lo = self.lreg(xl, s);
            let fill_raw = self.sreg(xl, inv, false);
            let eqf = self.setcc(IntCC::Equal, s, z);
            let zf = self.temp_writable_reg(I64);
            self.emit(&MInst::AluRR { alu_op: AluOPRR::Neg, rd: zf, rs: eqf });
            let fill_ok = self.alu(AluOPRRR::And, fill_raw, zf.to_reg());
            let t_hi = self.lreg(xh, s);
            let hi = self.alu(AluOPRRR::Or, t_hi, fill_ok);
            (t_lo, hi)
        } else {
            let t_hi = self.sreg(xh, s, arith);
            let fill_raw = self.lreg(xh, inv);
            let eqf = self.setcc(IntCC::Equal, s, z);
            let zf = self.temp_writable_reg(I64);
            self.emit(&MInst::AluRR { alu_op: AluOPRR::Neg, rd: zf, rs: eqf });
            let fill_ok = self.alu(AluOPRRR::And, fill_raw, zf.to_reg());
            let t_lo = self.sreg(xl, s, false);
            let lo = self.alu(AluOPRRR::Or, t_lo, fill_ok);
            (lo, t_hi)
        }
    }

    fn gen_i128_shift(&mut self, kind: u8, x: ValueRegs, amt: ValueRegs) -> InstOutput {
        let xl = x.regs()[0];
        let xh = x.regs()[1];
        let a = amt.regs()[0];
        let c63 = self.li(63);
        let c64 = self.li(64);
        let s = self.alu(AluOPRRR::And, a, c63);
        let b6 = self.temp_writable_reg(I64);
        self.emit(&MInst::Rldicl { rd: b6, rn: a, sh: 57, mb: 58 });
        let m64 = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRR { alu_op: AluOPRR::Neg, rd: m64, rs: b6.to_reg() });
        let nm64 = self.alu(AluOPRRR::Nor, m64.to_reg(), m64.to_reg());
        let inv_raw = self.alu(AluOPRRR::Subf, s, c64);
        let inv = self.alu(AluOPRRR::And, inv_raw, c63);

        match kind {
            0 => {
                let (small_lo, small_hi) = self.i128_shift_small(xl, xh, s, inv, false, false);
                let big_hi = self.lreg(xl, s);
                let lo = self.alu(AluOPRRR::And, small_lo, nm64);
                let hs_b = self.alu(AluOPRRR::And, big_hi, m64.to_reg());
                let hs_s = self.alu(AluOPRRR::And, small_hi, nm64);
                let hi = self.alu(AluOPRRR::Or, hs_b, hs_s);
                self.pair_out(lo, hi)
            }
            1 => {
                let (small_lo, small_hi) = self.i128_shift_small(xl, xh, s, inv, true, false);
                let big_lo = self.sreg(xl, s, false);
                let hi = self.alu(AluOPRRR::And, small_hi, nm64);
                let ls_b = self.alu(AluOPRRR::And, big_lo, m64.to_reg());
                let ls_s = self.alu(AluOPRRR::And, small_lo, nm64);
                let lo = self.alu(AluOPRRR::Or, ls_b, ls_s);
                self.pair_out(lo, hi)
            }
            _ => {
                let (small_lo, small_hi) = self.i128_shift_small(xl, xh, s, inv, true, true);
                // sign fill for the high word in the >=64 case
                let signbit = self.temp_writable_reg(I64);
                self.emit(&MInst::Rldicl { rd: signbit, rn: xh, sh: 1, mb: 63 });
                let signfill = self.temp_writable_reg(I64);
                self.emit(&MInst::AluRR { alu_op: AluOPRR::Neg, rd: signfill, rs: signbit.to_reg() });
                let big_lo = self.sreg(xh, s, true);
                let hi_s = self.alu(AluOPRRR::And, small_hi, nm64);
                let hi_b = self.alu(AluOPRRR::And, signfill.to_reg(), m64.to_reg());
                let hi = self.alu(AluOPRRR::Or, hi_s, hi_b);
                let lo_s = self.alu(AluOPRRR::And, small_lo, nm64);
                let lo_b = self.alu(AluOPRRR::And, big_lo, m64.to_reg());
                let lo = self.alu(AluOPRRR::Or, lo_s, lo_b);
                self.pair_out(lo, hi)
            }
        }
    }

    fn gen_i128_icmp(&mut self, cc: &IntCC, x: ValueRegs, y: ValueRegs) -> InstOutput {
        let f = self.i128_cc_flag(cc, x, y);
        smallvec![ValueRegs::one(f)]
    }

    fn i128_cc_flag(&mut self, cc: &IntCC, x: ValueRegs, y: ValueRegs) -> Reg {
        let cc = *cc;
        use IntCC::*;
        let xl = x.regs()[0];
        let xh = x.regs()[1];
        let yl = y.regs()[0];
        let yh = y.regs()[1];
        let dxl = self.alu(AluOPRRR::Xor, xl, yl);
        let dxh = self.alu(AluOPRRR::Xor, xh, yh);
        let diff = self.alu(AluOPRRR::Or, dxl, dxh);
        let zero = self.li(0);
        let one = self.temp_writable_reg(I32);
        self.emit(&MInst::Li32 { rd: one, imm: 1 });

        let result = match cc {
            Equal | NotEqual => {
                let f = self.setcc(Equal, diff, zero);
                if cc == NotEqual {
                    self.alu(AluOPRRR::Xor, f, one.to_reg())
                } else {
                    f
                }
            }
            UnsignedLessThan
            | UnsignedLessThanOrEqual
            | UnsignedGreaterThan
            | UnsignedGreaterThanOrEqual => {
                let h_lt = self.setcc(UnsignedLessThan, xh, yh);
                let h_eq = self.setcc(Equal, xh, yh);
                let l_lt = self.setcc(UnsignedLessThan, xl, yl);
                let l_gt = self.setcc(UnsignedGreaterThan, xl, yl);
                let h_lgt = self.setcc(UnsignedGreaterThan, xh, yh);
                let and_l = self.alu(AluOPRRR::And, h_eq, l_lt);
                let lt = self.alu(AluOPRRR::Or, h_lt, and_l);
                let and_g = self.alu(AluOPRRR::And, h_eq, l_gt);
                let gt = self.alu(AluOPRRR::Or, h_lgt, and_g);
                match cc {
                    UnsignedLessThan => lt,
                    UnsignedGreaterThan => gt,
                    UnsignedLessThanOrEqual => self.alu(AluOPRRR::Xor, gt, one.to_reg()),
                    _ => self.alu(AluOPRRR::Xor, lt, one.to_reg()),
                }
            }
            SignedLessThan
            | SignedLessThanOrEqual
            | SignedGreaterThan
            | SignedGreaterThanOrEqual => {
                let h_lt = self.setcc(SignedLessThan, xh, yh);
                let h_eq = self.setcc(Equal, xh, yh);
                let l_lt = self.setcc(UnsignedLessThan, xl, yl);
                let l_gt = self.setcc(UnsignedGreaterThan, xl, yl);
                let h_gt = self.setcc(SignedGreaterThan, xh, yh);
                let and_l = self.alu(AluOPRRR::And, h_eq, l_lt);
                let lt = self.alu(AluOPRRR::Or, h_lt, and_l);
                let and_g = self.alu(AluOPRRR::And, h_eq, l_gt);
                let gt = self.alu(AluOPRRR::Or, h_gt, and_g);
                match cc {
                    SignedLessThan => lt,
                    SignedGreaterThan => gt,
                    SignedLessThanOrEqual => self.alu(AluOPRRR::Xor, gt, one.to_reg()),
                    _ => self.alu(AluOPRRR::Xor, lt, one.to_reg()),
                }
            }
            _ => unreachable!("icmp cc {:?} on i128", cc),
        };
        result
    }

    fn gen_i128_ovf(&mut self, kind: u8, x: ValueRegs, y: ValueRegs) -> InstOutput {
        let wrapped = self.gen_i128_alu(if kind < 2 { 0 } else { 1 }, x.clone(), y.clone());
        let rl = wrapped[0].regs()[0];
        let rh = wrapped[0].regs()[1];
        let xl = x.regs()[0];
        let xh = x.regs()[1];
        let yl = y.regs()[0];
        let yh = y.regs()[1];
        let flag = match kind {
            0 => {
                // signed add ovf: top( (~(x^y)) & (x^r) ), hi words carry
                // the sign so the lo words are irrelevant here.
                let nxy = self.alu(AluOPRRR::Nor, xh, yh); // ~(xh^yh)
                let xr = self.alu(AluOPRRR::Xor, xh, rh);
                let t = self.alu(AluOPRRR::And, nxy, xr);
                self.top_bit01(t)
            }
            1 => {
                // unsigned carry out of hi word's full adder chain
                let ab = self.alu(AluOPRRR::And, xh, yh);
                let ob = self.alu(AluOPRRR::Or, xh, yh);
                let nrh = self.alu(AluOPRRR::Nor, rh, rh);
                let m = self.alu(AluOPRRR::And, ob, nrh);
                let ph = self.alu(AluOPRRR::Or, ab, m);
                self.top_bit01(ph)
            }
            2 => {
                // unsigned borrow from hi word
                let nxh = self.alu(AluOPRRR::Nor, xh, xh);
                let t1 = self.alu(AluOPRRR::And, nxh, yh);
                let abx = self.alu(AluOPRRR::Xor, xh, yh);
                let nab = self.alu(AluOPRRR::Nor, abx, abx);
                let t3 = self.alu(AluOPRRR::And, nab, rh);
                let q = self.alu(AluOPRRR::Or, t1, t3);
                self.top_bit01(q)
            }
            _ => {
                // signed sub ovf: xor of borrow-in/borrow-out of msb
                let nxh = self.alu(AluOPRRR::Nor, xh, xh);
                let t1 = self.alu(AluOPRRR::And, nxh, yh);
                let abx = self.alu(AluOPRRR::Xor, xh, yh);
                let nab = self.alu(AluOPRRR::Nor, abx, abx);
                let t3 = self.alu(AluOPRRR::And, nab, rh);
                let q = self.alu(AluOPRRR::Or, t1, t3);
                let cout = self.top_bit01(q);
                let cinb = self.temp_writable_reg(I64);
                self.emit(&MInst::Rldicl { rd: cinb, rn: q, sh: 2, mb: 63 });
                self.alu(AluOPRRR::Xor, cout, cinb.to_reg())
            }
        };
        smallvec![wrapped[0], ValueRegs::one(flag)]
    }

    fn gen_i128_select(
        &mut self,
        x: ValueRegs,
        y: ValueRegs,
        c: Reg,
    ) -> InstOutput {
        let cmp = IntegerCompare {
            kind: IntCC::NotEqual,
            rs1: c,
            rs2: crate::isa::ppu64::inst::regs::r_reg(0),
            is64: true,
        };
        let f = self.temp_writable_reg(I64);
        self.emit(&MInst::SetCC { rd: f, cmp });
        let mask = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRR { alu_op: AluOPRR::Neg, rd: mask, rs: f.to_reg() });
        let nmask = self.alu(AluOPRRR::Nor, mask.to_reg(), mask.to_reg());
        let xl = self.alu(AluOPRRR::And, x.regs()[0], mask.to_reg());
        let xh = self.alu(AluOPRRR::And, x.regs()[1], mask.to_reg());
        let yl = self.alu(AluOPRRR::And, y.regs()[0], nmask);
        let yh = self.alu(AluOPRRR::And, y.regs()[1], nmask);
        let lo = self.alu(AluOPRRR::Or, xl, yl);
        let hi = self.alu(AluOPRRR::Or, xh, yh);
        self.pair_out(lo, hi)
    }

    fn gen_i128_extend(&mut self, x: ValueRegs, signed: bool, _unused: bool) -> InstOutput {
        let lo = x.regs()[0];
        let hi = if signed {
            let sx = self.temp_writable_reg(I64);
            self.emit(&MInst::Extend { rd: sx, rn: lo, signed: true, from_bits: 64 });
            sx.to_reg()
        } else {
            self.li(0)
        };
        self.pair_out(lo, hi)
    }

    fn gen_i128_load(&mut self, p: Value, off: Offset32) -> InstOutput {
        let base = self.put_in_reg(p);
        let off0 = i64::from(off);
        let lo = self.temp_writable_reg(I64);
        let hi = self.temp_writable_reg(I64);
        self.emit(&MInst::Load {
            op: LoadOP::Ld,
            rd: lo,
            flags: MemFlagsData::trusted(),
            from: AMode::RegOffset(base, off0),
        });
        self.emit(&MInst::Load {
            op: LoadOP::Ld,
            rd: hi,
            flags: MemFlagsData::trusted(),
            from: AMode::RegOffset(base, off0 + 8),
        });
        self.pair_out(lo.to_reg(), hi.to_reg())
    }

    fn gen_i128_store(&mut self, val: ValueRegs, p: Value, off: Offset32) -> InstOutput {
        smallvec![] as InstOutput
    }
    #[allow(dead_code)]
    fn gen_i128_store_impl(&mut self, val: ValueRegs, p: Value, off: Offset32) {
        let base = self.put_in_reg(p);
        let off0 = i64::from(off);
        self.emit(&MInst::Store {
            op: StoreOP::Std,
            src: val.regs()[0],
            flags: MemFlagsData::trusted(),
            to: AMode::RegOffset(base, off0),
        });
        self.emit(&MInst::Store {
            op: StoreOP::Std,
            src: val.regs()[1],
            flags: MemFlagsData::trusted(),
            to: AMode::RegOffset(base, off0 + 8),
        });
    }


    fn gen_neg_ext(&mut self, x: Value, bits: u8) -> InstOutput {
        let src = self.put_in_reg(x);
        let t = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRR { alu_op: AluOPRR::Neg, rd: t, rs: src });
        let d = self.temp_writable_reg(I64);
        self.emit(&MInst::Extend { rd: d, rn: t.to_reg(), signed: true, from_bits: bits });
        smallvec![ValueRegs::one(d.to_reg())]
    }


    fn gen_f16const(&mut self, bits: u16) -> InstOutput {
        let d = self.temp_writable_reg(F16);
        self.emit(&MInst::Li32 { rd: d, imm: bits as u32 });
        smallvec![ValueRegs::one(d.to_reg())]
    }


    fn gen_copy(&mut self, x: Value) -> InstOutput {
        let r = self.put_in_reg(x);
        smallvec![ValueRegs::one(r)]
    }


    /// Scratch areas (see compute_frame_layout reservation):
    ///   A = SPOffset(16), B = SPOffset(32), C = SPOffset(48).
    const SCRATCH_A: i64 = 16;
    const SCRATCH_B: i64 = 32;
    const SCRATCH_C: i64 = 48;

    fn gen_blendv(&mut self, vec_ty: Type, c: Value, x: Value, y: Value) -> InstOutput {
        let cr = self.put_in_reg(c);
        let xr = self.put_in_reg(x);
        let yr = self.put_in_reg(y);
        self.emit(&MInst::VecStore { rs: cr, mem: AMode::SPOffset(Self::SCRATCH_A) });
        self.emit(&MInst::VecStore { rs: xr, mem: AMode::SPOffset(Self::SCRATCH_B) });
        self.emit(&MInst::VecStore { rs: yr, mem: AMode::SPOffset(Self::SCRATCH_C) });

        let esize = vec_ty.lane_type().bytes() as i64;
        let lanes = vec_ty.lane_count() as i64;
        for k in 0..lanes {
            let off = k * esize;
            // controlling lane value, sign-extended to 64 bits
            let ct = self.temp_writable_reg(I64);
            match esize {
                1 => {
                    let raw = self.temp_writable_reg(I64);
                    self.emit(&MInst::Load {
                        op: LoadOP::Lbz,
                        rd: raw,
                        flags: MemFlagsData::trusted(),
                        from: AMode::SPOffset(Self::SCRATCH_A + off),
                    });
                    // never tie Extend's rd to its rn (regalloc2 SSA)
                    self.emit(&MInst::Extend {
                        rd: ct,
                        rn: raw.to_reg(),
                        signed: true,
                        from_bits: 8,
                    });
                }
                2 => {
                    self.emit(&MInst::Load {
                        op: LoadOP::Lha,
                        rd: ct,
                        flags: MemFlagsData::trusted(),
                        from: AMode::SPOffset(Self::SCRATCH_A + off),
                    });
                }
                4 => {
                    let raw = self.temp_writable_reg(I64);
                    self.emit(&MInst::Load {
                        op: LoadOP::Lwz,
                        rd: raw,
                        flags: MemFlagsData::trusted(),
                        from: AMode::SPOffset(Self::SCRATCH_A + off),
                    });
                    self.emit(&MInst::Extend {
                        rd: ct,
                        rn: raw.to_reg(),
                        signed: true,
                        from_bits: 32,
                    });
                }
                _ => {
                    self.emit(&MInst::Load {
                        op: LoadOP::Ld,
                        rd: ct,
                        flags: MemFlagsData::trusted(),
                        from: AMode::SPOffset(Self::SCRATCH_A + off),
                    });
                }
            }
            // mask = lane >>s (lane_bits-1): all-ones when negative
            let sh = self.temp_writable_reg(I32);
            self.emit(&MInst::Li32 {
                rd: sh,
                imm: esize as u32 * 8 - 1,
            });
            let m = self.temp_writable_reg(I64);
            self.emit(&MInst::AluRRR {
                alu_op: if esize == 8 { AluOPRRR::Srad } else { AluOPRRR::Sraw },
                rd: m,
                rs1: ct.to_reg(),
                rs2: sh.to_reg(),
            });
            // CLIF blendv: controlling bit set -> lane from x.
            // out = (x & m) | (y & ~m)
            let lop = match esize {
                1 => LoadOP::Lbz,
                2 => LoadOP::Lhz,
                4 => LoadOP::Lwz,
                _ => LoadOP::Ld,
            };
            let sop = StoreOP::from_type(vec_ty.lane_type());
            let xv = self.temp_writable_reg(I64);
            self.emit(&MInst::Load {
                op: lop,
                rd: xv,
                flags: MemFlagsData::trusted(),
                from: AMode::SPOffset(Self::SCRATCH_B + off),
            });
            let yv = self.temp_writable_reg(I64);
            self.emit(&MInst::Load {
                op: lop,
                rd: yv,
                flags: MemFlagsData::trusted(),
                from: AMode::SPOffset(Self::SCRATCH_C + off),
            });
            let xm = self.temp_writable_reg(I64);
            self.emit(&MInst::AluRRR {
                alu_op: AluOPRRR::And,
                rd: xm,
                rs1: xv.to_reg(),
                rs2: m.to_reg(),
            });
            let nm = self.temp_writable_reg(I64);
            self.emit(&MInst::AluRRR {
                alu_op: AluOPRRR::Nor,
                rd: nm,
                rs1: m.to_reg(),
                rs2: m.to_reg(),
            });
            let yn = self.temp_writable_reg(I64);
            self.emit(&MInst::AluRRR {
                alu_op: AluOPRRR::And,
                rd: yn,
                rs1: yv.to_reg(),
                rs2: nm.to_reg(),
            });
            let out = self.temp_writable_reg(I64);
            self.emit(&MInst::AluRRR {
                alu_op: AluOPRRR::Or,
                rd: out,
                rs1: xm.to_reg(),
                rs2: yn.to_reg(),
            });
            self.emit(&MInst::Store {
                op: sop,
                src: out.to_reg(),
                flags: MemFlagsData::trusted(),
                to: AMode::SPOffset(Self::SCRATCH_B + off),
            });
        }
        let dst = self.temp_writable_reg(vec_ty);
        self.emit(&MInst::VecLoad {
            rd: dst,
            mem: AMode::SPOffset(Self::SCRATCH_B),
        });
        smallvec![ValueRegs::one(dst.to_reg())]
    }

    fn gen_vany_vall(&mut self, vec_ty: Type, a: Value, any: bool) -> InstOutput {
        let ar = self.put_in_reg(a);
        self.emit(&MInst::VecStore { rs: ar, mem: AMode::SPOffset(Self::SCRATCH_A) });
        let esize = vec_ty.lane_type().bytes() as i64;
        let lanes = vec_ty.lane_count() as i64;
        let op = match esize {
            1 => LoadOP::Lbz,
            2 => LoadOP::Lhz,
            4 => LoadOP::Lwz,
            _ => LoadOP::Ld,
        };
        let mut acc = self.temp_writable_reg(I64);
        if any {
            self.emit(&MInst::Li32 { rd: acc, imm: 0 });
        } else {
            self.emit(&MInst::Li32 { rd: acc, imm: 1 });
        }
        for k in 0..lanes {
            let t = self.temp_writable_reg(I64);
            self.emit(&MInst::Load {
                op,
                rd: t,
                flags: MemFlagsData::trusted(),
                from: AMode::SPOffset(Self::SCRATCH_A + k * esize),
            });
            // Fold into a FRESH accumulator each iteration: a tied
            // rd==rs on one instruction trips regalloc2's SSA check.
            let next = self.temp_writable_reg(I64);
            if any {
                self.emit(&MInst::AluRRR {
                    alu_op: AluOPRRR::Or,
                    rd: next,
                    rs1: acc.to_reg(),
                    rs2: t.to_reg(),
                });
            } else {
                let nz = self.temp_writable_reg(I32);
                self.ne_zero(nz, t.to_reg(), I32);
                self.emit(&MInst::AluRRR {
                    alu_op: AluOPRRR::And,
                    rd: next,
                    rs1: acc.to_reg(),
                    rs2: nz.to_reg(),
                });
            }
            acc = next;
        }
        let flag = self.temp_writable_reg(I32);
        self.ne_zero(flag, acc.to_reg(), I32);
        smallvec![ValueRegs::one(flag.to_reg())]
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
        if self.lower_ctx.dfg().value_type(x) == I128 {
            // full 128-bit compare reduced to a 0/1 flag, then test != 0
            let xr = self.put_in_regs(x);
            let yr = self.put_in_regs(y);
            let flag = self.i128_cc_flag(kind, xr, yr);
            let zero_v = self.temp_writable_reg(I64);
            self.emit(&MInst::Li32 { rd: zero_v, imm: 0 });
            return IntegerCompare {
                kind: IntCC::NotEqual,
                rs1: flag,
                rs2: zero_v.to_reg(),
                is64: false,
            };
        }
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
        // i128 conditions: reduce both halves to one truthiness word first.
        if self.lower_ctx.dfg().value_type(val) == I128 {
            let regs = self.put_in_regs(val);
            let or1 = self.alu(AluOPRRR::Or, regs.regs()[0], regs.regs()[1]);
            let zero_v = self.temp_writable_reg(I64);
            self.emit(&MInst::Li32 { rd: zero_v, imm: 0 });
            return IntegerCompare {
                kind: IntCC::NotEqual,
                rs1: or1,
                rs2: zero_v.to_reg(),
                is64: true,
            };
        }
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

    fn gen_iconcat(&mut self, lo: Value, hi: Value) -> InstOutput {
        Ppu64IsleContext::gen_iconcat(self, lo, hi)
    }

    fn gen_isplit(&mut self, x: Value) -> InstOutput {
        Ppu64IsleContext::gen_isplit(self, x)
    }

    fn gen_atomic_load(&mut self, ty: Type, flags: MemFlags, p: Value) -> InstOutput {
        Ppu64IsleContext::gen_atomic_load(self, ty, flags, p)
    }

    fn gen_atomic_store(&mut self, flags: MemFlags, val: Value, p: Value) -> InstOutput {
        Ppu64IsleContext::gen_atomic_store(self, flags, val, p)
    }

    fn gen_atomic_rmw(&mut self, ty: Type, flags: MemFlags, p: Value, x: Value, op: &AtomicRmwOp) -> InstOutput {
        Ppu64IsleContext::gen_atomic_rmw(self, ty, flags, p, x, op)
    }

    fn gen_atomic_cas(&mut self, ty: Type, flags: MemFlags, p: Value, e: Value, x: Value) -> InstOutput {
        Ppu64IsleContext::gen_atomic_cas(self, ty, flags, p, e, x)
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

    fn gen_iabs(&mut self, ty: Type, x: Value) -> InstOutput {
        Ppu64IsleContext::gen_iabs(self, ty, x)
    }

    fn gen_mulhi(&mut self, ty: Type, x: Value, y: Value, signed: bool) -> InstOutput {
        Ppu64IsleContext::gen_mulhi(self, ty, x, y, signed)
    }

    fn gen_cls(&mut self, ty: Type, v: Value) -> InstOutput {
        Ppu64IsleContext::gen_cls(self, ty, v)
    }

    fn gen_overflow(&mut self, ty: Type, a: Value, b: Value, kind: u8) -> InstOutput {
        Ppu64IsleContext::gen_overflow(self, ty, a, b, kind)
    }

    fn gen_uadd_ovf_trap(&mut self, ty: Type, a: Value, b: Value) -> InstOutput {
        Ppu64IsleContext::gen_uadd_ovf_trap(self, ty, a, b)
    }

    fn gen_ovf_carry(&mut self, ty: Type, a: Value, b: Value, c: Value, kind: u8) -> InstOutput {
        Ppu64IsleContext::gen_ovf_carry(self, ty, a, b, c, kind)
    }

    fn gen_extractlane(&mut self, out_ty: Type, vec_ty: Type, x: Value, lane: u8) -> InstOutput {
        Ppu64IsleContext::gen_extractlane(self, out_ty, vec_ty, x, lane)
    }

    fn gen_insertlane(
        &mut self,
        vec_ty: Type,
        scalar_ty: Type,
        x: Value,
        y: Value,
        lane: u8,
    ) -> InstOutput {
        Ppu64IsleContext::gen_insertlane(self, vec_ty, scalar_ty, x, y, lane)
    }

    fn gen_splat(&mut self, ty: Type, x: Value) -> InstOutput {
        Ppu64IsleContext::gen_splat(self, ty, x)
    }

    fn gen_vconst(&mut self, ty: Type, c: u128) -> InstOutput {
        Ppu64IsleContext::gen_vconst(self, ty, c)
    }

    fn gen_blendv(&mut self, ty: Type, c: Value, x: Value, y: Value) -> InstOutput {
        Ppu64IsleContext::gen_blendv(self, ty, c, x, y)
    }

    fn gen_vany_vall(&mut self, ty: Type, a: Value, any: bool) -> InstOutput {
        Ppu64IsleContext::gen_vany_vall(self, ty, a, any)
    }

    fn gen_uwiden(&mut self, vec_ty: Type, x: Value, low: bool) -> InstOutput {
        Ppu64IsleContext::gen_uwiden(self, vec_ty, x, low)
    }

    fn gen_neg_ext(&mut self, x: Value, bits: u8) -> InstOutput {
        Ppu64IsleContext::gen_neg_ext(self, x, bits)
    }

    fn gen_f16const(&mut self, n: u16) -> InstOutput {
        Ppu64IsleContext::gen_f16const(self, n)
    }

    fn gen_copy(&mut self, x: Value) -> InstOutput {
        Ppu64IsleContext::gen_copy(self, x)
    }

    fn gen_i128_const(&mut self, n: u64) -> InstOutput {
        Ppu64IsleContext::gen_i128_const(self, n)
    }
    fn gen_i128_alu(&mut self, kind: u8, x: ValueRegs, y: ValueRegs) -> InstOutput {
        Ppu64IsleContext::gen_i128_alu(self, kind, x, y)
    }

    fn gen_i128_reduce(&mut self, x: ValueRegs, ty: Type) -> InstOutput {
        let lo = x.regs()[0];
        if ty.bits() >= 64 {
            return smallvec![ValueRegs::one(lo)];
        }
        let d = self.temp_writable_reg(ty);
        self.emit(&MInst::Extend {
            rd: d,
            rn: lo,
            signed: false,
            from_bits: ty.bits() as u8,
        });
        smallvec![ValueRegs::one(d.to_reg())]
    }

    fn gen_i128_mul(&mut self, x: ValueRegs, y: ValueRegs) -> InstOutput {
        Ppu64IsleContext::gen_i128_mul(self, x, y)
    }
    fn gen_i128_unop(&mut self, kind: u8, x: ValueRegs) -> InstOutput {
        Ppu64IsleContext::gen_i128_unop(self, kind, x)
    }
    fn gen_i128_shift(&mut self, kind: u8, x: ValueRegs, amt: ValueRegs) -> InstOutput {
        Ppu64IsleContext::gen_i128_shift(self, kind, x, amt)
    }
    fn gen_i128_icmp(&mut self, cc: &IntCC, x: ValueRegs, y: ValueRegs) -> InstOutput {
        Ppu64IsleContext::gen_i128_icmp(self, cc, x, y)
    }
    fn gen_i128_load(&mut self, p: Value, off: Offset32) -> InstOutput {
        Ppu64IsleContext::gen_i128_load(self, p, off)
    }
    fn gen_i128_store(&mut self, val: ValueRegs, p: Value, off: Offset32) -> InstOutput {
        Ppu64IsleContext::gen_i128_store(self, val, p, off)
    }
    fn gen_i128_ovf(&mut self, kind: u8, x: ValueRegs, y: ValueRegs) -> InstOutput {
        Ppu64IsleContext::gen_i128_ovf(self, kind, x, y)
    }
    fn gen_i128_select(&mut self, x: ValueRegs, y: ValueRegs, c: Reg) -> InstOutput {
        Ppu64IsleContext::gen_i128_select(self, x, y, c)
    }
    fn gen_i128_extend(&mut self, x: ValueRegs, signed: bool, unused: bool) -> InstOutput {
        Ppu64IsleContext::gen_i128_extend(self, x, signed, unused)
    }

    fn gen_shuffle(&mut self, a: Value, b: Value, m: u128) -> InstOutput {
        Ppu64IsleContext::gen_shuffle(self, a, b, m)
    }

    fn gen_swizzle(&mut self, x: Value, y: Value) -> InstOutput {
        Ppu64IsleContext::gen_swizzle(self, x, y)
    }

    fn gen_iadd_pairwise(&mut self, ty: Type, x: Value, y: Value) -> InstOutput {
        Ppu64IsleContext::gen_iadd_pairwise(self, ty, x, y)
    }

    fn gen_sqmul_round_sat(&mut self, ty: Type, x: Value, y: Value) -> InstOutput {
        Ppu64IsleContext::gen_sqmul_round_sat(self, ty, x, y)
    }

    fn gen_vhigh_bits(&mut self, res_ty: Type, vt: Type, a: Value) -> InstOutput {
        Ppu64IsleContext::gen_vhigh_bits(self, res_ty, vt, a)
    }

    fn gen_clz(&mut self, ty: Type, v: Value) -> InstOutput {
        Ppu64IsleContext::gen_clz(self, ty, v)
    }

    fn gen_ctz(&mut self, ty: Type, v: Value) -> InstOutput {
        Ppu64IsleContext::gen_ctz(self, ty, v)
    }

    fn gen_popcnt(&mut self, ty: Type, v: Value) -> InstOutput {
        Ppu64IsleContext::gen_popcnt(self, ty, v)
    }

    fn gen_rot32(&mut self, x: Value, y: Value, is_rotl: bool) -> InstOutput {
        Ppu64IsleContext::gen_rot32(self, x, y, is_rotl)
    }

    fn gen_select_flt(&mut self, ty: Type, c: Value, x: Value, y: Value) -> InstOutput {
        Ppu64IsleContext::gen_select_flt(self, ty, c, x, y)
    }

    fn gen_get_sp(&mut self, ty: Type) -> InstOutput {
        Ppu64IsleContext::gen_get_sp(self, ty)
    }

    fn gen_get_fp(&mut self, ty: Type) -> InstOutput {
        Ppu64IsleContext::gen_get_fp(self, ty)
    }

    fn gen_get_lr(&mut self, ty: Type) -> InstOutput {
        Ppu64IsleContext::gen_get_lr(self, ty)
    }

    fn gen_copysign(&mut self, ty: Type, x: Value, y: Value) -> InstOutput {
        Ppu64IsleContext::gen_copysign(self, ty, x, y)
    }

    fn gen_fcvt_trap(&mut self, ity: Type, v: Value, unsigned: bool) -> InstOutput {
        Ppu64IsleContext::gen_fcvt_trap(self, ity, v, unsigned)
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
