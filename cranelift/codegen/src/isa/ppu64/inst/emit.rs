//! ppu64 ISA definitions: instruction emission.

use crate::binemit::Reloc;
use crate::ir::condcodes::{FloatCC, IntCC};
use crate::ir::types::{F32, F64, I8, I16, I32, I64};
use crate::ir::{self};
use crate::isa::ppu64::abi::Ppu64MachineDeps;
use crate::machinst::{
    ArgPair, CallInfo, FrameLayout, MachBuffer, MachInst, MachInstEmit, MachInstEmitState,
    MachLabel, RetPair,
};
use crate::settings;

use super::*;
use alloc::string::{String, ToString};

use crate::ir::AtomicRmwOp;

/// Constant-pool-free constant load sequences may be up to six
/// instructions long (`LoadConst64`).
pub(crate) const MAX_CONST_LOAD_LEN: i32 = 24;

pub struct EmitInfo {
    shared_flag: settings::Flags,
    isa_flags: super::super::settings::Flags,
}

impl EmitInfo {
    pub(crate) fn new(
        shared_flag: settings::Flags,
        isa_flags: super::super::settings::Flags,
    ) -> Self {
        Self {
            shared_flag,
            isa_flags,
        }
    }
}

//============================================================================
// Encoding helpers.
//
// All words are emitted big-endian; `put_word` handles the conversion since
// `MachBuffer::put4` writes little-endian bytes.

fn put_word(sink: &mut MachBuffer<Inst>, word: u32) {
    sink.put4(word.swap_bytes());
}

fn reg_num(r: Reg) -> u32 {
    u32::from(r.to_real_reg().unwrap().hw_enc() & 31)
}

fn writable_reg_num(r: Writable<Reg>) -> u32 {
    reg_num(r.to_reg())
}

fn vreg_num(r: VReg) -> u32 {
    u32::from(r.to_reg().to_real_reg().unwrap().hw_enc() & 31)
}

fn writable_vreg_num(r: WritableVReg) -> u32 {
    vreg_num(r.to_reg())
}

/// D-form: `op RT, RA, SI`.
fn enc_d(op: u32, rt: u32, ra: u32, si: u16) -> u32 {
    debug_assert!(op < (1 << 6));
    (op << 26) | (rt << 21) | (ra << 16) | u32::from(si)
}

/// DS-form (`ld`, `std`, `lwa`, `stdu`): the displacement is stored divided
/// by four.
fn enc_ds(op: u32, rt: u32, ra: u32, disp_bytes: i64, xo: u32) -> u32 {
    debug_assert!(disp_bytes % 4 == 0);
    let ds = disp_bytes >> 2;
    debug_assert!(ds >= -8192 && ds <= 8191);
    (op << 26) | (rt << 21) | (ra << 16) | (((ds as u32) & 0x3fff) << 2) | xo
}

/// X-form, primary opcode 31.
fn enc_x(rt: u32, ra: u32, rb: u32, xo: u32, rc: bool) -> u32 {
    (31 << 26) | (rt << 21) | (ra << 16) | (rb << 11) | (xo << 1) | rc as u32
}

/// X-form, primary opcode 63 (floating point).
fn enc_f(frt: u32, fra: u32, frb: u32, xo: u32, rc: bool) -> u32 {
    (63 << 26) | (frt << 21) | (fra << 16) | (frb << 11) | (xo << 1) | rc as u32
}

/// B-form conditional branch (opcode 16): `bc BO, BI, BD`.
fn enc_bc(bo: u32, bi: u32, bd_words: i32) -> u32 {
    (16 << 26) | (bo << 21) | (bi << 16) | (((bd_words as u32) & 0x3fff) << 2)
}

/// X-form branch-conditional (opcode 19): includes `blr` and `bctr[l]`.
fn enc_bx(bo: u32, bi: u32, bb: u32, xo: u32, lk: bool) -> u32 {
    (19 << 26) | (bo << 21) | (bi << 16) | (bb << 11) | (xo << 1) | lk as u32
}

/// XL-form CR-logical op (opcode 19): `crand`/`cror`/etc.
fn enc_cr(xo: u32, bt: u32, ba: u32, bb: u32) -> u32 {
    (19 << 26) | (bt << 21) | (ba << 16) | (bb << 11) | (xo << 1)
}

/// A-form floating-point op (opcode 63): `op rT, rA, rB, rC`.
fn enc_a(frt: u32, fra: u32, frb: u32, frc: u32, xo5: u32, rc: bool) -> u32 {
    (63 << 26) | (frt << 21) | (fra << 16) | (frb << 11) | (frc << 6) | (xo5 << 1) | rc as u32
}

/// I-form unconditional branch (opcode 18): `b`/`bl`. The LI field here is
/// filled in later by a relocation or label-use patch.
const ENC_B_BASE: u32 = 18 << 26;

/// M-form rotate-and-mask (opcode 21), used for zero-extends: `rlwinm`.
fn enc_m(rt: u32, ra: u32, sh: u32, mb: u32, me: u32) -> u32 {
    debug_assert!(sh < 32 && mb < 32 && me < 32);
    (21 << 26) | (rt << 21) | (ra << 16) | (sh << 11) | (mb << 6) | (me << 1)
}

/// MD-form rotate-and-mask with immediate (opcode 30): `rldicl` and
/// friends. Each 6-bit immediate is split: the low 5 bits sit at their
/// normal field position and bit 5 goes near the end of the word.
fn enc_md(rs: u32, ra: u32, sh: u32, mb: u32) -> u32 {
    debug_assert!(sh < 64 && mb < 64);
    (30 << 26)
        | (rs << 21)
        | (ra << 16)
        | ((sh & 31) << 11)
        | ((mb & 31) << 6)
        | (((mb >> 5) & 1) << 5)
        | (((sh >> 5) & 1) << 1)
}

/// mfspr/mtspr (XO 339/467): the SPR number is split across the RA and RB
/// fields.
fn enc_spr(mt: bool, rt: u32, spr: u32) -> u32 {
    let lo = spr & 0x1f;
    let hi = (spr >> 5) & 0x1f;
    enc_x(rt, lo, hi, if mt { 467 } else { 339 }, false)
}

/// VX-form vector op (opcode 4).
fn enc_vx(vrt: u32, vra: u32, vrb: u32, xo: u32) -> u32 {
    (4 << 26) | (vrt << 21) | (vra << 16) | (vrb << 11) | (xo << 1)
}

// Primary opcodes for D-form loads/stores and immediate ALU ops.
mod opcode {
    pub const MULLI: u32 = 7;
    pub const ADDI: u32 = 14;
    pub const ADDIS: u32 = 15;
    pub const ORI: u32 = 24;
    pub const ORIS: u32 = 25;
    pub const LWZ: u32 = 32;
    pub const STW: u32 = 36;
    pub const LBZ: u32 = 34;
    pub const STB: u32 = 38;
    pub const LHZ: u32 = 40;
    pub const LHA: u32 = 42;
    pub const STH: u32 = 44;
    pub const LFS: u32 = 48;
    pub const LFD: u32 = 50;
    pub const STFS: u32 = 52;
    pub const STFD: u32 = 54;
    pub const LD_GROUP: u32 = 58; // DS-form group (ld/ldu/lwa)
    pub const STD_GROUP: u32 = 62; // DS-form group (std/stdu)
    pub const RLWINM: u32 = 21;
    pub const MD_GROUP: u32 = 30;
}

// XO values for X-form ops (primary opcode 31).
mod xo {
    pub const CMP: u32 = 0;
    pub const CMPL: u32 = 32;
    pub const ADD: u32 = 266;
    pub const SUBF: u32 = 40;
    pub const NEG: u32 = 104;
    pub const SLW: u32 = 24;
    pub const SLD: u32 = 27;
    pub const AND: u32 = 28;
    pub const MULLD: u32 = 233;
    pub const DIVDU: u32 = 457;
    pub const DIVD: u32 = 489;
    pub const NOR: u32 = 124;
    pub const SRW: u32 = 536;
    pub const SRD: u32 = 539;
    pub const EXTSB: u32 = 954;
    pub const EXTSH: u32 = 922;
    pub const EXTSW: u32 = 986;
    pub const OR: u32 = 444;
    pub const XOR: u32 = 316;
    pub const NAND: u32 = 476;
    pub const ANDC: u32 = 60;
    pub const ORC: u32 = 412;
    pub const SRAW: u32 = 792;
    pub const SRAD: u32 = 794;
    pub const BCCLR: u32 = 16;
    pub const BCCTRL: u32 = 528;
}

impl LoadOP {
    pub(crate) fn from_type(ty: Type) -> LoadOP {
        match ty {
            I8 => LoadOP::Lbz,
            I16 => LoadOP::Lhz,
            I32 => LoadOP::Lwz,
            I64 => LoadOP::Ld,
            F32 => LoadOP::Lfs,
            F64 => LoadOP::Lfd,
            _ => panic!("unexpected load type: {ty}"),
        }
    }

    pub(crate) fn op_name(self) -> &'static str {
        match self {
            LoadOP::Lbz => "lbz",
            LoadOP::Lhz => "lhz",
            LoadOP::Lha => "lha",
            LoadOP::Lwz => "lwz",
            LoadOP::Lwa => "lwa",
            LoadOP::Ld => "ld",
            LoadOP::Lfs => "lfs",
            LoadOP::Lfd => "lfd",
        }
    }

    /// Is this a DS-form load (displacement divided by four)?
    pub(crate) fn is_ds_form(self) -> bool {
        matches!(self, LoadOP::Lwa | LoadOP::Ld)
    }

    pub(crate) fn dform_opcode(self) -> Option<u32> {
        match self {
            LoadOP::Lbz => Some(opcode::LBZ),
            LoadOP::Lhz => Some(opcode::LHZ),
            LoadOP::Lha => Some(opcode::LHA),
            LoadOP::Lwz => Some(opcode::LWZ),
            LoadOP::Lfs => Some(opcode::LFS),
            LoadOP::Lfd => Some(opcode::LFD),
            LoadOP::Lwa | LoadOP::Ld => None,
        }
    }

    pub(crate) fn ds_xo(self) -> u32 {
        match self {
            LoadOP::Lwa => 2,
            LoadOP::Ld => 0,
            _ => unreachable!(),
        }
    }

    pub(crate) fn encode_load(self, rt: u32, ra: u32, off: i64) -> u32 {
        if let Some(op) = self.dform_opcode() {
            enc_d(op, rt, ra, off as i16 as u16)
        } else {
            enc_ds(opcode::LD_GROUP, rt, ra, off, self.ds_xo())
        }
    }
}

impl StoreOP {
    pub(crate) fn from_type(ty: Type) -> StoreOP {
        match ty {
            I8 => StoreOP::Stb,
            I16 => StoreOP::Sth,
            I32 => StoreOP::Stw,
            I64 => StoreOP::Std,
            F32 => StoreOP::Stfs,
            F64 => StoreOP::Stfd,
            _ => panic!("unexpected store type: {ty}"),
        }
    }

    pub(crate) fn op_name(self) -> &'static str {
        match self {
            StoreOP::Stb => "stb",
            StoreOP::Sth => "sth",
            StoreOP::Stw => "stw",
            StoreOP::Std => "std",
            StoreOP::Stfs => "stfs",
            StoreOP::Stfd => "stfd",
        }
    }

    pub(crate) fn is_ds_form(self) -> bool {
        matches!(self, StoreOP::Std)
    }

    pub(crate) fn dform_opcode(self) -> Option<u32> {
        match self {
            StoreOP::Stb => Some(opcode::STB),
            StoreOP::Sth => Some(opcode::STH),
            StoreOP::Stw => Some(opcode::STW),
            StoreOP::Stfs => Some(opcode::STFS),
            StoreOP::Stfd => Some(opcode::STFD),
            StoreOP::Std => None,
        }
    }

    pub(crate) fn ds_xo(self) -> u32 {
        match self {
            StoreOP::Std => 0,
            _ => unreachable!(),
        }
    }

    pub(crate) fn encode_store(self, rs: u32, ra: u32, off: i64) -> u32 {
        if let Some(op) = self.dform_opcode() {
            enc_d(op, rs, ra, off as i16 as u16)
        } else {
            enc_ds(opcode::STD_GROUP, rs, ra, off, self.ds_xo())
        }
    }
}

impl AluOPRRR {
    pub(crate) fn op_name(self) -> &'static str {
        match self {
            AluOPRRR::Add => "add",
            AluOPRRR::Subf => "subf",
            AluOPRRR::And => "and",
            AluOPRRR::Or => "or",
            AluOPRRR::Xor => "xor",
            AluOPRRR::Nand => "nand",
            AluOPRRR::Nor => "nor",
            AluOPRRR::Andc => "andc",
            AluOPRRR::Orc => "orc",
            AluOPRRR::Slw => "slw",
            AluOPRRR::Srw => "srw",
            AluOPRRR::Sld => "sld",
            AluOPRRR::Srd => "srd",
            AluOPRRR::Sraw => "sraw",
            AluOPRRR::Srad => "srad",
            AluOPRRR::Mulld => "mulld",
            AluOPRRR::Divd => "divd",
            AluOPRRR::Divdu => "divdu",
            AluOPRRR::Mulhd => "mulhd",
            AluOPRRR::Mulhdu => "mulhdu",
            AluOPRRR::Mullw => "mullw",
            AluOPRRR::Mulhw => "mulhw",
            AluOPRRR::Mulhwu => "mulhwu",
        }
    }

    pub(crate) fn xo(self) -> u32 {
        match self {
            AluOPRRR::Add => xo::ADD,
            AluOPRRR::Subf => xo::SUBF,
            AluOPRRR::And => xo::AND,
            AluOPRRR::Or => xo::OR,
            AluOPRRR::Xor => xo::XOR,
            AluOPRRR::Nand => xo::NAND,
            AluOPRRR::Nor => xo::NOR,
            AluOPRRR::Andc => xo::ANDC,
            AluOPRRR::Orc => xo::ORC,
            AluOPRRR::Slw => xo::SLW,
            AluOPRRR::Srw => xo::SRW,
            AluOPRRR::Sld => xo::SLD,
            AluOPRRR::Srd => xo::SRD,
            AluOPRRR::Sraw => xo::SRAW,
            AluOPRRR::Srad => xo::SRAD,
            AluOPRRR::Mulld => xo::MULLD,
            AluOPRRR::Divd => xo::DIVD,
            AluOPRRR::Divdu => xo::DIVDU,
            AluOPRRR::Mulhd => 73,
            AluOPRRR::Mulhdu => 9,
            AluOPRRR::Mullw => 235,
            AluOPRRR::Mulhw => 75,
            AluOPRRR::Mulhwu => 11,
        }
    }

    /// True when this operation's assembly form places the destination
    /// register in the RA field (`and rA, rS, rB`) instead of the usual
    /// RT field (`add rT, rA, rB`). This applies to all the logical ops
    /// and all the shift ops (`slw rA, rS, rB`).
    pub(crate) fn reversed_operand_fields(self) -> bool {
        matches!(
            self,
            AluOPRRR::And
                | AluOPRRR::Or
                | AluOPRRR::Xor
                | AluOPRRR::Nand
                | AluOPRRR::Nor
                | AluOPRRR::Andc
                | AluOPRRR::Orc
                | AluOPRRR::Slw
                | AluOPRRR::Srw
                | AluOPRRR::Sld
                | AluOPRRR::Srd
                | AluOPRRR::Sraw
                | AluOPRRR::Srad
        )
    }

    pub(crate) fn encode(self, rt: u32, ra: u32, rb: u32) -> u32 {
        if self.reversed_operand_fields() {
            enc_x(ra, rt, rb, self.xo(), false)
        } else {
            enc_x(rt, ra, rb, self.xo(), false)
        }
    }
}

impl VecALUOp {
    pub(crate) fn op_name(self) -> &'static str {
        match self {
            VecALUOp::Vaddubm => "vaddubm",
            VecALUOp::Vadduhm => "vadduhm",
            VecALUOp::Vadduwm => "vadduwm",
            VecALUOp::Vaddfp => "vaddfp",
            VecALUOp::Vsububm => "vsububm",
            VecALUOp::Vsubfp => "vsubfp",
            VecALUOp::Vand => "vand",
            VecALUOp::Vor => "vor",
            VecALUOp::Vxor => "vxor",
        }
    }

    pub(crate) fn xo(self) -> u32 {
        match self {
            VecALUOp::Vaddubm => 0,
            VecALUOp::Vadduhm => 64,
            VecALUOp::Vadduwm => 128,
            VecALUOp::Vaddfp => 10,
            VecALUOp::Vsububm => 512,
            VecALUOp::Vsubfp => 74,
            VecALUOp::Vand => 1028,
            VecALUOp::Vor => 1156,
            VecALUOp::Vxor => 1220,
        }
    }

    pub(crate) fn encode(self, vrt: u32, vra: u32, vrb: u32) -> u32 {
        enc_vx(vrt, vra, vrb, self.xo())
    }
}

impl AluOPRR {
    pub(crate) fn op_name(self) -> &'static str {
        match self {
            AluOPRR::ExtSb => "extsb",
            AluOPRR::ExtSh => "extsh",
            AluOPRR::ExtSw => "extsw",
            AluOPRR::Neg => "neg",
        }
    }

    /// Encode a one-source operation. Note the PowerPC quirk: the extend
    /// operations take their source in the RT field position and destination
    /// in RA (`extsw rA, rS`), while `neg` follows the usual destination-
    /// first convention.
    pub(crate) fn encode(self, rd: u32, rs: u32) -> u32 {
        match self {
            AluOPRR::ExtSb => enc_x(rs, rd, 0, xo::EXTSB, false),
            AluOPRR::ExtSh => enc_x(rs, rd, 0, xo::EXTSH, false),
            AluOPRR::ExtSw => enc_x(rs, rd, 0, xo::EXTSW, false),
            AluOPRR::Neg => enc_x(rd, rs, 0, xo::NEG, false),
        }
    }
}

impl AluOPRRI {
    pub(crate) fn op_name(self) -> &'static str {
        match self {
            AluOPRRI::Addi => "addi",
            AluOPRRI::Addis => "addis",
        }
    }

    pub(crate) fn opcode(self) -> u32 {
        match self {
            AluOPRRI::Addi => opcode::ADDI,
            AluOPRRI::Addis => opcode::ADDIS,
        }
    }
}

impl FpuOPRRR {
    pub(crate) fn op_name(self) -> &'static str {
        match self {
            FpuOPRRR::Fadd => "fadd",
            FpuOPRRR::Fsub => "fsub",
            FpuOPRRR::Fdiv => "fdiv",
            FpuOPRRR::Fmul => "fmul",
        }
    }

    /// A-form 5-bit sub-opcode.
    fn xo5(self) -> u32 {
        match self {
            FpuOPRRR::Fadd => 21,
            FpuOPRRR::Fsub => 20,
            FpuOPRRR::Fdiv => 18,
            FpuOPRRR::Fmul => 25,
        }
    }

    pub(crate) fn encode(self, rd: u32, rs1: u32, rs2: u32) -> u32 {
        // `fmul` takes its second source in the A-form C field with the
        // B field reserved to zero.
        if self == FpuOPRRR::Fmul {
            enc_a(rd, rs1, 0, rs2, self.xo5(), false)
        } else {
            enc_a(rd, rs1, rs2, 0, self.xo5(), false)
        }
    }
}

impl FpuOPRRRR {
    pub(crate) fn op_name(self) -> &'static str {
        match self {
            FpuOPRRRR::Fmadd => "fmadd",
            FpuOPRRRR::Fmsub => "fmsub",
            FpuOPRRRR::Fnmadd => "fnmadd",
            FpuOPRRRR::Fnmsub => "fnmsub",
            FpuOPRRRR::Fsel => "fsel",
        }
    }

    fn xo5(self) -> u32 {
        match self {
            FpuOPRRRR::Fmadd => 29,
            FpuOPRRRR::Fmsub => 28,
            FpuOPRRRR::Fnmadd => 31,
            FpuOPRRRR::Fnmsub => 30,
            FpuOPRRRR::Fsel => 23,
        }
    }

    /// Encode `op ra*rc +/- rb`. The textual operand order is
    /// `op rT, rA, rC, rB`.
    pub(crate) fn encode(self, rd: u32, ra: u32, rb: u32, rc: u32) -> u32 {
        enc_a(rd, ra, rb, rc, self.xo5(), false)
    }
}

impl FpuOPRR {
    pub(crate) fn op_name(self) -> &'static str {
        match self {
            FpuOPRR::Fneg => "fneg",
            FpuOPRR::Fabs => "fabs",
            FpuOPRR::Fnabs => "fnabs",
            FpuOPRR::Fmr => "fmr",
            FpuOPRR::Frsp => "frsp",
            FpuOPRR::Fcfid => "fcfid",
            FpuOPRR::Fctidz => "fctidz",
            FpuOPRR::Fctiwz => "fctiwz",
            FpuOPRR::Fsqrt => "fsqrt",
        }
    }

    /// One-source ops place the destination in the RT field and the source
    /// in the RB field; the low bits carry the (verified) sub-opcode.
    pub(crate) fn encode(self, rd: u32, rs: u32) -> u32 {
        let tail: u32 = match self {
            FpuOPRR::Fneg => 0x050,
            FpuOPRR::Fabs => 0x210,
            FpuOPRR::Fnabs => 0x110,
            FpuOPRR::Fmr => 0x090,
            FpuOPRR::Frsp => 0x018,
            FpuOPRR::Fcfid => 0x69c,
            FpuOPRR::Fctidz => 0x65e,
            FpuOPRR::Fctiwz => 0x01e,
            FpuOPRR::Fsqrt => 0x02c,
        };
        (63 << 26) | (rd << 21) | (rs << 11) | tail
    }
}

//============================================================================
// Comparison and branch helpers.

/// Map an `IntCC` to the BO/BI fields of a CR0-based branch. `true` in the
/// first tuple element's position means "branch when condition true".
fn cc_to_bo_bi(cc: IntCC) -> (u32, u32) {
    match cc {
        IntCC::Equal => (12, 2),
        IntCC::NotEqual => (4, 2),
        IntCC::SignedLessThan | IntCC::UnsignedLessThan => (12, 0),
        IntCC::SignedGreaterThanOrEqual | IntCC::UnsignedGreaterThanOrEqual => (4, 0),
        IntCC::SignedGreaterThan | IntCC::UnsignedGreaterThan => (12, 1),
        IntCC::SignedLessThanOrEqual | IntCC::UnsignedLessThanOrEqual => (4, 1),
    }
}

/// CR1 bit numbers after an `fcmpu cr1, ...`.
const CR1_LT: u32 = 4;
const CR1_GT: u32 = 5;
const CR1_EQ: u32 = 6;
const CR1_SO: u32 = 7;
/// Scratch CR bit (cr7's LT), unused by the comparisons we emit.
const CR_TMP: u32 = 28;

const CRAND: u32 = 257;
const CROR: u32 = 449;
const CRANDC: u32 = 129;

/// Emit `fcmpu` plus any CR-bit combining needed for `cc`, then
/// materialize 0/1 into `rd`.
fn emit_fcmp_setcc(sink: &mut MachBuffer<Inst>, rd: Writable<Reg>, cc: FloatCC, a: u32, b: u32) {
    // fcmpu bf=1(cr1), a, b
    put_word(sink, (63 << 26) | (1 << 23) | (a << 16) | (b << 11));

    // Plan: list of (xo, ba, bb) CR-ops writing to CR_TMP, the final bit to
    // test, and whether the test is inverted.
    let (ops, final_bit, negate): (alloc::vec::Vec<(u32, u32, u32)>, u32, bool) = match cc {
        FloatCC::Ordered => (vec![], CR1_SO, true),
        FloatCC::Unordered => (vec![], CR1_SO, false),
        FloatCC::Equal => (vec![(CRANDC, CR1_EQ, CR1_SO)], CR_TMP, false),
        FloatCC::NotEqual => (vec![(CROR, CR1_EQ, CR1_SO)], CR_TMP, true),
        // LT | GT: equivalent to !(EQ) on an ordered compare where SO
        // implies neither; combine directly.
        FloatCC::OrderedNotEqual => (
            vec![(CROR, CR_TMP, CR1_LT), (CROR, CR_TMP, CR1_GT)],
            CR_TMP,
            false,
        ),
        // UN | EQ
        FloatCC::UnorderedOrEqual => (vec![(CROR, CR1_SO, CR1_EQ)], CR_TMP, false),
        FloatCC::LessThan => (vec![], CR1_LT, false),
        FloatCC::LessThanOrEqual => (vec![(CROR, CR1_LT, CR1_EQ)], CR_TMP, false),
        FloatCC::GreaterThan => (vec![], CR1_GT, false),
        FloatCC::GreaterThanOrEqual => (vec![(CROR, CR1_GT, CR1_EQ)], CR_TMP, false),
        FloatCC::UnorderedOrLessThan => (vec![(CROR, CR1_SO, CR1_LT)], CR_TMP, false),
        FloatCC::UnorderedOrGreaterThan => (vec![(CROR, CR1_SO, CR1_GT)], CR_TMP, false),
        FloatCC::UnorderedOrLessThanOrEqual => (
            vec![(CROR, CR1_SO, CR1_LT), (CROR, CR_TMP, CR1_EQ)],
            CR_TMP,
            false,
        ),
        FloatCC::UnorderedOrGreaterThanOrEqual => (
            vec![(CROR, CR1_SO, CR1_GT), (CROR, CR_TMP, CR1_EQ)],
            CR_TMP,
            false,
        ),
    };

    for &(xo, ba, bb) in &ops {
        put_word(sink, enc_cr(xo, CR_TMP, ba, bb));
    }

    let rdn = writable_reg_num(rd);
    // li rd, 0
    put_word(sink, enc_d(opcode::ADDI, rdn, 0, 0));
    // bc <test>, +1
    let bo = if negate { 12 } else { 4 };
    put_word(sink, enc_bc(bo, final_bit, 2));
    // li rd, 1
    put_word(sink, enc_d(opcode::ADDI, rdn, 0, 1));
}

fn enc_compare(cmp: &IntegerCompare) -> u32 {
    let xo = match cmp.kind {
        IntCC::UnsignedLessThan
        | IntCC::UnsignedGreaterThan
        | IntCC::UnsignedLessThanOrEqual
        | IntCC::UnsignedGreaterThanOrEqual => xo::CMPL,
        _ => xo::CMP,
    };
    // BF = 0 (CR field 0); L bit selects 32- vs 64-bit compare.
    let field5 = u32::from(cmp.is64);
    enc_x(field5, reg_num(cmp.rs1), reg_num(cmp.rs2), xo, false)
}

fn emit_b(sink: &mut MachBuffer<Inst>, label: MachLabel) {
    let at = sink.cur_offset();
    sink.use_label_at_offset(at, label, LabelUse);
    sink.add_uncond_branch(at, at + 4, label);
    put_word(sink, ENC_B_BASE);
}

//============================================================================
// Addressing-mode resolution.

fn amode_parts(amode: &AMode, state: &EmitState) -> (Reg, i64) {
    match amode {
        &AMode::RegOffset(reg, off) => (reg, off),
        &AMode::SPOffset(off) => (stack_reg(), off),
        &AMode::FPOffset(off) => (fp_reg(), off),
        &AMode::SlotOffset(off) => (
            stack_reg(),
            off + i64::from(state.frame_layout.outgoing_args_size),
        ),
        AMode::Const(..) => {
            panic!("constant-pool addressing is not implemented on ppu64")
        }
    }
}

/// Split a (possibly large) offset into `addis` high and `addi` low parts
/// such that `(hi << 16) + lo == offset` and both fit in signed 16 bits.
fn split_offset(off: i64) -> (i64, i64) {
    let hi = ((off + 0x8000) >> 16) & 0xffff;
    let lo = off - (hi << 16);
    debug_assert!(lo >= -32768 && lo <= 32767);
    (hi, lo)
}

/// Materialize `base + off` for a load/store whose offset does not fit
/// the addressing form. This previously computed the address into r0,
/// but r0 reads as literal zero whenever it appears in an operand field
/// (including the RA/base field of the very load or store that would
/// consume it), so the resulting code was silently wrong. Out-of-range
/// offsets must instead be folded at lowering time into an allocated
/// temp (see LoadAddr); panic loudly if this path is ever reached.
fn materialize_address(sink: &mut MachBuffer<Inst>, base: Reg, off: i64) -> Reg {
    let _ = (sink, base, off);
    panic!("out-of-range memory offset reached emission; fold into a temp vreg during lowering");
}

//============================================================================
// Emission state.

#[derive(Default, Clone, Debug)]
pub struct EmitState {
    user_stack_map: Option<ir::UserStackMap>,
    ctrl_plane: cranelift_control::ControlPlane,
    frame_layout: FrameLayout,
}

impl EmitState {
    pub fn frame_layout(&self) -> &FrameLayout {
        &self.frame_layout
    }
}

impl MachInstEmitState<Inst> for EmitState {
    fn frame_layout(&self) -> &FrameLayout {
        &self.frame_layout
    }

    fn new(
        abi: &crate::machinst::abi::Callee<Ppu64MachineDeps>,
        ctrl_plane: cranelift_control::ControlPlane,
    ) -> Self {
        EmitState {
            user_stack_map: None,
            ctrl_plane,
            frame_layout: abi.frame_layout().clone(),
        }
    }

    fn pre_safepoint(&mut self, user_stack_map: Option<ir::UserStackMap>) {
        self.user_stack_map = user_stack_map;
    }

    fn ctrl_plane_mut(&mut self) -> &mut cranelift_control::ControlPlane {
        &mut self.ctrl_plane
    }

    fn take_ctrl_plane(self) -> cranelift_control::ControlPlane {
        self.ctrl_plane
    }
}

//============================================================================
// Instruction emission.

impl MachInstEmit for Inst {
    type State = EmitState;
    type Info = EmitInfo;

    fn pretty_print_inst(&self, state: &mut Self::State) -> String {
        self.print_with_state(state)
    }

    fn emit(&self, sink: &mut MachBuffer<Inst>, _emit_info: &Self::Info, state: &mut Self::State) {
        let start_off = sink.cur_offset();

        match self {
            Inst::Nop0 => {}
            Inst::Nop4 => {
                put_word(sink, enc_d(opcode::ORI, 0, 0, 0));
            }

            Inst::Mov { rd, rm, ty } => {
                if ty.is_float() {
                    put_word(
                        sink,
                        FpuOPRR::Fmr.encode(writable_reg_num(*rd), reg_num(*rm)),
                    );
                } else {
                    put_word(
                        sink,
                        AluOPRRR::Or.encode(writable_reg_num(*rd), reg_num(*rm), reg_num(*rm)),
                    );
                }
            }

            Inst::AluRRR {
                alu_op,
                rd,
                rs1,
                rs2,
            } => {
                put_word(
                    sink,
                    alu_op.encode(writable_reg_num(*rd), reg_num(*rs1), reg_num(*rs2)),
                );
            }

            Inst::AluRR { alu_op, rd, rs } => {
                put_word(sink, alu_op.encode(writable_reg_num(*rd), reg_num(*rs)));
            }

            Inst::AluRRSimm16 {
                alu_op,
                rd,
                ra,
                simm,
            } => {
                put_word(
                    sink,
                    enc_d(
                        alu_op.opcode(),
                        writable_reg_num(*rd),
                        reg_num(*ra),
                        simm.bits() as u16,
                    ),
                );
            }

            Inst::Li32 { rd, imm } => {
                emit_li32(sink, *rd, u64::from(*imm));
            }

            Inst::LoadConst64 { rd, imm } => {
                emit_li64(sink, *rd, *imm);
            }

            Inst::Load { op, rd, from, .. } => {
                let (base, off) = amode_parts(from, state);
                let (base, off) = if offset_ok_for(*op, off) {
                    (base, off)
                } else {
                    (materialize_address(sink, base, off), 0)
                };
                put_word(
                    sink,
                    op.encode_load(writable_reg_num(*rd), reg_num(base), off),
                );
            }

            Inst::Store { op, src, to, .. } => {
                let (base, off) = amode_parts(to, state);
                let (base, off) = if offset_ok_for_store(*op, off) {
                    (base, off)
                } else {
                    (materialize_address(sink, base, off), 0)
                };
                put_word(sink, op.encode_store(reg_num(*src), reg_num(base), off));
            }

            Inst::LoadAddr { rd, mem } => {
                let (base, off) = amode_parts(mem, state);
                if off >= -32768 && off <= 32767 {
                    put_word(
                        sink,
                        enc_d(
                            opcode::ADDI,
                            writable_reg_num(*rd),
                            reg_num(base),
                            off as i16 as u16,
                        ),
                    );
                } else {
                    let (hi, lo) = split_offset(off);
                    put_word(
                        sink,
                        enc_d(
                            opcode::ADDIS,
                            writable_reg_num(*rd),
                            reg_num(base),
                            hi as u16,
                        ),
                    );
                    if lo != 0 {
                        put_word(
                            sink,
                            enc_d(
                                opcode::ADDI,
                                writable_reg_num(*rd),
                                writable_reg_num(*rd),
                                lo as i16 as u16,
                            ),
                        );
                    }
                }
            }

            Inst::Args { args: _ } | Inst::Rets { rets: _ } => {}
            Inst::Ret {} => {
                put_word(sink, enc_bx(20, 0, 0, xo::BCCLR, false));
            }

            Inst::Extend {
                rd,
                rn,
                signed,
                from_bits,
            } => {
                emit_extend(sink, *rd, *rn, *signed, *from_bits);
            }

            Inst::Call { info } => {
                let CallInfo { dest, .. } = &**info;
                sink.add_reloc(Reloc::Ppc64Call, dest, 0);
                put_word(sink, ENC_B_BASE | 1);
                sink.add_call_site();
            }

            Inst::CallInd { info } => {
                let CallInfo { dest, .. } = &**info;
                put_word(sink, enc_spr(true, reg_num(*dest), 9));
                put_word(sink, enc_bx(20, 0, 0, xo::BCCTRL, true));
                sink.add_call_site();
            }

            Inst::Jmp { label } => {
                emit_b(sink, *label);
            }

            Inst::CondBr {
                taken,
                not_taken,
                kind,
            } => {
                put_word(sink, enc_compare(kind));
                let inv = kind.inverse();
                let (bo, bi) = cc_to_bo_bi(inv.kind);
                put_word(sink, enc_bc(bo, bi, 2));

                match (taken.as_label(), not_taken.as_label()) {
                    (Some(t), Some(n)) => {
                        emit_b(sink, t);
                        emit_b(sink, n);
                    }
                    (Some(t), None) => {
                        emit_b(sink, t);
                    }
                    (None, Some(n)) => {
                        emit_b(sink, n);
                    }
                    (None, None) => panic!("conditional branch with no real target"),
                }
            }

            Inst::BrTable {
                index,
                tmp1,
                tmp2,
                default,
                targets,
            } => {
                let n = targets.len() as u32;
                let idx = reg_num(*index);
                let tmp1_n = writable_reg_num(*tmp1);
                let tmp2_n = writable_reg_num(*tmp2);
                // li tmp2, n
                put_word(sink, enc_d(opcode::ADDI, tmp2_n, 0, n as u16));
                // cmplw cr0, index, tmp2
                put_word(sink, enc_x(0, idx, tmp2_n, xo::CMPL, false));
                // blt 1f (skip b default) ; b default
                put_word(sink, enc_bc(12, 0, 2)); // blt cr0, 1f (BO=12, BI=0)
                emit_b(sink, *default);
                // 1: bl 2f ; 2: mflr tmp1
                let lbl = sink.get_label();
                let at_bl = sink.cur_offset();
                sink.use_label_at_offset(at_bl, lbl, LabelUse);
                put_word(sink, ENC_B_BASE | 1); // bl
                sink.bind_label(lbl, state.ctrl_plane_mut());
                put_word(sink, enc_spr(false, tmp1_n, 8)); // mflr tmp1
                // addi tmp1, tmp1, 28  (to table base)
                put_word(sink, enc_d(opcode::ADDI, tmp1_n, tmp1_n, 28));
                // sldi tmp2, index, 2  (via li 2; sld)
                put_word(sink, enc_d(opcode::ADDI, tmp2_n, 0, 2));
                put_word(sink, enc_x(idx, tmp2_n, tmp2_n, xo::SLD, false));
                // add tmp1, tmp1, tmp2
                put_word(
                    sink,
                    AluOPRRR::Add.encode(tmp1_n, tmp1_n, tmp2_n),
                );
                // mtctr tmp1; bctr
                put_word(sink, enc_spr(true, tmp1_n, 9));
                put_word(sink, enc_bx(20, 0, 0, xo::BCCTRL, false));
                // table: b target0, b target1, ...
                for label in targets.iter() {
                    emit_b(sink, *label);
                }
            }

            Inst::TrapIf { cmp, trap_code } => {
                put_word(sink, enc_compare(cmp));
                let inv = cmp.inverse();
                let (bo, bi) = cc_to_bo_bi(inv.kind);
                put_word(sink, enc_bc(bo, bi, 2));
                sink.add_trap(*trap_code);
                put_word(sink, 0);
            }

            Inst::SetCC { rd, cmp } => {
                put_word(sink, enc_compare(cmp));
                let rdn = writable_reg_num(*rd);
                // li rd, 0
                put_word(sink, enc_d(opcode::ADDI, rdn, 0, 0));
                let inv = cmp.inverse();
                let (bo, bi) = cc_to_bo_bi(inv.kind);
                // bc <inverted>, +2 words: when the condition is false,
                // skip over the store of 1.
                put_word(sink, enc_bc(bo, bi, 2));
                // li rd, 1
                put_word(sink, enc_d(opcode::ADDI, rdn, 0, 1));
            }

            Inst::Udf { trap_code } => {
                sink.add_trap(*trap_code);
                put_word(sink, 0);
            }

            Inst::MovFromPReg { rd, rm } => {
                let rm = Reg::from(*rm);
                put_word(
                    sink,
                    AluOPRRR::Or.encode(writable_reg_num(*rd), reg_num(rm), reg_num(rm)),
                );
            }

            Inst::Mfspr { rd, spr } => {
                put_word(sink, enc_spr(false, writable_reg_num(*rd), u32::from(*spr)));
            }

            Inst::Mtspr { rs, spr } => {
                put_word(sink, enc_spr(true, reg_num(*rs), u32::from(*spr)));
            }

            Inst::FpuRRR {
                alu_op,
                rd,
                rs1,
                rs2,
            } => {
                put_word(
                    sink,
                    alu_op.encode(writable_reg_num(*rd), reg_num(*rs1), reg_num(*rs2)),
                );
            }

            Inst::FpuRRRR {
                alu_op,
                rd,
                ra,
                rb,
                rc,
            } => {
                put_word(
                    sink,
                    alu_op.encode(
                        writable_reg_num(*rd),
                        reg_num(*ra),
                        reg_num(*rb),
                        reg_num(*rc),
                    ),
                );
            }

            Inst::FpuRR { alu_op, rd, rs } => {
                put_word(sink, alu_op.encode(writable_reg_num(*rd), reg_num(*rs)));
            }

            Inst::FCmpSetCC { rd, cc, a, b } => {
                emit_fcmp_setcc(sink, *rd, *cc, reg_num(*a), reg_num(*b));
            }

            Inst::MoveGprToFpr { rd, rs } => {
                let rdn = writable_reg_num(*rd);
                let rsn = reg_num(*rs);
                // std rs, 0(sp); lfd rd, 0(sp)
                put_word(
                    sink,
                    enc_ds(opcode::STD_GROUP, rsn, reg_num(stack_reg()), 0, 0),
                );
                put_word(sink, enc_d(opcode::LFD, rdn, reg_num(stack_reg()), 0));
            }

            Inst::MoveFprToGpr { rd, rs } => {
                let rdn = writable_reg_num(*rd);
                let rsn = reg_num(*rs);
                // stfd rs(FPR), 0(sp); ld rd(GPR), 0(sp): raw bit move.
                put_word(
                    sink,
                    enc_d(opcode::STFD, rsn, reg_num(stack_reg()), 0),
                );
                put_word(
                    sink,
                    enc_ds(opcode::LD_GROUP, rdn, reg_num(stack_reg()), 0, 0),
                );
            }

            Inst::MoveGprToFpr32 { rd, rs } => {
                let rdn = writable_reg_num(*rd);
                let rsn = reg_num(*rs);
                // stw rs, 4(sp); lfs rd, 4(sp): the load converts the
                // 32-bit pattern to the internal single-precision form.
                put_word(sink, enc_d(opcode::STW, rsn, reg_num(stack_reg()), 0));
                put_word(sink, enc_d(opcode::LFS, rdn, reg_num(stack_reg()), 0));
            }

            Inst::MoveFprBitsToGpr32 { rd, rs } => {
                let rdn = writable_reg_num(*rd);
                let rsn = reg_num(*rs);
                // stfs rs, 0(sp); lwz rd, 0(sp): stores the raw single
                // bit pattern, read back as an integer.
                put_word(sink, enc_d(opcode::STFS, rsn, reg_num(stack_reg()), 0));
                put_word(
                    sink,
                    enc_d(opcode::LWZ, reg_num(*rs), reg_num(stack_reg()), 0),
                );
            }

            Inst::MoveFprToGprI32 { rd, rs } => {
                let rdn = writable_reg_num(*rd);
                let rsn = reg_num(*rs);
                // stfd rs, 0(sp); lwz rd, 0(sp): big-endian high word of
                // the stored double carries the fctiwz integer result.
                put_word(sink, enc_ds(opcode::STD_GROUP, rsn, reg_num(stack_reg()), 0, 0));
                put_word(
                    sink,
                    enc_d(opcode::LWZ, rdn, reg_num(stack_reg()), 0),
                );
            }

            Inst::Clz { rd, rn, is64 } => {
                // cntlzd/cntlzw take their source in the RT field position
                // and destination in RA (`cntlzd rA, rS`).
                let xo = if *is64 { 58 } else { 26 };
                put_word(
                    sink,
                    enc_x(reg_num(*rn), writable_reg_num(*rd), 0, xo, false),
                );
            }

            Inst::Rlwinm { rd, rn, sh, mb, me } => {
                put_word(
                    sink,
                    enc_m(
                        reg_num(*rn),
                        writable_reg_num(*rd),
                        u32::from(*sh),
                        u32::from(*mb),
                        u32::from(*me),
                    ),
                );
            }

            Inst::Rldicl { rd, rn, sh, mb } => {
                put_word(sink, enc_md(reg_num(*rn), writable_reg_num(*rd), (*sh).into(), (*mb).into()));
            }

            Inst::LoadExtName { rd, name, offset } => {
                // lis rd, sym@ha(+offset)
                let rdn = writable_reg_num(*rd);
                sink.add_reloc(Reloc::Ppc64Addr16Ha, &**name, *offset);
                put_word(sink, enc_d(opcode::ADDIS, rdn, 0, 0));
                // ori rd, rd, sym@l(+offset)
                sink.add_reloc(Reloc::Ppc64Addr16Lo, &**name, *offset);
                put_word(sink, enc_d(opcode::ORI, rdn, rdn, 0));
            }

            Inst::PromoteF32ToF64 { rd, rs } => {
                let rdn = writable_reg_num(*rd);
                let rsn = reg_num(*rs);
                // PPC FPRs hold values in internal double form — an f32
                // loaded via lfs is already widened, so promote is a copy.
                // (A stfs+lfd roundtrip here composes a garbage double:
                // stfs writes only 4 bytes but lfd reads 8.)
                put_word(sink, FpuOPRR::Fmr.encode(rdn, rsn));
            }

            Inst::VecALU { op, rd, ra, rb } => {
                put_word(
                    sink,
                    op.encode(writable_reg_num(*rd), reg_num(*ra), reg_num(*rb)),
                );
            }

            Inst::VecBitSel { rd, ra, rb, rc } => {
                // VA-form vsel: op 4 | VRT | VRA | VRB | VRC | XO(6).
                put_word(
                    sink,
                    (4 << 26)
                        | (writable_reg_num(*rd) << 21)
                        | (reg_num(*ra) << 16)
                        | (reg_num(*rb) << 11)
                        | (reg_num(*rc) << 6)
                        | 42,
                );
            }

            Inst::VecLoad { rd, mem, .. } => {
                let (base, off) = amode_parts(mem, state);
                if off == 0 {
                    put_word(
                        sink,
                        enc_x(writable_reg_num(*rd), 0, reg_num(base), 103, false),
                    );
                } else {
                    // materialize base+off into spilltmp (r11)
                    let tmp = spilltmp_reg();
                    let (hi, lo) = split_offset(off);
                    put_word(
                        sink,
                        enc_d(opcode::ADDIS, reg_num(tmp), reg_num(base), hi as u16),
                    );
                    if lo != 0 {
                        put_word(
                            sink,
                            enc_d(opcode::ADDI, reg_num(tmp), reg_num(tmp), lo as u16),
                        );
                    }
                    put_word(sink, enc_x(writable_reg_num(*rd), 0, reg_num(tmp), 103, false));
                }
            }

            Inst::VecStore { rs, mem, .. } => {
                let (base, off) = amode_parts(mem, state);
                if off == 0 {
                    put_word(
                        sink,
                        enc_x(reg_num(*rs), 0, reg_num(base), 231, false),
                    );
                } else {
                    let tmp = spilltmp_reg();
                    let (hi, lo) = split_offset(off);
                    put_word(
                        sink,
                        enc_d(opcode::ADDIS, reg_num(tmp), reg_num(base), hi as u16),
                    );
                    if lo != 0 {
                        put_word(
                            sink,
                            enc_d(opcode::ADDI, reg_num(tmp), reg_num(tmp), lo as u16),
                        );
                    }
                    put_word(sink, enc_x(reg_num(*rs), 0, reg_num(tmp), 231, false));
                }
            }

            Inst::Fence => {
                // sync 0 (hwsync) for SC
                put_word(sink, enc_x(0, 0, 0, 598, false));
            }

            Inst::AtomicLoad { op, rd, mem, .. } => {
                // sync; load
                put_word(sink, enc_x(0, 0, 0, 598, false));
                let (base, off) = amode_parts(mem, state);
                let (base, off) = if off == 0 {
                    (base, 0)
                } else {
                    (materialize_address(sink, base, off), 0)
                };
                put_word(sink, op.encode_load(writable_reg_num(*rd), reg_num(base), off));
            }

            Inst::AtomicStore { op, src, mem, .. } => {
                // store; sync
                let (base, off) = amode_parts(mem, state);
                let (base, off) = if off == 0 {
                    (base, 0)
                } else {
                    (materialize_address(sink, base, off), 0)
                };
                put_word(sink, op.encode_store(reg_num(*src), reg_num(base), off));
                put_word(sink, enc_x(0, 0, 0, 598, false));
            }

            Inst::AtomicRmw { ty, op, rd, mem, rs } => {
                // NOTE: reservation ops must carry the address in RB with
                // RA=0; RPCS3's interpreter mishandles RA-based EAs here.
                let (base, off) = amode_parts(mem, state);
                let tmp = spilltmp_reg();
                let addr = if off == 0 {
                    base
                } else {
                    let (hi, lo) = split_offset(off);
                    put_word(sink, enc_d(opcode::ADDIS, reg_num(tmp), reg_num(base), hi as u16));
                    if lo != 0 {
                        put_word(sink, enc_d(opcode::ADDI, reg_num(tmp), reg_num(tmp), lo as u16));
                    }
                    tmp
                };
                let is64 = ty.bits() > 32;
                // loop:
                let loop_off = sink.cur_offset();
                // lwarx (XO 20) / ldarx (XO 84): rd = MEM[addr], reserve.
                // Address goes in RB; RA stays 0.
                put_word(sink, enc_x(writable_reg_num(*rd), 0, reg_num(addr), if is64 { 84 } else { 20 }, false));
                // Compute new value into spilltmp based on op.
                match op {
                    AtomicRmwOp::Add => {
                        put_word(sink, AluOPRRR::Add.encode(reg_num(tmp), writable_reg_num(*rd), reg_num(*rs)));
                    }
                    AtomicRmwOp::Sub => {
                        // subf tmp, rs, rd => rd - rs
                        put_word(sink, AluOPRRR::Subf.encode(reg_num(tmp), reg_num(*rs), writable_reg_num(*rd)));
                    }
                    AtomicRmwOp::And => {
                        put_word(sink, AluOPRRR::And.encode(reg_num(tmp), writable_reg_num(*rd), reg_num(*rs)));
                    }
                    AtomicRmwOp::Or => {
                        put_word(sink, AluOPRRR::Or.encode(reg_num(tmp), writable_reg_num(*rd), reg_num(*rs)));
                    }
                    AtomicRmwOp::Xor => {
                        put_word(sink, AluOPRRR::Xor.encode(reg_num(tmp), writable_reg_num(*rd), reg_num(*rs)));
                    }
                    AtomicRmwOp::Nand => {
                        put_word(sink, AluOPRRR::Nand.encode(reg_num(tmp), writable_reg_num(*rd), reg_num(*rs)));
                    }
                    AtomicRmwOp::Xchg => {
                        put_word(sink, enc_d(opcode::ORI, reg_num(tmp), reg_num(*rs), 0));
                    }
                    _ => {
                        // Min/max not yet supported: fall back to xchg-ish behavior
                        put_word(sink, enc_d(opcode::ORI, reg_num(tmp), reg_num(*rs), 0));
                    }
                }
                // stwcx. (XO 301) / stdcx. (XO 429): conditional store, CR0.EQ=1 on success
                put_word(sink, enc_x(reg_num(tmp), 0, reg_num(addr), if is64 { 214 } else { 150 }, true));
                // bne- loop: backward branch to loop start. BO=4 (branch if
                // CR bit false), BI=2 (CR0.EQ). BD is in words relative to
                // this instruction; computed directly since loop_off is known.
                let at_bne = sink.cur_offset();
                let delta_words = ((loop_off as i64 - at_bne as i64) / 4) as i32;
                put_word(sink, enc_bc(4, 2, delta_words));
                // sync for SC release
                put_word(sink, enc_x(0, 0, 0, 598, false));
            }

            Inst::AtomicCas { rd, mem, expected, replacement, ty, .. } => {
                let (base, off) = amode_parts(mem, state);
                let tmp = spilltmp_reg();
                let addr = if off == 0 { base } else {
                    let (hi, lo) = split_offset(off);
                    put_word(sink, enc_d(opcode::ADDIS, reg_num(tmp), reg_num(base), hi as u16));
                    if lo != 0 { put_word(sink, enc_d(opcode::ADDI, reg_num(tmp), reg_num(tmp), lo as u16)); }
                    tmp
                };
                let is64 = ty.bits() > 32;
                // loop:
                let loop_off = sink.cur_offset();
                // lwarx (XO 20) / ldarx (XO 84): rd = MEM[addr]
                put_word(sink, enc_x(writable_reg_num(*rd), 0, reg_num(addr), if is64 { 84 } else { 20 }, false));
                // cmp/cmpd cr0, expected, rd (bf:L in RT slot; L=is64)
                put_word(sink, enc_x(u32::from(is64), reg_num(*expected), writable_reg_num(*rd), xo::CMP, false));
                // bne- +3: skip stwcx./bne when not equal (land on sync)
                put_word(sink, enc_bc(4, 2, 3));
                // stwcx. (XO 301) / stdcx. (XO 429): replacement, sets CR0.EQ
                put_word(sink, enc_x(reg_num(*replacement), 0, reg_num(addr), if is64 { 214 } else { 150 }, true));
                // bne- loop: retry on reservation failure (backward, known)
                let at_bne = sink.cur_offset();
                let delta_words = ((loop_off as i64 - at_bne as i64) / 4) as i32;
                put_word(sink, enc_bc(4, 2, delta_words));
                // sync:
                put_word(sink, enc_x(0, 0, 0, 598, false));
            }

            Inst::DebugTrap => {
                // tw 31,0,0: unconditional program interrupt
                put_word(sink, enc_x(31, 0, 0, 4, false));
            }

            Inst::Unwind { .. } | Inst::DummyUse { .. } => {}
        }

        let end_off = sink.cur_offset();
        debug_assert!(
            (end_off - start_off) <= Inst::worst_case_size(),
            "ppu64 instruction exceeded worst-case size: {} > {} ({self:?})",
            end_off - start_off,
            Inst::worst_case_size(),
        );
    }
}

impl Inst {
    fn print_with_state(&self, _state: &mut EmitState) -> String {
        format_inst(self)
    }
}

fn offset_ok_for(op: LoadOP, off: i64) -> bool {
    let in_range = off >= -32768 && off <= 32767;
    if op.is_ds_form() {
        in_range && off % 4 == 0
    } else {
        in_range
    }
}

fn offset_ok_for_store(op: StoreOP, off: i64) -> bool {
    let in_range = off >= -32768 && off <= 32767;
    if op.is_ds_form() {
        in_range && off % 4 == 0
    } else {
        in_range
    }
}

fn emit_li32(sink: &mut MachBuffer<Inst>, rd: Writable<Reg>, imm: u64) {
    debug_assert!(imm <= u32::MAX as u64);
    let rdn = writable_reg_num(rd);
    let val = imm as u32;
    if (val as i32) >= -32768 && (val as i32) <= 32767 {
        // li rd, imm (addi has normal dest-first layout, RA=0 reads zero)
        put_word(sink, enc_d(opcode::ADDI, rdn, 0, val as u16));
    } else {
        // PowerPC has no hardwired zero register: r0 only reads as zero in
        // specific D-form RA fields (addi/addis, load/store addressing).
        // The ORI family reads its RS operand for real, so unsigned-16-bit
        // constants take the same two-instruction sequence as full 32-bit
        // values, with a zero high half.
        put_word(sink, enc_d(opcode::ADDIS, rdn, 0, (val >> 16) as u16));
        put_word(sink, enc_d(opcode::ORI, rdn, rdn, val as u16));
    }
}

fn emit_li64(sink: &mut MachBuffer<Inst>, rd: Writable<Reg>, imm: u64) {
    if imm <= u32::MAX as u64 {
        emit_li32(sink, rd, imm);
        return;
    }
    let rdn = writable_reg_num(rd);
    let hi = (imm >> 32) as u32;
    let lo = imm as u32;
    // lis   rd, hi_hi
    // ori   rd, rd, hi_lo
    // rldicr rd, rd, 32, 31   (shift high half into place; no scratch —
    //                          r0 must never be consumed as an operand,
    //                          it reads as literal zero)
    // oris  rd, rd, lo_hi
    // ori   rd, rd, lo_lo
    put_word(sink, enc_d(opcode::ADDIS, rdn, 0, (hi >> 16) as u16));
    put_word(sink, enc_d(opcode::ORI, rdn, rdn, hi as u16));
    // rldicr: same MD-form encoding as rldicl with XO bits 27-29 = 001.
    put_word(sink, enc_md(rdn, rdn, 32, 31) | (1 << 2));
    put_word(sink, enc_d(opcode::ORIS, rdn, rdn, (lo >> 16) as u16));
    put_word(sink, enc_d(opcode::ORI, rdn, rdn, lo as u16));
}

fn emit_extend(sink: &mut MachBuffer<Inst>, rd: Writable<Reg>, rn: Reg, signed: bool, bits: u8) {
    let rdn = writable_reg_num(rd);
    let rnn = reg_num(rn);
    if signed {
        let op = match bits {
            8 => AluOPRR::ExtSb,
            16 => AluOPRR::ExtSh,
            32 => AluOPRR::ExtSw,
            _ => panic!("unsupported extend width"),
        };
        put_word(sink, op.encode(rdn, rnn));
    } else {
        // Zero-extension must produce a clean 64-bit value. M-form
        // (rlwinm) replicates its 32-bit result into both halves of the
        // register, which would leak stale bits into 64-bit consumers —
        // use rldicl to clear the upper half of the doubleword instead.
        let mb = match bits {
            8 => 56,
            16 => 48,
            32 => 32,
            _ => panic!("unsupported extend width"),
        };
        put_word(sink, enc_md(rnn, rdn, 0, mb));
    }
}

//============================================================================
// Disassembly.

fn fmt_cond(cc: IntCC) -> String {
    match cc {
        IntCC::Equal => "==".into(),
        IntCC::NotEqual => "!=".into(),
        IntCC::SignedLessThan => "<".into(),
        IntCC::SignedGreaterThan => ">".into(),
        IntCC::SignedLessThanOrEqual => "<=".into(),
        IntCC::SignedGreaterThanOrEqual => ">=".into(),
        IntCC::UnsignedLessThan => "<u".into(),
        IntCC::UnsignedGreaterThan => ">u".into(),
        IntCC::UnsignedLessThanOrEqual => "<=u".into(),
        IntCC::UnsignedGreaterThanOrEqual => ">=u".into(),
    }
}

fn format_inst(inst: &Inst) -> String {
    let freg = |r: &Reg| reg_name(*r);
    let fwreg = |r: &Writable<Reg>| reg_name(r.to_reg());
    match inst {
        Inst::Nop0 => "##zero-length nop".to_string(),
        Inst::Nop4 => "nop".to_string(),
        Inst::Mov { rd, rm, ty } => {
            if ty.is_float() {
                format!("fmr {},{}", fwreg(rd), freg(rm))
            } else {
                format!("mr {},{}", fwreg(rd), freg(rm))
            }
        }
        Inst::AluRRR {
            alu_op,
            rd,
            rs1,
            rs2,
        } => format!(
            "{} {},{},{}",
            alu_op.op_name(),
            fwreg(rd),
            freg(rs1),
            freg(rs2)
        ),
        Inst::AluRR { alu_op, rd, rs } => {
            format!("{} {},{}", alu_op.op_name(), fwreg(rd), freg(rs))
        }
        Inst::AluRRSimm16 {
            alu_op,
            rd,
            ra,
            simm,
        } => {
            if alu_op == &AluOPRRI::Addi && reg_num(*ra) == 0 {
                format!("li {},{:+}", fwreg(rd), simm.as_i16())
            } else if alu_op == &AluOPRRI::Addis && reg_num(*ra) == 0 {
                format!("lis {},{}", fwreg(rd), simm.as_i16())
            } else {
                format!(
                    "{} {},{},{:+}",
                    alu_op.op_name(),
                    fwreg(rd),
                    freg(ra),
                    simm.as_i16()
                )
            }
        }
        Inst::Li32 { rd, imm } => format!("li32 {},{:#x}", fwreg(rd), imm),
        Inst::LoadConst64 { rd, imm } => format!("li64 {},{:#x}", fwreg(rd), imm),
        Inst::Load {
            op,
            rd,
            from,
            flags: _,
        } => format!("{} {},{}", op.op_name(), fwreg(rd), from),
        Inst::Store {
            op,
            src,
            to,
            flags: _,
        } => format!("{} {},{}", op.op_name(), freg(src), to),
        Inst::LoadAddr { rd, mem } => format!("load_addr {},{}", fwreg(rd), mem),
        Inst::Args { args } => {
            let mut s = "args".to_string();
            for ArgPair { vreg, preg } in args {
                let preg = reg_name(*preg);
                let def = reg_name(vreg.to_reg());
                s += &format!(" {def}={preg}");
            }
            s
        }
        Inst::Rets { rets } => {
            let mut s = "rets".to_string();
            for RetPair { vreg, preg } in rets {
                let preg = reg_name(*preg);
                let vreg = reg_name(*vreg);
                s += &format!(" {vreg}={preg}");
            }
            s
        }
        Inst::Ret {} => "ret".to_string(),
        Inst::Extend {
            rd,
            rn,
            signed,
            from_bits,
        } => {
            let op = match (*signed, from_bits) {
                (true, 8) => "sext8",
                (true, 16) => "sext16",
                (true, 32) => "sext32",
                (false, 8) => "zext8",
                (false, 16) => "zext16",
                (false, 32) => "zext32",
                _ => "extend?",
            };
            format!("{op} {},{}", fwreg(rd), freg(rn))
        }
        Inst::Call { info } => format!("call {}", info.dest.display(None)),
        Inst::CallInd { info } => format!("callind {}", freg(&info.dest)),
        Inst::Jmp { label } => format!("j {}", label.to_string()),
        Inst::CondBr {
            taken,
            not_taken,
            kind,
        } => format!(
            "cmp{},taken({}),not_taken({})##({} {} {})",
            if kind.is64 { "d" } else { "w" },
            taken.to_string(),
            not_taken.to_string(),
            freg(&kind.rs1),
            fmt_cond(kind.kind),
            freg(&kind.rs2),
        ),
        Inst::BrTable {
            index,
            default,
            targets,
            ..
        } => format!(
            "br_table {},default({}),targets({})",
            freg(index),
            default.to_string(),
            targets.len()
        ),
        Inst::TrapIf { cmp, trap_code } => format!(
            "trap_if {}##({} {} {})",
            trap_code,
            freg(&cmp.rs1),
            fmt_cond(cmp.kind),
            freg(&cmp.rs2)
        ),
        Inst::SetCC { rd, cmp } => format!(
            "setcc {},##({} {} {})",
            fwreg(rd),
            freg(&cmp.rs1),
            fmt_cond(cmp.kind),
            freg(&cmp.rs2)
        ),
        Inst::Udf { trap_code } => format!("udf##trap_code={trap_code}"),
        Inst::MovFromPReg { rd, rm } => format!("mv {},{:?}", fwreg(rd), rm),
        Inst::Mfspr { rd, spr } => format!("mfspr {},{}", fwreg(rd), spr),
        Inst::Mtspr { rs, spr } => format!("mtspr {},{}", freg(rs), spr),
        Inst::FpuRRR {
            alu_op,
            rd,
            rs1,
            rs2,
        } => {
            if *alu_op == FpuOPRRR::Fmul {
                format!("fmul {},{},{}", fwreg(rd), freg(rs1), freg(rs2))
            } else {
                format!(
                    "{} {},{},{}",
                    alu_op.op_name(),
                    fwreg(rd),
                    freg(rs1),
                    freg(rs2)
                )
            }
        }
        Inst::FpuRRRR {
            alu_op,
            rd,
            ra,
            rb,
            rc,
        } => format!(
            "{} {},{},{},{}",
            alu_op.op_name(),
            fwreg(rd),
            freg(ra),
            freg(rc),
            freg(rb)
        ),
        Inst::FpuRR { alu_op, rd, rs } => {
            format!("{} {},{}", alu_op.op_name(), fwreg(rd), freg(rs))
        }
        Inst::FCmpSetCC { rd, cc, a, b } => format!(
            "fcmp_setcc {},##({} {} {})",
            fwreg(rd),
            freg(a),
            format!("{cc:?}"),
            freg(b)
        ),
        Inst::MoveGprToFpr { rd, rs } => format!("mvbits.{{gpr->fpr}} {},{}", fwreg(rd), freg(rs)),
        Inst::MoveGprToFpr32 { rd, rs } => {
            format!("mvbits32.{{gpr->fpr}} {},{}", fwreg(rd), freg(rs))
        }
        Inst::MoveFprBitsToGpr32 { rd, rs } => {
            format!("mvbits32.{{fpr->gpr}} {},{}", fwreg(rd), freg(rs))
        }
        Inst::MoveFprToGprI32 { rd, rs } => {
            format!("mvbits32hi.{{fpr->gpr}} {},{}", fwreg(rd), freg(rs))
        }
        Inst::PromoteF32ToF64 { rd, rs } => format!("promote {},{}", fwreg(rd), freg(rs)),
        Inst::Clz { rd, rn, is64 } => {
            format!(
                "{} {},{}",
                if *is64 { "cntlzd" } else { "cntlzw" },
                fwreg(rd),
                freg(rn)
            )
        }
        Inst::Rlwinm { rd, rn, sh, mb, me } => {
            format!("rlwinm {},{},{},{},{}", fwreg(rd), freg(rn), sh, mb, me)
        }
        Inst::Rldicl { rd, rn, sh, mb } => {
            format!("rldicl {},{},{},{}", fwreg(rd), freg(rn), sh, mb)
        }
        Inst::LoadExtName { rd, name, offset } => {
            format!("lis {},[sym@ha{:+}]; ori {},{},[sym@l{:+}]", fwreg(rd), offset, fwreg(rd), fwreg(rd), offset)
        }
        Inst::VecALU { op, rd, ra, rb } => {
            format!(
                "{} {},{},{}",
                op.op_name(),
                reg_name(Reg::from(rd.to_reg())),
                reg_name(Reg::from(*ra)),
                reg_name(Reg::from(*rb))
            )
        }
        Inst::VecBitSel { rd, ra, rb, rc } => {
            format!(
                "vsel {},{},{},{}",
                reg_name(Reg::from(rd.to_reg())),
                reg_name(Reg::from(*ra)),
                reg_name(Reg::from(*rb)),
                reg_name(Reg::from(*rc))
            )
        }
        Inst::VecLoad { rd, mem, .. } => {
            format!("lvx {},{}", reg_name(Reg::from(rd.to_reg())), mem)
        }
        Inst::VecStore { rs, mem, .. } => {
            format!("stvx {},{}", reg_name(Reg::from(*rs)), mem)
        }
        Inst::Fence => "sync".to_string(),
        Inst::DebugTrap => "debugtrap".to_string(),
        Inst::AtomicLoad { op, rd, mem, .. } => {
            format!("atomic.{} {},{}", op.op_name(), fwreg(rd), mem)
        }
        Inst::AtomicStore { op, src, mem, .. } => {
            format!("atomic.{} {},{}", op.op_name(), freg(src), mem)
        }
        Inst::AtomicRmw { op, rd, mem, rs, .. } => {
            format!("atomic.rmw.{:?} {},{},{}", op, fwreg(rd), freg(rs), mem)
        }
        Inst::AtomicCas { rd, mem, expected, replacement, .. } => {
            format!(
                "atomic.cas {},{}({},{})",
                fwreg(rd),
                mem,
                freg(expected),
                freg(replacement)
            )
        }
        Inst::MoveFprToGpr { rd, rs } => format!("mvbits.{{fpr->gpr}} {},{}", fwreg(rd), freg(rs)),
        Inst::Unwind { inst } => format!("unwind {inst:?}"),
        Inst::DummyUse { reg } => format!("dummy_use {}", freg(reg)),
    }
}
