//! Known-bits analysis for scalar integer values.
//!
//! For each scalar integer SSA value we track, in the aggregate, which bits
//! are known to be zero and which bits are known to be one. This information
//! is computed in a single forward pass over the CFG (in dominator-tree order,
//! as part of the egraph traversal) by propagating the known bits of each
//! instruction's operands through the instruction's semantics.
//!
//! This is a classic "bits-and-branches" style analysis that lets us
//! constant-fold and otherwise simplify expressions that no single
//! constant-propagation pass would be able to see through: for example
//! `band x, mask` can be turned into a constant when `x`'s known bits cover
//! the whole mask, and comparisons against zero can be resolved when any bit
//! of `x` is known to be one.

use crate::ir::{Function, Inst, InstructionData, Opcode, Type, Value};
use cranelift_entity::SecondaryMap;

/// The bits of a scalar integer value that are known.
///
/// `known_zero` is a bitmask of the bits that are known to be zero, and
/// `known_one` is a bitmask of the bits that are known to be one. A bit that
/// is set in neither mask is unknown. The two masks never overlap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KnownBits {
    /// Bits known to be zero.
    pub known_zero: u128,
    /// Bits known to be one.
    pub known_one: u128,
}

impl KnownBits {
    /// No information about any bit.
    pub const fn all_unknown() -> Self {
        Self {
            known_zero: 0,
            known_one: 0,
        }
    }

    /// A value known to be exactly `val` (masked to `ty`'s width).
    pub fn from_constant(val: u128, ty: Type) -> Self {
        let mask = ty_mask(ty);
        Self {
            known_zero: mask & !val,
            known_one: mask & val,
        }
    }

    /// The union of the information in `self` and `other`: a bit is known in
    /// the result only if it is known (to the same value) in both inputs.
    #[inline]
    pub fn meet(self, other: Self) -> Self {
        Self {
            known_zero: self.known_zero & other.known_zero,
            known_one: self.known_one & other.known_one,
        }
    }
}

/// The bitmask of `ty`'s width, i.e. all `ty.bits()` low bits set.
pub fn ty_mask(ty: Type) -> u128 {
    let bits = ty.bits();
    if bits >= 128 {
        u128::MAX
    } else {
        (1u128 << bits) - 1
    }
}

/// A bitmask with `bits` low bits set.
fn low_mask(bits: usize) -> u128 {
    if bits >= 128 {
        u128::MAX
    } else {
        (1u128 << bits) - 1
    }
}

/// Compute the known bits of the result of `inst`, given the known bits of
/// every value defined so far in the traversal (`known_bits`).
///
/// The known bits of `inst`'s arguments must already be in `known_bits`, so
/// this must be called after `inst`'s arguments have been inserted into the
/// egraph. The result is then written back into `known_bits[result]` by the
/// caller.
///
/// For scalar integer types only; returns `all_unknown` for anything else.
pub fn compute_known_bits(
    func: &Function,
    inst: Inst,
    known_bits: &SecondaryMap<Value, KnownBits>,
) -> KnownBits {
    let inst_data = &func.dfg.insts[inst];
    let ty = func.dfg.ctrl_typevar(inst);
    if !ty.is_int() {
        return KnownBits::all_unknown();
    }
    let ty_bits = ty.bits() as usize;
    let mask = ty_mask(ty);

    let arg = |i: usize| -> KnownBits {
        let v = func.dfg.inst_values(inst).nth(i).unwrap();
        known_bits[v]
    };

    match inst_data {
        InstructionData::UnaryImm {
            opcode: Opcode::Iconst,
            imm,
        } => KnownBits::from_constant(imm.bits() as u128, ty),

        InstructionData::Unary {
            opcode: Opcode::Uextend,
            arg: x,
        } => {
            let x_ty = func.dfg.value_type(*x);
            if !x_ty.is_int() {
                return KnownBits::all_unknown();
            }
            let x_bits = x_ty.bits() as usize;
            let x_mask = low_mask(x_bits);
            let x = known_bits[*x];
            // The low `x_bits` bits come from `x`; the high bits are zero.
            KnownBits {
                known_zero: (x.known_zero & x_mask) | (!x_mask & mask),
                known_one: x.known_one & x_mask,
            }
        }

        InstructionData::Unary {
            opcode: Opcode::Sextend,
            arg: x,
        } => {
            let x_ty = func.dfg.value_type(*x);
            if !x_ty.is_int() {
                return KnownBits::all_unknown();
            }
            let x_bits = x_ty.bits() as usize;
            let x_mask = low_mask(x_bits);
            let x = known_bits[*x];
            let sign_bit = 1u128 << (x_bits - 1);
            let high = !x_mask & mask;
            // If the sign bit of `x` is known, the high bits replicate it.
            let known_zero = if x.known_zero & sign_bit != 0 {
                x.known_zero | high
            } else {
                x.known_zero & x_mask
            };
            let known_one = if x.known_one & sign_bit != 0 {
                x.known_one | high
            } else {
                x.known_one & x_mask
            };
            KnownBits {
                known_zero,
                known_one,
            }
        }

        InstructionData::Unary {
            opcode: Opcode::Ireduce,
            ..
        } => KnownBits {
            known_zero: arg(0).known_zero & mask,
            known_one: arg(0).known_one & mask,
        },

        InstructionData::Unary {
            opcode: Opcode::Bnot,
            ..
        } => KnownBits {
            known_zero: arg(0).known_one & mask,
            known_one: arg(0).known_zero & mask,
        },

        InstructionData::Binary {
            opcode: Opcode::Band,
            ..
        } => {
            let x = arg(0);
            let y = arg(1);
            KnownBits {
                known_zero: (x.known_zero | y.known_zero) & mask,
                known_one: (x.known_one & y.known_one) & mask,
            }
        }

        InstructionData::Binary {
            opcode: Opcode::Bor,
            ..
        } => {
            let x = arg(0);
            let y = arg(1);
            KnownBits {
                known_zero: (x.known_zero & y.known_zero) & mask,
                known_one: (x.known_one | y.known_one) & mask,
            }
        }

        InstructionData::Binary {
            opcode: Opcode::Bxor,
            ..
        } => {
            let x = arg(0);
            let y = arg(1);
            KnownBits {
                known_zero: ((x.known_zero & y.known_zero) | (x.known_one & y.known_one)) & mask,
                known_one: ((x.known_zero & y.known_one) | (x.known_one & y.known_zero)) & mask,
            }
        }

        InstructionData::Binary {
            opcode: Opcode::Ishl,
            ..
        } => shift_known_bits(func, inst, arg(0), ty_bits, mask, Shift::Left),

        InstructionData::Binary {
            opcode: Opcode::Ushr,
            ..
        } => shift_known_bits(func, inst, arg(0), ty_bits, mask, Shift::LogicalRight),

        InstructionData::Binary {
            opcode: Opcode::Sshr,
            ..
        } => shift_known_bits(func, inst, arg(0), ty_bits, mask, Shift::ArithmeticRight),

        InstructionData::IntCompare {
            opcode: Opcode::Icmp,
            ..
        } => KnownBits {
            // The result is either zero or one: only the high bits are known
            // to be zero; bit 0 is unknown.
            known_zero: mask & !1,
            known_one: 0,
        },

        _ => KnownBits::all_unknown(),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Shift {
    Left,
    LogicalRight,
    ArithmeticRight,
}

/// The known bits of `x << s`, `x >>u s`, or `x >>s s` where `s` is given by
/// the second argument of `inst` (which must be an `iconst`, otherwise all
/// known bits are lost).
fn shift_known_bits(
    func: &Function,
    inst: Inst,
    x: KnownBits,
    ty_bits: usize,
    mask: u128,
    shift: Shift,
) -> KnownBits {
    let Some(raw_s) = const_shift_amount(func, inst, 1) else {
        return KnownBits::all_unknown();
    };
    // Shift amounts are masked to the size of `x` (the shift is taken
    // modulo the bitwidth).
    let s = (raw_s as usize) % ty_bits;

    match shift {
        Shift::Left => KnownBits {
            known_zero: ((x.known_zero << s) | low_mask(s)) & mask,
            known_one: (x.known_one << s) & mask,
        },
        Shift::LogicalRight => KnownBits {
            known_zero: ((x.known_zero >> s) | (low_mask(ty_bits) & !low_mask(ty_bits - s))) & mask,
            known_one: (x.known_one >> s) & mask,
        },
        Shift::ArithmeticRight => {
            let sign_bit = 1u128 << (ty_bits - 1);
            let high_bits = low_mask(ty_bits) & !low_mask(ty_bits - s);
            let mut known_zero = (x.known_zero >> s) & low_mask(ty_bits - s);
            let mut known_one = (x.known_one >> s) & low_mask(ty_bits - s);
            if x.known_zero & sign_bit != 0 {
                known_zero |= high_bits;
            } else if x.known_one & sign_bit != 0 {
                known_one |= high_bits;
            }
            KnownBits {
                known_zero: known_zero & mask,
                known_one: known_one & mask,
            }
        }
    }
}

/// If the argument at index `idx` of `inst` is an `iconst`, return its value
/// as a `u64`; otherwise `None`.
fn const_shift_amount(func: &Function, inst: Inst, idx: usize) -> Option<u64> {
    let value = func.dfg.inst_values(inst).nth(idx)?;
    let value_inst = func.dfg.value_def(value).inst()?;
    if let InstructionData::UnaryImm {
        opcode: Opcode::Iconst,
        imm,
    } = &func.dfg.insts[value_inst]
    {
        Some(imm.bits() as u64)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::types;

    #[test]
    fn masks() {
        assert_eq!(ty_mask(types::I8), 0xff);
        assert_eq!(ty_mask(types::I16), 0xffff);
        assert_eq!(ty_mask(types::I32), 0xffff_ffff);
        assert_eq!(ty_mask(types::I64), u64::MAX as u128);
        assert_eq!(ty_mask(types::I128), u128::MAX);
    }

    #[test]
    fn constant() {
        let kb = KnownBits::from_constant(0x34, types::I8);
        assert_eq!(kb.known_zero, 0xff & !0x34);
        assert_eq!(kb.known_one, 0x34);
    }

    #[test]
    fn meet_is_intersection() {
        let a = KnownBits {
            known_zero: 0xf0,
            known_one: 0x03,
        };
        let b = KnownBits {
            known_zero: 0x0f,
            known_one: 0x02,
        };
        let m = a.meet(b);
        assert_eq!(m.known_zero, 0x00);
        assert_eq!(m.known_one, 0x02);
    }
}
