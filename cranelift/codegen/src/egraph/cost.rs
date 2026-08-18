//! Cost functions for egraph representation.

use crate::ir::Opcode;
use crate::loop_analysis::LoopLevel;

/// The multiplier applied to a cost for each loop-nest level of the
/// block in which it is computed.
const LOOP_NEST_LEVEL_SCALE: u32 = 1024;

/// The maximum loop-nest level that we scale costs for. Because costs
/// are only 32 bits wide, we cap the scaling at two levels (i.e.,
/// multiply by ~2^20) as documented on the `Cost` type.
const MAX_LOOP_NEST_LEVEL: usize = 2;

/// A cost of computing some value in the program.
///
/// Costs are measured in an arbitrary union that we represent in a
/// `u32`. The ordering is meant to be meaningful, but the value of a
/// single unit is arbitrary (and "not to scale"). We use a collection
/// of heuristics to try to make this approximation at least usable.
///
/// We start by defining costs for each opcode (see `pure_op_cost`
/// below). The cost of computing some value, initially, is the cost
/// of its opcode, plus the cost of computing its inputs.
///
/// We then adjust the cost according to loop nests: for each
/// loop-nest level, we multiply by 1024. Because we only have 32
/// bits, we limit this scaling to a loop-level of two (i.e., multiply
/// by 2^20 ~= 1M).
///
/// Arithmetic on costs is always saturating: we don't want to wrap
/// around and return to a tiny cost when adding the costs of two very
/// expensive operations. It is better to approximate and lose some
/// precision than to lose the ordering by wrapping.
///
/// Finally, we reserve the highest value, `u32::MAX`, as a sentinel
/// that means "infinite". This is separate from the finite costs and
/// not reachable by doing arithmetic on them (even when overflowing)
/// -- we saturate just *below* infinity. (This is done by the
/// `finite()` method.) An infinite cost is used to represent a value
/// that cannot be computed, or otherwise serve as a sentinel when
/// performing search for the lowest-cost representation of a value.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Cost(u32);

impl core::fmt::Debug for Cost {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if *self == Cost::infinity() {
            write!(f, "Cost::Infinite")
        } else {
            f.debug_tuple("Cost::Finite").field(&self.cost()).finish()
        }
    }
}

impl Cost {
    pub(crate) fn infinity() -> Cost {
        // 2^32 - 1 is, uh, pretty close to infinite... (we use `Cost`
        // only for heuristics and always saturate so this suffices!)
        Cost(u32::MAX)
    }

    pub(crate) fn zero() -> Cost {
        Cost(0)
    }

    /// Construct a new `Cost`.
    fn new(cost: u32) -> Cost {
        Cost(cost)
    }

    fn cost(&self) -> u32 {
        self.0
    }

    /// Return the cost of an opcode.
    fn of_opcode(op: Opcode) -> Cost {
        match op {
            // Constants.
            Opcode::Iconst | Opcode::F32const | Opcode::F64const => Cost::new(1),

            // Extends/reduces.
            Opcode::Uextend
            | Opcode::Sextend
            | Opcode::Ireduce
            | Opcode::Iconcat
            | Opcode::Isplit => Cost::new(1),

            // "Simple" arithmetic.
            Opcode::Iadd
            | Opcode::Isub
            | Opcode::Band
            | Opcode::Bor
            | Opcode::Bxor
            | Opcode::Bnot
            | Opcode::Ishl
            | Opcode::Ushr
            | Opcode::Sshr => Cost::new(3),

            // "Expensive" arithmetic.
            Opcode::Imul => Cost::new(10),

            // Everything else.
            _ => {
                // By default, be slightly more expensive than "simple"
                // arithmetic.
                let mut c = Cost::new(4);

                // And then get more expensive as the opcode does more side
                // effects.
                if op.can_trap() || op.other_side_effects() {
                    c = c + Cost::new(10);
                }
                if op.can_load() {
                    c = c + Cost::new(20);
                }
                if op.can_store() {
                    c = c + Cost::new(50);
                }

                c
            }
        }
    }

    /// Compute the cost of the operation and its given operands.
    ///
    /// Caller is responsible for checking that the opcode came from an instruction
    /// that satisfies `inst_predicates::is_pure_for_egraph()`.
    pub(crate) fn of_pure_op(op: Opcode, operand_costs: impl IntoIterator<Item = Self>) -> Self {
        let c = Self::of_opcode(op) + operand_costs.into_iter().sum();
        Cost::new(c.cost())
    }

    /// Compute the cost of an operation in the side-effectful skeleton.
    pub(crate) fn of_skeleton_op(op: Opcode, arity: usize) -> Self {
        Cost::of_opcode(op) + Cost::new(u32::try_from(arity).unwrap())
    }

    /// Scale a cost according to the loop-nest level of the block in
    /// which the value is computed: for each loop-nest level, multiply
    /// by `LOOP_NEST_LEVEL_SCALE`, capped at `MAX_LOOP_NEST_LEVEL`
    /// (see the module-level documentation). This makes representations
    /// that can be hoisted out of loops (i.e., whose operands are all
    /// loop-invariant, and which therefore carry a lower loop-nest
    /// level) strictly cheaper than equivalent representations that must
    /// be recomputed on every iteration.
    pub(crate) fn scale_by_loop_level(self, loop_level: LoopLevel) -> Self {
        let factor = match core::cmp::min(loop_level.level(), MAX_LOOP_NEST_LEVEL) {
            0 => 1,
            1 => LOOP_NEST_LEVEL_SCALE,
            _ => LOOP_NEST_LEVEL_SCALE * LOOP_NEST_LEVEL_SCALE,
        };
        Cost::new(self.cost().saturating_mul(factor))
    }
}

impl core::iter::Sum<Cost> for Cost {
    fn sum<I: Iterator<Item = Cost>>(iter: I) -> Self {
        iter.fold(Self::zero(), |a, b| a + b)
    }
}

impl core::default::Default for Cost {
    fn default() -> Cost {
        Cost::zero()
    }
}

impl core::ops::Add<Cost> for Cost {
    type Output = Cost;

    fn add(self, other: Cost) -> Cost {
        Cost::new(self.cost().saturating_add(other.cost()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_cost() {
        let a = Cost::new(5);
        let b = Cost::new(37);
        assert_eq!(a + b, Cost::new(42));
        assert_eq!(b + a, Cost::new(42));
    }

    #[test]
    fn add_infinity() {
        let a = Cost::new(5);
        let b = Cost::infinity();
        assert_eq!(a + b, Cost::infinity());
        assert_eq!(b + a, Cost::infinity());
    }

    #[test]
    fn op_cost_saturates_to_infinity() {
        let a = Cost::new(u32::MAX - 10);
        let b = Cost::new(11);
        assert_eq!(a + b, Cost::infinity());
        assert_eq!(b + a, Cost::infinity());
    }

    #[test]
    fn loop_nest_scaling() {
        // At loop-nest level 0 (no loop), the cost is unchanged.
        assert_eq!(
            Cost::new(42).scale_by_loop_level(LoopLevel::root()),
            Cost::new(42)
        );

        // Each loop-nest level multiplies by 1024.
        assert_eq!(
            Cost::new(3).scale_by_loop_level(LoopLevel::clamped(1)),
            Cost::new(3 * 1024)
        );

        // Scaling is capped at loop-nest level two (~2^20).
        assert_eq!(
            Cost::new(3).scale_by_loop_level(LoopLevel::clamped(2)),
            Cost::new(3 * 1024 * 1024)
        );
        assert_eq!(
            Cost::new(3).scale_by_loop_level(LoopLevel::clamped(5)),
            Cost::new(3 * 1024 * 1024)
        );

        // Scaling saturates to infinity instead of wrapping.
        assert_eq!(
            Cost::new(u32::MAX - 10).scale_by_loop_level(LoopLevel::clamped(1)),
            Cost::infinity()
        );
    }
}
