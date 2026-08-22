//! ppu64 ISA definitions: instruction arguments and addressing modes.

use super::*;

/// A macro for defining a newtype of `Reg` that enforces some invariant about
/// the wrapped `Reg` (such as that it is of a particular register class).
macro_rules! newtype_of_reg {
    (
        $newtype_reg:ident,
        $newtype_writable_reg:ident,
        |$check_reg:ident| $check:expr
    ) => {
        /// A newtype wrapper around `Reg`.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $newtype_reg(Reg);

        impl PartialEq<Reg> for $newtype_reg {
            fn eq(&self, other: &Reg) -> bool {
                self.0 == *other
            }
        }

        impl From<$newtype_reg> for Reg {
            fn from(r: $newtype_reg) -> Self {
                r.0
            }
        }

        impl $newtype_reg {
            /// Create this newtype from the given register, or return `None` if the register
            /// is not a valid instance of this newtype.
            pub fn new($check_reg: Reg) -> Option<Self> {
                if $check { Some(Self($check_reg)) } else { None }
            }

            /// Get this newtype's underlying `Reg`.
            pub fn to_reg(self) -> Reg {
                self.0
            }
        }

        impl core::ops::Deref for $newtype_reg {
            type Target = Reg;

            fn deref(&self) -> &Reg {
                &self.0
            }
        }

        /// Writable Reg.
        pub type $newtype_writable_reg = Writable<$newtype_reg>;
    };
}

// Newtypes for register classes.
newtype_of_reg!(XReg, WritableXReg, |reg| reg.class() == RegClass::Int);
newtype_of_reg!(FReg, WritableFReg, |reg| reg.class() == RegClass::Float);

/// An addressing mode specified for a load/store operation.
///
/// The PPU's D-form allows a signed 16-bit offset from a register; larger
/// offsets must be materialized into the scratch register at emission time.
#[derive(Clone, Debug, Copy)]
pub enum AMode {
    /// Arbitrary offset from a register.
    RegOffset(Reg, i64),
    /// Offset from the stack pointer.
    SPOffset(i64),
    /// Offset from the frame pointer.
    FPOffset(i64),
    /// Offset into the slot area of the stack, which lies just above the
    /// outgoing argument area that's set up by the function prologue.
    SlotOffset(i64),
    /// A reference to a constant which is placed outside of the function's
    /// body, typically at the end.
    Const(VCodeConstant),
}

impl AMode {
    /// Add the registers referenced by this AMode to `collector`.
    pub(crate) fn get_operands(&mut self, collector: &mut impl OperandVisitor) {
        match self {
            AMode::RegOffset(reg, ..) => collector.reg_use(reg),
            // Registers used in these modes aren't allocatable.
            AMode::SPOffset(..)
            | AMode::FPOffset(..)
            | AMode::SlotOffset(..)
            | AMode::Const(..) => {}
        }
    }
}

impl Display for AMode {
    fn fmt(&self, f: &mut Formatter) -> core::fmt::Result {
        match self {
            AMode::RegOffset(reg, offset) => write!(f, "{offset}({})", reg_name(*reg)),
            AMode::SPOffset(off) => write!(f, "{off}(%sp)"),
            AMode::FPOffset(off) => write!(f, "{off}(%fp)"),
            AMode::SlotOffset(off) => write!(f, "slot{off}"),
            AMode::Const(c) => write!(f, "const[{c:?}]"),
        }
    }
}

impl From<StackAMode> for AMode {
    fn from(stack: StackAMode) -> AMode {
        match stack {
            StackAMode::IncomingArg(offset, stack_args_size) => {
                AMode::SPOffset(i64::from(stack_args_size) - i64::from(offset))
            }
            StackAMode::OutgoingArg(offset) => AMode::SPOffset(i64::from(offset)),
            StackAMode::Slot(offset) => AMode::SlotOffset(i64::from(offset)),
        }
    }
}
