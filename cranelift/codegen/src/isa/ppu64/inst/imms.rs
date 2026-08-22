//! ppu64 ISA definitions: immediate constants.

use core::fmt::{Debug, Display, Formatter, Result};

/// A signed 16-bit immediate, as found in the D-form `SI` field.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SImm16 {
    bits: u16,
}

impl SImm16 {
    pub fn maybe_from_i64(val: i64) -> Option<SImm16> {
        if val >= -32768 && val <= 32767 {
            Some(SImm16 {
                bits: (val as u16) & 0xffff,
            })
        } else {
            None
        }
    }

    pub fn maybe_from_u64(val: u64) -> Option<SImm16> {
        Self::maybe_from_i64(val as i64)
    }

    pub fn from_i16(bits: i16) -> SImm16 {
        SImm16 { bits: bits as u16 }
    }

    pub fn as_i16(self) -> i16 {
        self.bits as i16
    }

    pub fn bits(self) -> u32 {
        u32::from(self.bits)
    }

    /// Sign-extend to 64 bits.
    pub fn as_i64(self) -> i64 {
        i64::from(self.as_i16())
    }
}

impl Display for SImm16 {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{:+}", self.as_i16())
    }
}

/// An unsigned 16-bit immediate, as found in the D-form logical-op `UI`
/// field (`ori`, `oris`, `xori`, `xoris`, `andi.`, `andis.`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UImm16 {
    bits: u16,
}

impl UImm16 {
    pub fn maybe_from_u64(val: u64) -> Option<UImm16> {
        if val <= 0xffff {
            Some(UImm16 { bits: val as u16 })
        } else {
            None
        }
    }

    pub fn from_u16(bits: u16) -> UImm16 {
        UImm16 { bits }
    }

    pub fn bits(self) -> u32 {
        u32::from(self.bits)
    }
}

impl Display for UImm16 {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}", self.bits)
    }
}
