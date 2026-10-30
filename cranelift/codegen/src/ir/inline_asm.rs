//! Inline-assembly templates referenced by [`AsmRef`].

use alloc::boxed::Box;
use alloc::string::String;
#[cfg(feature = "enable-serde")]
use serde::{Deserialize, Serialize};

/// The contents of an inline-assembly statement, in the style of GCC's
/// extended `asm` (but with the template written for the target assembler
/// dialect rather than GCC's own operand formatting).
///
/// Operand placeholders in the template use GCC numbering: `%N` in AT&T
/// syntax or `$N` in Intel syntax, where operand 0..n are the outputs
/// (in order) and n..n+m are the inputs. `%%` escapes a literal register
/// prefix in AT&T templates.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "enable-serde", derive(Serialize, Deserialize))]
pub struct InlineAsmContents {
    /// The assembly template, one or more instructions.
    pub template: Box<str>,

    /// GCC-style constraint strings, outputs first then inputs, matching
    /// the template's operand numbering: `"=r"`, `"+a"`, `"ri"`, ...
    pub operands: Box<[Box<str>]>,

    /// Explicitly clobbered registers or resources: register names like
    /// `"rax"`, or the pseudo-resources `"cc"` and `"memory"`.
    pub clobbers: Box<[Box<str>]>,

    /// AT&T (`%%reg`, `src, dst`) when true; Intel (`reg`, `dst, src`)
    /// when false.
    pub att_syntax: bool,
}

impl InlineAsmContents {
    /// Create a new inline-assembly statement with no operands.
    pub fn new(template: impl Into<String>, att_syntax: bool) -> Self {
        Self {
            template: template.into().into_boxed_str(),
            operands: Box::new([]),
            clobbers: Box::new([]),
            att_syntax,
        }
    }

    /// Add an operand constraint string (outputs first, then inputs).
    pub fn operand(mut self, constraint: impl Into<String>) -> Self {
        let mut ops = self.operands.into_vec();
        ops.push(constraint.into().into_boxed_str());
        self.operands = ops.into_boxed_slice();
        self
    }

    /// Add an explicitly clobbered register or resource.
    pub fn clobber(mut self, clobber: impl Into<String>) -> Self {
        let mut clobbers = self.clobbers.into_vec();
        clobbers.push(clobber.into().into_boxed_str());
        self.clobbers = clobbers.into_boxed_slice();
        self
    }

    /// Number of output operands: constraints starting with `=` (write-only)
    /// or `+` (read-write).
    pub fn num_outputs(&self) -> usize {
        self.operands
            .iter()
            .filter(|c| c.starts_with('=') || c.starts_with('+'))
            .count()
    }

    /// Number of input *values* consumed: plain inputs plus the read
    /// sides of `+` (inout) operands, in template order.
    pub fn num_inputs(&self) -> usize {
        self.operands
            .iter()
            .filter(|c| !c.starts_with('='))
            .count()
    }
}
