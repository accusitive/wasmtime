//! GCC-style inline-asm constraint parsing for x64 and the resolved
//! machine-level form carried by `MInst::InlineAsm`.

use crate::ir::InlineAsmContents;
use crate::isa::x64::inst::regs;
use crate::machinst::{Reg, Writable};
use regalloc2::{PReg, PRegSet, RegClass};
use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use cranelift_assembler_x64 as asm;
use smallvec::SmallVec;

/// How an operand number in the template resolves at emission time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AsmOperandRef {
    /// Output operand: the register of def `n`.
    Def(u16),
    /// Input operand: the register of use `n`.
    Use(u16),
    /// Memory operand: the memory at the address held in use `n`.
    MemUse(u16),
    /// Immediate operand: the constant itself.
    Const(i64),
}

/// The x64-resolved form of an [`InlineAsmContents`]: template text plus
/// the mapping from GCC operand numbers to machine operands, and the
/// clobber set.
#[derive(Clone, Debug)]
pub struct InlineAsmTemplate {
    /// Raw template text, with `%N` operand placeholders and `%%` escapes
    /// still in place.
    pub text: Box<str>,
    /// True for AT&T (`%%reg`, `src, dst`) syntax; false for Intel.
    pub att_syntax: bool,
    /// GCC operand number (outputs first, then inputs) to machine operand.
    pub refs: Box<[AsmOperandRef]>,
    /// GPRs clobbered by the assembly. Fixed-constraint registers are not
    /// included; they are tracked as fixed operands.
    pub clobbers: PRegSet,
    /// True if `"cc"` was listed: x86 flags are destroyed. Cranelift does
    /// not allocate flags, so this is advisory only.
    pub clobbers_flags: bool,
    /// True if `"memory"` was listed: arbitrary memory may be read or
    /// written.
    pub clobbers_memory: bool,
}

/// The kind of operand a constraint selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstraintKind {
    /// A general-purpose register, with a register-selection refinement.
    Reg(RegConstraint),
    /// An integer constant (`"i"`/`"n"`), substituted into the template.
    Imm,
    /// Memory at the address held in the input value (`"m"`); the
    /// template sees `disp(%reg)` (AT&T) or `[reg]` (Intel).
    Mem,
}

/// Which register an `Reg`-kind operand may occupy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegConstraint {
    /// Any general-purpose register (`"r"` and class subsets widened to
    /// it).
    Any,
    /// A specific register (`"a"`, `"b"`, `"c"`, `"d"`, `"S"`, `"D"`).
    Fixed(Reg),
    /// The first `limit` GPRs by hardware encoding: `"Q"` (limit 4 =
    /// rax/rcx/rdx/rbx, the high-byte-capable registers) and `"R"`
    /// (limit 8 = the legacy registers).
    Limit(u16),
}

/// A parsed GCC constraint for one operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Constraint {
    /// `=` (output), `+` (inout) or nothing (input).
    pub is_output: bool,
    /// `+`: the operand is both read and written; its input value is
    /// copied into the output register before the assembly runs.
    pub inout: bool,
    /// `&`: the output is written before all inputs are consumed
    /// (early clobber). Accepted for compatibility; only meaningful when
    /// matching constraints (`"0"`-style ties) are in use, which are not
    /// supported yet.
    pub early_clobber: bool,
    /// What kind of operand this constraint selects.
    pub kind: ConstraintKind,
}

impl Constraint {
    fn parse(text: &str) -> Constraint {
        let mut c = Constraint {
            is_output: false,
            inout: false,
            early_clobber: false,
            kind: ConstraintKind::Reg(RegConstraint::Any),
        };

        for ch in text.chars() {
            match ch {
                '=' => c.is_output = true,
                '+' => {
                    c.is_output = true;
                    c.inout = true;
                }
                '&' => c.early_clobber = true,
                'r' | 'q' | 'l' => {
                    // `r` and its class subsets: any GPR (`q`/`l` are
                    // widened; template writers must only refer to valid
                    // registers for the subset).
                }
                'a' => c.kind = ConstraintKind::Reg(RegConstraint::Fixed(regs::rax())),
                'b' => c.kind = ConstraintKind::Reg(RegConstraint::Fixed(regs::rbx())),
                'c' => c.kind = ConstraintKind::Reg(RegConstraint::Fixed(regs::rcx())),
                'd' => c.kind = ConstraintKind::Reg(RegConstraint::Fixed(regs::rdx())),
                'S' => c.kind = ConstraintKind::Reg(RegConstraint::Fixed(regs::rsi())),
                'D' => c.kind = ConstraintKind::Reg(RegConstraint::Fixed(regs::rdi())),
                'Q' => c.kind = ConstraintKind::Reg(RegConstraint::Limit(4)),
                'R' => c.kind = ConstraintKind::Reg(RegConstraint::Limit(8)),
                'x' | 'v' => {
                    // An xmm register (`"x"`). No new constraint variant:
                    // the regalloc2 class rides the vreg, so the value must
                    // be vector-typed — an integer value here would
                    // allocate a GPR and render wrong.
                }
                'i' | 'n' => c.kind = ConstraintKind::Imm,
                'm' | 'o' | 'V' => c.kind = ConstraintKind::Mem,
                _ => unsupported(text),
            }
        }
        if c.is_output && !matches!(c.kind, ConstraintKind::Reg(_)) {
            unsupported(text);
        }
        c
    }
}

fn unsupported(text: &str) -> ! {
    panic!("inline asm: unsupported constraint `{text}`")
}

/// One machine operand of the assembled instruction.
#[derive(Clone, Copy, Debug)]
pub struct AsmUse {
    /// The (virtual, later physical) register holding the input value.
    pub reg: Reg,
    /// The register-selection refinement from the constraint.
    pub constraint: RegConstraint,
}

/// One output register of the assembled instruction.
#[derive(Clone, Copy, Debug)]
pub struct AsmDef {
    /// The (virtual, later physical) register receiving the output value.
    pub reg: Writable<Reg>,
    /// The register-selection refinement from the constraint.
    pub constraint: RegConstraint,
    /// For `+` (inout) operands: the index of the use operand whose
    /// register this def must share (the regalloc2 `Reuse` constraint).
    /// The use is a fresh copy of the input value, so clobbering it is
    /// always safe.
    pub reuse: Option<u16>,
}

/// The machine-level operand lists of an `MInst::InlineAsm`.
#[derive(Clone, Debug)]
pub struct InlineAsmOperands {
    /// Input registers.
    pub uses: SmallVec<[AsmUse; 4]>,
    /// Output registers.
    pub defs: SmallVec<[AsmDef; 4]>,
}

/// Parse the constraint strings of an [`InlineAsmContents`] into the
/// machine-level form used by the x64 backend.
pub fn parse(contents: &InlineAsmContents) -> (InlineAsmTemplate, Vec<Constraint>) {
    let mut refs = Vec::with_capacity(contents.operands.len());
    let mut constraints = Vec::with_capacity(contents.operands.len());
    let mut clobbers = PRegSet::empty();
    let mut clobbers_flags = false;
    let mut clobbers_memory = false;

    let mut n_defs = 0u16;
    let mut n_uses = 0u16;
    for text in contents.operands.iter() {
        let constraint = Constraint::parse(text);
        match constraint.kind {
            ConstraintKind::Reg(_) => {
                if constraint.is_output {
                    refs.push(AsmOperandRef::Def(n_defs));
                    n_defs += 1;
                } else {
                    refs.push(AsmOperandRef::Use(n_uses));
                    n_uses += 1;
                }
            }
            ConstraintKind::Mem => {
                refs.push(AsmOperandRef::MemUse(n_uses));
                n_uses += 1;
            }
            ConstraintKind::Imm => refs.push(AsmOperandRef::Const(0)),
        }
        constraints.push(constraint);
    }

    for text in contents.clobbers.iter() {
        match text.as_ref() {
            "cc" => clobbers_flags = true,
            "memory" => clobbers_memory = true,
            name => clobbers.add(lookup_clobber_reg(name)),
        }
    }

    let template = InlineAsmTemplate {
        text: contents.template.clone(),
        att_syntax: contents.att_syntax,
        refs: refs.into_boxed_slice(),
        clobbers,
        clobbers_flags,
        clobbers_memory,
    };
    (template, constraints)
}

/// Resolve a clobber name (any width alias, e.g. `rax`, `eax`, `al`) to
/// its 64-bit GPR.
fn lookup_clobber_reg(name: &str) -> PReg {
    const NAMES: &[(&str, u8)] = &[
        ("rax", asm::gpr::enc::RAX),
        ("rbx", asm::gpr::enc::RBX),
        ("rcx", asm::gpr::enc::RCX),
        ("rdx", asm::gpr::enc::RDX),
        ("rsi", asm::gpr::enc::RSI),
        ("rdi", asm::gpr::enc::RDI),
        ("r8", asm::gpr::enc::R8),
        ("r9", asm::gpr::enc::R9),
        ("r10", asm::gpr::enc::R10),
        ("r11", asm::gpr::enc::R11),
        ("r12", asm::gpr::enc::R12),
        ("r13", asm::gpr::enc::R13),
        ("r14", asm::gpr::enc::R14),
        ("r15", asm::gpr::enc::R15),
    ];
    let name = name.trim_start_matches('%');
    if let Some(suffix) = name.strip_prefix("xmm") {
        if let Ok(enc) = suffix.parse::<u8>() {
            if enc < 16 {
                return regs::fpr_preg(enc);
            }
        }
    }
    for (canonical, enc) in NAMES {
        if reg_matches(name, canonical) {
            return regs::gpr_preg(*enc);
        }
    }
    if name == "rsp" || name == "rbp" {
        panic!("inline asm: clobbering the stack pointer or frame pointer is not supported");
    }
    panic!("inline asm: unknown clobber register `{name}`")
}

/// True if `name` names the same register as `canonical` (e.g. `eax`
/// matches `rax`).
fn reg_matches(name: &str, canonical: &str) -> bool {
    if name == canonical {
        return true;
    }
    let Some(suffix) = canonical.strip_prefix('r') else {
        return false;
    };
    if suffix.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        // r8 family: r8d / r8w / r8b.
        return ["d", "w", "b"].iter().any(|s| name == format!("r{suffix}{s}"));
    }
    // Legacy family: rax -> ax / eax / al, plus ah..dh for a-d.
    let low = match suffix {
        "ax" => "al",
        "bx" => "bl",
        "cx" => "cl",
        "dx" => "dl",
        "si" => "sil",
        "di" => "dil",
        _ => return false,
    };
    let high = match suffix {
        "ax" | "bx" | "cx" | "dx" => Some(format!("{}h", &suffix[..1])),
        _ => None,
    };
    name == suffix || name == format!("e{suffix}") || name == low || high.as_deref() == Some(name)
}

#[cfg(test)]
mod tests {
    use crate::ir::InlineAsmContents;

    #[test]
    fn clobbers_accept_width_aliases() {
        for name in [
            "rax", "eax", "ax", "al", "ah", "dil", "r8", "r8d", "r8b", "r15w",
        ] {
            let contents = InlineAsmContents::new("nop", true).clobber(name);
            let _ = super::parse(&contents);
        }
    }

    #[test]
    fn clobbers_accept_xmm() {
        for name in ["xmm0", "xmm3", "xmm15", "%xmm5"] {
            let contents = InlineAsmContents::new("nop", true).clobber(name);
            let _ = super::parse(&contents);
        }
        let contents = InlineAsmContents::new("nop", true).clobber("xmm2");
        let (template, _) = super::parse(&contents);
        assert!(template.clobbers.contains(super::regs::fpr_preg(2)));
    }

    #[test]
    fn reg_name_renders_xmm() {
        let preg = super::regs::fpr_preg(11);
        assert_eq!(super::reg_name(preg, true), "%xmm11");
        assert_eq!(super::reg_name(preg, false), "xmm11");
    }

    #[test]
    #[should_panic(expected = "unknown clobber register")]
    fn rejects_unknown_clobber() {
        let contents = InlineAsmContents::new("nop", true).clobber("qq");
        let _ = super::parse(&contents);
    }

    #[test]
    #[should_panic(expected = "stack pointer")]
    fn rejects_stack_pointer_clobber() {
        let contents = InlineAsmContents::new("nop", true).clobber("rsp");
        let _ = super::parse(&contents);
    }
}

/// The register-name text substituted for a `%N` placeholder: AT&T gets a
/// `%` prefix, Intel does not.
pub fn reg_name(preg: PReg, att_syntax: bool) -> String {
    reg_name_sized(preg, asm::gpr::Size::Quadword, att_syntax)
}

/// Like [`reg_name`], at the given register width (`%b`/`%w`/`%k`/`%q`
/// template modifiers).
pub fn reg_name_sized(preg: PReg, size: asm::gpr::Size, att_syntax: bool) -> String {
    if preg.class() == RegClass::Float {
        assert!(
            matches!(size, asm::gpr::Size::Quadword),
            "inline asm: width modifiers are not valid on xmm registers"
        );
        let name = asm::xmm::enc::to_string(preg.hw_enc() as u8);
        let name = name.trim_start_matches('%');
        return if att_syntax {
            format!("%{name}")
        } else {
            name.to_string()
        };
    }
    let name = asm::gpr::enc::to_string(preg.hw_enc() as u8, size);
    let name = name.trim_start_matches('%');
    if att_syntax {
        format!("%{name}")
    } else {
        name.to_string()
    }
}

/// The high-byte register name (`%h` modifier): `%ah`/`%ch`/`%dh`/`%bh`
/// — only the first four GPRs have high bytes.
pub fn reg_name_high_byte(preg: PReg, att_syntax: bool) -> String {
    const NAMES: [&str; 4] = ["ah", "ch", "dh", "bh"];
    let name = NAMES
        .get(preg.hw_enc())
        .copied()
        .unwrap_or_else(|| panic!("inline asm: %h requires a/b/c/d, not this register"));
    if att_syntax {
        format!("%{name}")
    } else {
        name.to_string()
    }
}

/// The boxed payload of `MInst::InlineAsm`: the resolved template plus the
/// machine operands, in the order regalloc visits them (all uses, then all
/// defs).
#[derive(Clone, Debug)]
pub struct InlineAsmData {
    /// The resolved template (operand mapping, clobbers).
    pub template: InlineAsmTemplate,
    /// Input registers, in use order.
    pub uses: SmallVec<[AsmUse; 4]>,
    /// Output registers, in def order.
    pub defs: SmallVec<[AsmDef; 4]>,
}

/// Expand the template (resolving `%N` operand placeholders, `%c`/`%b`/
/// `%w`/`%k`/`%q` modifiers and `%%` escapes), assemble it with the
/// vendored glasm encoder, and write the resulting machine-code bytes to
/// `sink`.
///
/// By the time this runs, regalloc has rewritten the operand registers in
/// `data` in place, so they are all physical registers.
pub fn emit(
    data: &InlineAsmData,
    sink: &mut crate::machinst::MachBuffer<crate::isa::x64::inst::Inst>,
) {
    use alloc::vec::Vec;

    let template = &data.template;


    let _ = ();

    let sources: Vec<Source> = template
        .refs
        .iter()
        .map(|operand_ref| match operand_ref {
            AsmOperandRef::Def(k) => {
                let reg = data.defs[*k as usize].reg.to_reg();
                Source::Reg(
                    reg.to_real_reg()
                        .expect("inline asm: operand not allocated")
                        .into(),
                )
            }
            AsmOperandRef::Use(k) | AsmOperandRef::MemUse(k) => {
                let reg = data.uses[*k as usize].reg;
                Source::Reg(
                    reg.to_real_reg()
                        .expect("inline asm: operand not allocated")
                        .into(),
                )
            }
            AsmOperandRef::Const(v) => Source::Const(*v),
        })
        .collect();

    // The CLIF reader does not process escapes in string literals;
    // templates written there use `\n`/`\t` for statement separation.
    let mut text: String = template
        .text
        .replace("\\n", "\n")
        .replace("\\t", "\t");

    // `%=`: a unique number per asm instantiation. The instruction's code
    // offset is unique within the function and deterministic across
    // compiles of the same function.
    if text.contains("%=") {
        let unique = sink.cur_offset();
        text = text.replace("%=", &unique.to_string());
    }

    let mut expanded = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '%' {
            expanded.push(ch);
            continue;
        }
        let next = chars.peek().copied();
        match next {
            Some('%') => {
                chars.next();
                expanded.push('%');
            }
            Some('c' | 'b' | 'w' | 'k' | 'q' | 'h' | 'n') => {
                // A modifier is only a modifier when an operand number
                // follows; `%rax` keeps its literal percent.
                let after = chars.clone().nth(1);
                if after.is_some_and(|n| n.is_ascii_digit()) {
                    let modifier = chars.next().unwrap();
                    expanded.push_str(&substitute(
                        template,
                        &sources,
                        &mut chars,
                        Some(modifier),
                    ));
                } else {
                    expanded.push('%');
                }
            }
            Some(d) if d.is_ascii_digit() => {
                expanded.push_str(&substitute(template, &sources, &mut chars, None));
            }
            // `%` before anything else is a literal percent sign (e.g. a
            // register reference like `%r10`).
            _ => expanded.push('%'),
        }
    }

    let bytes = cranelift_assembler_glasm::encode_module(&expanded, template.att_syntax)
        .unwrap_or_else(|e| panic!("inline asm: {e}"));
    for byte in bytes {
        sink.put1(byte);
    }
}

/// Where an operand's substitution text comes from.
enum Source {
    Reg(PReg),
    Const(i64),
}

/// Resolve one `%[modifier]N` reference, consuming the operand digits
/// from `chars`.
fn substitute(
    template: &InlineAsmTemplate,
    sources: &[Source],
    chars: &mut core::iter::Peekable<core::str::Chars>,
    modifier: Option<char>,
) -> String {
    let mut num = 0usize;
    while let Some(d) = chars.peek() {
        if let Some(v) = d.to_digit(10) {
            num = num * 10 + v as usize;
            chars.next();
        } else {
            break;
        }
    }
    let source = sources.get(num).unwrap_or_else(|| {
        panic!(
            "inline asm: template references operand %{num}, but only {} \
             operands exist",
            sources.len()
        )
    });

    // The template records whether the operand was a memory operand; its
    // default rendering wraps the address register.
    let is_mem = matches!(
        template.refs.get(num),
        Some(AsmOperandRef::MemUse(_))
    );

    match (modifier, source) {
        (Some('c'), Source::Const(v)) => format!("{v}"),
        (Some('c'), _) => panic!("inline asm: %c modifier applies to immediate operands only"),
        (Some('n'), Source::Const(v)) => {
            v.checked_neg()
                .unwrap_or_else(|| panic!("inline asm: %n cannot negate i64::MIN"))
                .to_string()
        }
        (Some('n'), _) => panic!("inline asm: %n modifier applies to immediate operands only"),
        (Some('h'), Source::Reg(preg)) => reg_name_high_byte(*preg, template.att_syntax),
        (Some('h'), _) => panic!("inline asm: %h modifier applies to register operands only"),
        (Some(m @ ('b' | 'w' | 'k' | 'q')), Source::Reg(preg)) => {
            let size = match m {
                'b' => asm::gpr::Size::Byte,
                'w' => asm::gpr::Size::Word,
                'k' => asm::gpr::Size::Doubleword,
                _ => asm::gpr::Size::Quadword,
            };
            reg_name_sized(*preg, size, template.att_syntax)
        }
        (Some(_), _) => panic!("inline asm: unsupported % modifier on this operand kind"),
        (None, Source::Reg(preg)) => {
            if is_mem {
                if template.att_syntax {
                    format!("({})", reg_name(*preg, true))
                } else {
                    format!("[{}]", reg_name(*preg, false))
                }
            } else {
                reg_name(*preg, template.att_syntax)
            }
        }
        (None, Source::Const(v)) => {
            if template.att_syntax {
                format!("${v}")
            } else {
                format!("{v}")
            }
        }
    }
}
