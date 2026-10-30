//! glasm assembler core, vendored from the glcc project's `glasm` crate.
//!
//! Pipeline: `parse` (Intel syntax) or `gas` (AT&T syntax) turn source text
//! into `Stmt`s, and `table::encode` matches each instruction against the
//! nasm-derived tables in `insns_gen` to produce bytes plus fixups. The ELF
//! writer and symbol machinery from the upstream crate are not vendored;
//! symbol references outside the template are rejected (see `encode_module`).

#![no_std]

extern crate alloc;

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::cell::RefCell;

pub mod encoding;
pub mod expr;
pub mod gas;
pub mod insns_gen;
pub mod insn;
pub mod parse;
pub mod regs;
pub mod table;

/// Assembles source text (one or more instructions, comments and labels
/// tolerated) and returns the concatenated machine-code bytes.
///
/// Labels defined inside the template may be referenced from it (local
/// loop targets, in either direction); anything needing a relocation —
/// external symbols, references out of the template — is rejected.
pub fn encode_module(src: &str, att_syntax: bool) -> Result<Vec<u8>, String> {
    let stmts = if att_syntax {
        gas::parse_module(src)
    } else {
        parse::parse_module(src)
    }
    .map_err(|errs| {
        errs.into_iter()
            .map(|e| format!("line {}: {}", e.line, e.msg))
            .collect::<Vec<_>>()
            .join("; ")
    })?;

    // Pre-register every label at offset zero so forward references
    // encode during the first pass; offsets are filled in as labels are
    // encountered.
    let labels: RefCell<Vec<(String, i64)>> = RefCell::new(Vec::new());
    for stmt in &stmts {
        if let insn::Stmt::Label(name, _) = stmt {
            let mut table = labels.borrow_mut();
            if !table.iter().any(|(n, _)| n == name) {
                table.push((name.clone(), 0));
            }
        }
    }
    let syms = TemplateSymbols(&labels);

    // Pass 1 discovers the label offsets (forward references encode with
    // zero displacements).
    let mut pass1 = expr::EvalCtx {
        cur_sect: 0,
        cur_off: 0,
        syms: &syms,
        optimistic: false,
        unknown_hit: None,
    };
    assemble(&stmts, &mut pass1, &labels, None)?;

    // Pass 2 with the complete label table, keeping bytes and fixups.
    let mut out = Vec::new();
    let mut pass2 = expr::EvalCtx {
        cur_sect: 0,
        cur_off: 0,
        syms: &syms,
        optimistic: false,
        unknown_hit: None,
    };
    let fixups = assemble(&stmts, &mut pass2, &labels, Some(&mut out))?;

    // Resolve fixups: same-template targets are patched directly;
    // everything else would need a relocation the caller would have to
    // service.
    for (pos, fixup) in fixups {
        let value = match fixup.val {
            expr::RValue::Const(v) => v,
            expr::RValue::SectVal { sect, off } if sect == 0 => {
                if fixup.pcrel {
                    off - (pos as i64 + fixup.size as i64 + fixup.tail as i64)
                } else {
                    off
                }
            }
            expr::RValue::SectVal { .. } | expr::RValue::ExternRef { .. } => {
                return Err(
                    "unsupported: inline asm referencing symbols outside the template"
                        .to_string(),
                )
            }
        };
        if !fits_signed(value, fixup.size) {
            return Err(format!(
                "inline asm: displacement {value} does not fit in {} bytes; \
                 templates must use short jumps",
                fixup.size
            ));
        }
        for byte in 0..fixup.size as usize {
            out[pos + byte] = (value >> (byte * 8)) as u8;
        }
    }
    Ok(out)
}

fn fits_signed(v: i64, size: u8) -> bool {
    match size {
        1 => (-128..=127).contains(&v),
        2 => (-32768..=32767).contains(&v),
        4 => (-2147483648..=2147483647).contains(&v),
        8 => true,
        _ => false,
    }
}

/// One encoding pass over the parsed statements. When `out` is `None` the
/// pass only measures lengths (for label placement); otherwise it appends
/// bytes and reports fixup positions.
fn assemble(
    stmts: &[insn::Stmt],
    ctx: &mut expr::EvalCtx,
    labels: &RefCell<Vec<(String, i64)>>,
    mut out: Option<&mut Vec<u8>>,
) -> Result<Vec<(usize, encoding::FixupReq)>, String> {
    let mut len = 0i64;
    let mut fixups = Vec::new();
    for stmt in stmts {
        match stmt {
            insn::Stmt::Insn(i) => {
                let encoded = table::encode(i, ctx).map_err(|e| e.to_string())?;
                for f in &encoded.fixups {
                    fixups.push(((len + f.at as i64) as usize, *f));
                }
                if let Some(o) = out.as_deref_mut() {
                    o.extend_from_slice(&encoded.bytes);
                }
                len += encoded.bytes.len() as i64;
                ctx.cur_off = len;
            }
            insn::Stmt::Label(name, _) => {
                let mut table = labels.borrow_mut();
                if let Some(entry) = table.iter_mut().find(|(n, _)| n == name) {
                    entry.1 = len;
                }
            }
            _ => {}
        }
    }
    Ok(fixups)
}

/// Symbol table holding the labels defined inside one template.
struct TemplateSymbols<'a>(&'a RefCell<Vec<(String, i64)>>);

impl expr::SymLookup for TemplateSymbols<'_> {
    fn lookup(&self, name: &str) -> Option<expr::SymbolVal> {
        self.0
            .borrow()
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, off)| expr::SymbolVal {
                index: 0,
                defined: true,
                seen: true,
                extern_: false,
                sect: Some(0),
                value: *off,
            })
    }
}
