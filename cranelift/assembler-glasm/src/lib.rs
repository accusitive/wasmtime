//! glasm assembler core, vendored from the glcc project's `glasm` crate.
//!
//! Pipeline: `parse` (Intel syntax) or `gas` (AT&T syntax) turn source text
//! into `Stmt`s, and `table::encode` matches each instruction against the
//! nasm-derived tables in `insns_gen` to produce bytes plus fixups. The ELF
//! writer and symbol machinery from the upstream crate are not vendored;
//! symbol references are rejected (see `encode_module`).

#![no_std]

extern crate alloc;

use alloc::string::{String, ToString};
use alloc::vec::Vec;

pub mod encoding;
pub mod expr;
pub mod gas;
pub mod insns_gen;
pub mod insn;
pub mod parse;
pub mod regs;
pub mod table;

/// Assembles source text (one or more instructions, comments and labels
/// tolerated) and returns the concatenated machine-code bytes. Fixups —
/// label or symbol references — are rejected; the caller owns relocation
/// resolution and can extend this API when needed.
pub fn encode_module(src: &str, att_syntax: bool) -> Result<Vec<u8>, String> {
    let stmts = if att_syntax {
        gas::parse_module(src)
    } else {
        parse::parse_module(src)
    }
    .map_err(|errs| {
        errs.into_iter()
            .map(|e| alloc::format!("line {}: {}", e.line, e.msg))
            .collect::<Vec<_>>()
            .join("; ")
    })?;

    let syms = NoSymbols;
    let ctx = expr::EvalCtx {
        cur_sect: 0,
        cur_off: 0,
        syms: &syms,
        optimistic: false,
        unknown_hit: None,
    };

    let mut out = Vec::new();
    for stmt in stmts {
        let insn = match stmt {
            insn::Stmt::Insn(i) => i,
            _ => continue,
        };
        let encoded = table::encode(&insn, &ctx).map_err(|e| e.to_string())?;
        if !encoded.fixups.is_empty() {
            return Err("unsupported: inline asm referencing labels or symbols".to_string());
        }
        out.extend_from_slice(&encoded.bytes);
    }
    Ok(out)
}

struct NoSymbols;

impl expr::SymLookup for NoSymbols {
    fn lookup(&self, _name: &str) -> Option<expr::SymbolVal> {
        None
    }
}
