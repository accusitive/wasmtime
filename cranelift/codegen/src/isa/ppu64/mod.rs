//! PowerPC64 (Cell Broadband Engine PPU) Instruction Set Architecture.

use crate::dominator_tree::DominatorTree;
use crate::ir;
use crate::ir::{Function, Type};
use crate::isa::ppu64::settings as ppu64_settings;
use crate::isa::{
    Builder as IsaBuilder, FunctionAlignment, IsaFlagsHashKey, OwnedTargetIsa, PointerWidth,
    TargetIsa,
};
#[cfg(feature = "unwind")]
use crate::machinst::CompiledCode;
use crate::machinst::{
    CompiledCodeStencil, MachTextSectionBuilder, Reg, SigSet, TextSectionBuilder, VCode, compile,
};
use crate::result::CodegenResult;
use crate::settings::{self as shared_settings, Flags};
use alloc::string::String;
use alloc::{boxed::Box, vec::Vec};
use core::fmt;
use cranelift_control::ControlPlane;
use target_lexicon::{Architecture, Triple};

mod abi;
pub(crate) mod inst;
mod lower;
mod settings;

use self::inst::EmitInfo;

/// A PowerPC64 (PPU) backend.
pub struct Ppu64Backend {
    triple: Triple,
    flags: shared_settings::Flags,
    isa_flags: ppu64_settings::Flags,
}

impl Ppu64Backend {
    /// Create a new ppu64 backend with the given (shared) flags.
    pub fn new_with_flags(
        triple: Triple,
        flags: shared_settings::Flags,
        isa_flags: ppu64_settings::Flags,
    ) -> Ppu64Backend {
        Ppu64Backend {
            triple,
            flags,
            isa_flags,
        }
    }

    /// This performs lowering to VCode, register-allocates the code, computes
    /// block layout and finalizes branches. The result is ready for binary
    /// emission.
    fn compile_vcode(
        &self,
        func: &Function,
        domtree: &DominatorTree,
        regalloc_ctx: &mut regalloc2::Ctx,
        ctrl_plane: &mut ControlPlane,
    ) -> CodegenResult<VCode<inst::Inst>> {
        let cellos = self.triple.architecture == target_lexicon::Architecture::Powerpc64
            && self.triple.operating_system == target_lexicon::OperatingSystem::Unknown;
        let emit_info = EmitInfo::new(self.flags.clone(), self.isa_flags.clone(), cellos);
        let sigs = SigSet::new::<abi::Ppu64MachineDeps>(func, &self.flags)?;
        let abi = abi::Ppu64Callee::new(func, self, &self.isa_flags, &sigs)?;
        compile::compile::<Ppu64Backend>(
            func,
            domtree,
            regalloc_ctx,
            self,
            abi,
            emit_info,
            sigs,
            ctrl_plane,
        )
    }
}

impl TargetIsa for Ppu64Backend {
    fn compile_function(
        &self,
        func: &Function,
        domtree: &DominatorTree,
        regalloc_ctx: &mut regalloc2::Ctx,
        want_disasm: bool,
        ctrl_plane: &mut ControlPlane,
    ) -> CodegenResult<CompiledCodeStencil> {
        let vcode = self.compile_vcode(func, domtree, regalloc_ctx, ctrl_plane)?;

        let want_disasm = want_disasm || log::log_enabled!(log::Level::Debug);
        let emit_result = vcode.emit(&regalloc_ctx.output, want_disasm, &self.flags, ctrl_plane)?;
        let value_labels_ranges = emit_result.value_labels_ranges;
        let buffer = emit_result.buffer;

        if let Some(disasm) = emit_result.disasm.as_ref() {
            log::debug!("disassembly:\n{disasm}");
        }

        Ok(CompiledCodeStencil {
            buffer,
            vcode: emit_result.disasm,
            value_labels_ranges,
            bb_starts: emit_result.bb_offsets,
            bb_edges: emit_result.bb_edges,
        })
    }

    fn name(&self) -> &'static str {
        "ppu64"
    }

    fn dynamic_vector_bytes(&self, _dynamic_ty: ir::Type) -> u32 {
        16
    }

    fn triple(&self) -> &Triple {
        &self.triple
    }

    /// CellOS ILP32 hybrid: report U32 pointer width so cg_clif emits
    /// I32-typed pointer values (PS3DK abi spec §4). The ELF container
    /// stays ELF64 EM_PPC64 either way.
    fn pointer_width(&self) -> PointerWidth {
        if self.isa_flags.ilp32() {
            PointerWidth::U32
        } else {
            self.triple.pointer_width().unwrap()
        }
    }

    fn flags(&self) -> &shared_settings::Flags {
        &self.flags
    }

    fn isa_flags(&self) -> Vec<shared_settings::Value> {
        self.isa_flags.iter().collect()
    }

    fn isa_flags_hash_key(&self) -> IsaFlagsHashKey<'_> {
        IsaFlagsHashKey(self.isa_flags.hash_key())
    }

    #[cfg(feature = "unwind")]
    fn emit_unwind_info(
        &self,
        _result: &CompiledCode,
        _kind: crate::isa::unwind::UnwindInfoKind,
    ) -> CodegenResult<Option<crate::isa::unwind::UnwindInfo>> {
        Ok(None)
    }

    fn text_section_builder(&self, num_funcs: usize) -> Box<dyn TextSectionBuilder> {
        Box::new(MachTextSectionBuilder::<inst::Inst>::new(num_funcs))
    }

    fn function_alignment(&self) -> FunctionAlignment {
        <inst::Inst as crate::machinst::MachInst>::function_alignment()
    }

    fn page_size_align_log2(&self) -> u8 {
        12
    }

    #[cfg(feature = "disas")]
    fn to_capstone(&self) -> Result<capstone::Capstone, capstone::Error> {
        use capstone::Endian;
        use capstone::arch::ppc::ArchMode;
        use capstone::prelude::*;
        Capstone::new()
            .ppc()
            .mode(ArchMode::Mode64)
            .endian(Endian::Big)
            .build()
    }

    fn pretty_print_reg(&self, reg: Reg, _size: u8) -> String {
        format!("{reg:?}")
    }

    fn has_native_fma(&self) -> bool {
        true
    }

    fn has_round(&self) -> bool {
        true
    }

    fn has_blendv_lowering(&self, _: Type) -> bool {
        false
    }

    fn has_x86_pshufb_lowering(&self) -> bool {
        false
    }

    fn has_x86_pmulhrsw_lowering(&self) -> bool {
        false
    }

    fn has_x86_pmaddubsw_lowering(&self) -> bool {
        false
    }

    fn default_argument_extension(&self) -> ir::ArgumentExtension {
        // The PPC64 ELFv1 ABI passes small integer arguments in the
        // low-order bits of a register without requiring extension.
        ir::ArgumentExtension::None
    }
}

impl fmt::Display for Ppu64Backend {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("MachBackend")
            .field("name", &self.name())
            .field("triple", &self.triple())
            .field("flags", &format!("{}", self.flags()))
            .finish()
    }
}

/// Create a new `isa::Builder`.
pub fn isa_builder(triple: Triple) -> IsaBuilder {
    match triple.architecture {
        Architecture::Powerpc64 | Architecture::Powerpc => {}
        _ => unreachable!(),
    }
    IsaBuilder {
        triple,
        setup: ppu64_settings::builder(),
        constructor: isa_constructor,
    }
}

fn isa_constructor(
    triple: Triple,
    shared_flags: Flags,
    builder: &shared_settings::Builder,
) -> CodegenResult<OwnedTargetIsa> {
    let isa_flags = ppu64_settings::Flags::new(&shared_flags, builder);
    let backend = Ppu64Backend::new_with_flags(triple, shared_flags, isa_flags);
    Ok(backend.wrapped())
}
