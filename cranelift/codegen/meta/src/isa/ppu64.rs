use crate::cdsl::isa::TargetIsa;
use crate::cdsl::settings::SettingGroupBuilder;

pub(crate) fn define() -> TargetIsa {
    let mut settings = SettingGroupBuilder::new("ppu64");

    // The Cell BE PPU is a POWER5-class core with AltiVec but no VSX.
    let _has_altivec = settings.add_bool(
        "has_altivec",
        "has AltiVec?",
        "AltiVec (VMX) vector instruction support",
        false,
    );

    // CellOS Lv-2 ILP32 hybrid (PS3DK abi spec §4): pointers are 32-bit
    // effective addresses inside a 64-bit ELF container. When set, the ISA
    // reports PointerWidth::U32 so cg_clif emits I32-typed pointer values.
    let _ilp32 = settings.add_bool(
        "ilp32",
        "ILP32 data model",
        "pointers are 32-bit effective addresses (CellOS Lv-2 hybrid)",
        false,
    );

    TargetIsa::new("ppu64", settings.build())
}
