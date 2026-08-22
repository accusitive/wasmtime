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

    TargetIsa::new("ppu64", settings.build())
}
