//! ppu64 ISA definitions: registers.

use crate::machinst::{Reg, Writable};

use regalloc2::{PReg, RegClass, VReg};

#[inline]
pub const fn r_reg(enc: usize) -> Reg {
    let p_reg = PReg::new(enc, RegClass::Int);
    let v_reg = VReg::new(p_reg.index(), p_reg.class());
    Reg::from_virtual_reg(v_reg)
}
pub const fn pr_reg(enc: usize) -> PReg {
    PReg::new(enc, RegClass::Int)
}

#[inline]
pub fn f_reg(enc: usize) -> Reg {
    let p_reg = PReg::new(enc, RegClass::Float);
    let v_reg = VReg::new(p_reg.index(), p_reg.class());
    Reg::from(v_reg)
}
pub const fn pf_reg(enc: usize) -> PReg {
    PReg::new(enc, RegClass::Float)
}

#[inline]
pub fn v_reg(enc: usize) -> Reg {
    let p_reg = PReg::new(enc, RegClass::Vector);
    let v_reg = VReg::new(p_reg.index(), p_reg.class());
    Reg::from(v_reg)
}
pub const fn pv_reg(enc: usize) -> PReg {
    PReg::new(enc, RegClass::Vector)
}

/// r0 reads as zero in address computations and some immediate forms,
/// which makes it hazardous to allocate; we exclude it from allocation
/// and use it only as an internal scratch register.
#[inline]
pub fn scratch_reg() -> Reg {
    r_reg(0)
}

#[inline]
pub fn writable_scratch_reg() -> Writable<Reg> {
    Writable::from_reg(scratch_reg())
}

#[inline]
pub fn stack_reg() -> Reg {
    r_reg(1)
}

#[inline]
pub fn writable_stack_reg() -> Writable<Reg> {
    Writable::from_reg(stack_reg())
}

/// r2 holds the TOC pointer in the ELFv1 ABI; excluded from allocation.
#[inline]
pub fn toc_reg() -> Reg {
    r_reg(2)
}

/// r13 holds the TLS pointer; excluded from allocation.
#[inline]
pub fn tls_reg() -> Reg {
    r_reg(13)
}

#[inline]
pub fn fp_reg() -> Reg {
    r_reg(31)
}

#[inline]
pub fn writable_fp_reg() -> Writable<Reg> {
    Writable::from_reg(fp_reg())
}

#[inline]
pub fn link_reg_tmp() -> Reg {
    // Scratch register used to save/restore LR around calls.
    scratch_reg()
}

#[inline]
pub fn spilltmp_reg() -> Reg {
    r_reg(11)
}

#[inline]
pub fn writable_spilltmp_reg() -> Writable<Reg> {
    Writable::from_reg(spilltmp_reg())
}

#[inline]
pub fn spilltmp_reg2() -> Reg {
    r_reg(12)
}

#[inline]
pub fn writable_spilltmp_reg2() -> Writable<Reg> {
    Writable::from_reg(spilltmp_reg2())
}
