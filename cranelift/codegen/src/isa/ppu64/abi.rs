//! Implementation of the PowerPC64 (ELFv1/SysV) ABI for the Cell BE PPU.

use crate::ir;
use crate::ir::condcodes::IntCC;
use crate::ir::types::*;
use crate::isa::unwind::UnwindInst;

use crate::isa;

use crate::isa::CallConv;
use crate::isa::ppu64::inst::*;
use crate::machinst::*;

use crate::CodegenResult;
use crate::ir::LibCall;
use crate::ir::Signature;
use crate::settings;
use alloc::boxed::Box;
use alloc::vec::Vec;
use regalloc2::{MachineEnv, PRegSet};

use alloc::borrow::ToOwned;
use smallvec::{SmallVec, smallvec};

/// Support for the PPC64 ABI from the callee side (within a function body).
pub(crate) type Ppu64Callee = Callee<Ppu64MachineDeps>;

/// ppu64-specific ABI behavior. This struct just serves as an implementation
/// point for the trait; it is never actually instantiated.
pub struct Ppu64MachineDeps;

impl IsaFlags for crate::isa::ppu64::settings::Flags {}

impl ABIMachineSpec for Ppu64MachineDeps {
    type I = Inst;
    type F = crate::isa::ppu64::settings::Flags;

    /// This is the limit for the size of argument and return-value areas on
    /// the stack.
    const STACK_ARG_RET_SIZE_LIMIT: u32 = 128 * 1024 * 1024;

    fn word_bits() -> u32 {
        64
    }

    /// Return required stack alignment in bytes.
    fn stack_align(_call_conv: isa::CallConv) -> u32 {
        16
    }

    fn compute_arg_locs(
        call_conv: isa::CallConv,
        flags: &settings::Flags,
        params: &[ir::AbiParam],
        args_or_rets: ArgsOrRets,
        add_ret_area_ptr: bool,
        mut args: ArgsAccumulator,
    ) -> CodegenResult<(u32, Option<usize>)> {
        // This implements a simplified ELFv1 (big-endian PPC64) ABI:
        //
        // - integer args go in r3..r10, then on the stack
        // - float args go in f1..f13, then on the stack
        // - integer return values go in r3..r8
        // - float return values go in f1..f8

        assert_ne!(
            call_conv,
            isa::CallConv::Winch,
            "ppu64 does not support the 'winch' calling convention yet"
        );

        let (x_start, x_end, f_start, f_end, v_start, v_end) = match args_or_rets {
            ArgsOrRets::Args => (3, 10, 1, 13, 2, 13),
            ArgsOrRets::Rets => (3, 8, 1, 8, 2, 8),
        };
        let mut next_x_reg = x_start;
        let mut next_f_reg = f_start;
        let mut next_v_reg = v_start;
        let mut next_stack: u32 = 0;

        let ret_area_ptr = if add_ret_area_ptr {
            assert!(ArgsOrRets::Args == args_or_rets);
            next_x_reg += 1;
            Some(ABIArg::reg(
                r_reg(x_start).to_real_reg().unwrap(),
                I64,
                ir::ArgumentExtension::None,
                ir::ArgumentPurpose::Normal,
            ))
        } else {
            None
        };

        for param in params {
            if let ir::ArgumentPurpose::StructArgument(_) = param.purpose {
                panic!(
                    "StructArgument parameters are not supported on ppu64. \
                     Use regular pointer arguments instead."
                );
            }

            let (rcs, reg_tys) = Inst::rc_for_type(&param.value_type)?;
            let mut slots = ABIArgSlotVec::new();
            for (rc, reg_ty) in rcs.iter().zip(reg_tys.iter()) {
                let next_reg = if (next_x_reg <= x_end) && *rc == RegClass::Int {
                    let x = Some(r_reg(next_x_reg));
                    next_x_reg += 1;
                    x
                } else if (next_f_reg <= f_end) && *rc == RegClass::Float {
                    let x = Some(f_reg(next_f_reg));
                    next_f_reg += 1;
                    x
                } else if (next_v_reg <= v_end) && *rc == RegClass::Vector {
                    let x = Some(v_reg(next_v_reg));
                    next_v_reg += 1;
                    x
                } else {
                    None
                };
                if let Some(reg) = next_reg {
                    slots.push(ABIArgSlot::Reg {
                        reg: reg.to_real_reg().unwrap(),
                        ty: *reg_ty,
                        extension: param.extension,
                    });
                } else {
                    if args_or_rets == ArgsOrRets::Rets && !flags.enable_multi_ret_implicit_sret() {
                        return Err(crate::CodegenError::Unsupported(
                            "Too many return values to fit in registers. \
                             Use a StructReturn argument instead."
                                .to_owned(),
                        ));
                    }

                    let size = reg_ty.bits() / 8;
                    let size = core::cmp::max(size, 8);
                    debug_assert!(size.is_power_of_two());
                    next_stack = align_to(next_stack, size);
                    slots.push(ABIArgSlot::Stack {
                        offset: next_stack as i64,
                        ty: *reg_ty,
                        extension: param.extension,
                    });
                    next_stack += size;
                }
            }
            args.push(ABIArg::Slots {
                slots,
                purpose: param.purpose,
            });
        }
        let pos = if let Some(ret_area_ptr) = ret_area_ptr {
            args.push_non_formal(ret_area_ptr);
            Some(args.args().len() - 1)
        } else {
            None
        };

        next_stack = align_to(next_stack, Self::stack_align(call_conv));

        Ok((next_stack, pos))
    }

    fn gen_load_stack(mem: StackAMode, into_reg: Writable<Reg>, ty: Type) -> Inst {
        Inst::gen_load(into_reg, mem.into(), ty, MemFlagsData::trusted())
    }

    fn gen_store_stack(mem: StackAMode, from_reg: Reg, ty: Type) -> Inst {
        Inst::gen_store(mem.into(), from_reg, ty, MemFlagsData::trusted())
    }

    fn gen_move(to_reg: Writable<Reg>, from_reg: Reg, ty: Type) -> Inst {
        Inst::gen_move(to_reg, from_reg, ty)
    }

    fn gen_extend(
        to_reg: Writable<Reg>,
        from_reg: Reg,
        signed: bool,
        from_bits: u8,
        to_bits: u8,
    ) -> Inst {
        assert!(from_bits < to_bits);
        Inst::Extend {
            rd: to_reg,
            rn: from_reg,
            signed,
            from_bits,
        }
    }

    fn get_ext_mode(
        _call_conv: isa::CallConv,
        specified: ir::ArgumentExtension,
    ) -> ir::ArgumentExtension {
        specified
    }

    fn gen_args(args: Vec<ArgPair>) -> Inst {
        Inst::Args { args }
    }

    fn gen_rets(rets: Vec<RetPair>) -> Inst {
        Inst::Rets { rets }
    }

    fn get_stacklimit_reg(_call_conv: isa::CallConv) -> Reg {
        spilltmp_reg2()
    }

    fn gen_add_imm(
        _call_conv: isa::CallConv,
        into_reg: Writable<Reg>,
        from_reg: Reg,
        imm: u32,
    ) -> SmallInstVec<Inst> {
        let mut insts = SmallInstVec::new();
        if let Some(simm) = SImm16::maybe_from_i64(imm as i64) {
            insts.push(Inst::AluRRSimm16 {
                alu_op: AluOPRRI::Addi,
                rd: into_reg,
                ra: from_reg,
                simm,
            });
        } else {
            insts.extend(Inst::load_constant_u32(
                writable_spilltmp_reg2(),
                imm as u64,
            ));
            insts.push(Inst::AluRRR {
                alu_op: AluOPRRR::Add,
                rd: into_reg,
                rs1: spilltmp_reg2(),
                rs2: from_reg,
            });
        }
        insts
    }

    fn gen_stack_lower_bound_trap(limit_reg: Reg) -> SmallInstVec<Inst> {
        let mut insts = SmallVec::new();
        insts.push(Inst::TrapIf {
            cmp: IntegerCompare {
                kind: IntCC::UnsignedLessThan,
                rs1: stack_reg(),
                rs2: limit_reg,
                is64: true,
            },
            trap_code: ir::TrapCode::STACK_OVERFLOW,
        });
        insts
    }

    fn gen_get_stack_addr(mem: StackAMode, into_reg: Writable<Reg>) -> Inst {
        Inst::LoadAddr {
            rd: into_reg,
            mem: mem.into(),
        }
    }

    fn gen_load_base_offset(into_reg: Writable<Reg>, base: Reg, offset: i32, ty: Type) -> Inst {
        let mem = AMode::RegOffset(base, offset as i64);
        Inst::gen_load(into_reg, mem, ty, MemFlagsData::trusted())
    }

    fn gen_store_base_offset(base: Reg, offset: i32, from_reg: Reg, ty: Type) -> Inst {
        let mem = AMode::RegOffset(base, offset as i64);
        Inst::gen_store(mem, from_reg, ty, MemFlagsData::trusted())
    }

    fn gen_sp_reg_adjust(amount: i32) -> SmallInstVec<Inst> {
        let mut insts = SmallVec::new();

        if amount == 0 {
            return insts;
        }

        if let Some(simm) = SImm16::maybe_from_i64(amount as i64) {
            insts.push(Inst::AluRRSimm16 {
                alu_op: AluOPRRI::Addi,
                rd: writable_stack_reg(),
                ra: stack_reg(),
                simm,
            })
        } else {
            let tmp = writable_spilltmp_reg2();
            insts.extend(Inst::load_constant_u64(tmp, amount as i64 as u64));
            insts.push(Inst::AluRRR {
                alu_op: AluOPRRR::Add,
                rd: writable_stack_reg(),
                rs1: stack_reg(),
                rs2: tmp.to_reg(),
            });
        }

        insts
    }

    fn gen_prologue_frame_setup(
        _call_conv: isa::CallConv,
        flags: &settings::Flags,
        _isa_flags: &crate::isa::ppu64::settings::Flags,
        frame_layout: &FrameLayout,
    ) -> SmallInstVec<Inst> {
        let mut insts = SmallVec::new();

        if frame_layout.setup_area_size > 0 {
            // mflr  r0          ; save LR into scratch
            // stdu  sp,-16(sp)  ; allocate the setup area (FP + LR slots)
            // std   fp, 0(sp)
            // std   r0, 8(sp)
            // mr    fp, sp
            insts.push(mflr(writable_scratch_reg()));
            insts.extend(Self::gen_sp_reg_adjust(-16));
            insts.push(Inst::gen_store(
                AMode::SPOffset(0),
                fp_reg(),
                I64,
                MemFlagsData::trusted(),
            ));
            insts.push(Inst::gen_store(
                AMode::SPOffset(8),
                scratch_reg(),
                I64,
                MemFlagsData::trusted(),
            ));

            if flags.unwind_info() {
                insts.push(Inst::Unwind {
                    inst: UnwindInst::PushFrameRegs {
                        offset_upward_to_caller_sp: frame_layout.setup_area_size,
                    },
                });
            }
            insts.push(Inst::gen_move(writable_fp_reg(), stack_reg(), I64));
        }

        insts
    }

    /// reverse of gen_prologue_frame_setup.
    fn gen_epilogue_frame_restore(
        call_conv: isa::CallConv,
        _flags: &settings::Flags,
        _isa_flags: &crate::isa::ppu64::settings::Flags,
        frame_layout: &FrameLayout,
    ) -> SmallInstVec<Inst> {
        let mut insts = SmallVec::new();

        if frame_layout.setup_area_size > 0 {
            // ld    fp, 0(sp)
            // ld    r0, 8(sp)
            // mtlr  r0
            // addi  sp, sp, 16
            insts.push(Inst::gen_load(
                writable_fp_reg(),
                AMode::SPOffset(0),
                I64,
                MemFlagsData::trusted(),
            ));
            insts.push(Inst::gen_load(
                writable_scratch_reg(),
                AMode::SPOffset(8),
                I64,
                MemFlagsData::trusted(),
            ));
            insts.push(mtlr(writable_scratch_reg()));
            insts.extend(Self::gen_sp_reg_adjust(16));
        }

        if call_conv == isa::CallConv::Tail && frame_layout.tail_args_size > 0 {
            insts.extend(Self::gen_sp_reg_adjust(
                frame_layout.tail_args_size.try_into().unwrap(),
            ));
        }

        insts
    }

    fn gen_return(
        _call_conv: isa::CallConv,
        _isa_flags: &crate::isa::ppu64::settings::Flags,
        _frame_layout: &FrameLayout,
    ) -> SmallInstVec<Inst> {
        smallvec![Inst::Ret {}]
    }

    fn gen_inline_probestack(
        insts: &mut SmallInstVec<Self::I>,
        _call_conv: isa::CallConv,
        frame_size: u32,
        guard_size: u32,
    ) {
        // Probe once per guard-size region by storing to the bottom of the
        // frame. Loops are not supported yet; large frames will probe only
        // the final page.
        if frame_size == 0 {
            return;
        }
        insts.push(Inst::gen_store(
            AMode::SPOffset(-(core::cmp::min(frame_size, guard_size * 16) as i64)),
            scratch_reg(),
            I32,
            MemFlagsData::trusted(),
        ));
    }

    fn gen_probestack(insts: &mut SmallInstVec<Self::I>, frame_size: u32) {
        insts.extend(Inst::load_constant_u32(
            Writable::from_reg(r_reg(3)),
            frame_size as u64,
        ));
        let mut info = CallInfo::empty(
            ExternalName::LibCall(LibCall::Probestack),
            CallConv::SystemV,
        );
        info.uses.push(CallArgPair {
            vreg: r_reg(3),
            preg: r_reg(3),
        });
        insts.push(Inst::Call {
            info: Box::new(info),
        });
    }

    fn gen_clobber_save(
        _call_conv: isa::CallConv,
        flags: &settings::Flags,
        frame_layout: &FrameLayout,
    ) -> SmallVec<[Inst; 16]> {
        let mut insts = SmallVec::new();
        let setup_frame = frame_layout.setup_area_size > 0;

        let incoming_args_diff = frame_layout.tail_args_size - frame_layout.incoming_args_size;
        if incoming_args_diff > 0 {
            insts.extend(Self::gen_sp_reg_adjust(-(incoming_args_diff as i32)));

            if setup_frame {
                insts.push(Inst::gen_store(
                    AMode::SPOffset(8),
                    scratch_reg(),
                    I64,
                    MemFlagsData::trusted(),
                ));
                insts.push(Inst::gen_load(
                    writable_fp_reg(),
                    AMode::SPOffset(i64::from(incoming_args_diff)),
                    I64,
                    MemFlagsData::trusted(),
                ));
                insts.push(Inst::gen_store(
                    AMode::SPOffset(0),
                    fp_reg(),
                    I64,
                    MemFlagsData::trusted(),
                ));

                insts.push(Inst::gen_move(writable_fp_reg(), stack_reg(), I64));
            }
        }

        if flags.unwind_info() && setup_frame {
            insts.push(Inst::Unwind {
                inst: UnwindInst::DefineNewFrame {
                    offset_downward_to_clobbers: frame_layout.clobber_size,
                    offset_upward_to_caller_sp: frame_layout.setup_area_size,
                },
            });
        }

        let stack_size = frame_layout.clobber_size
            + frame_layout.fixed_frame_storage_size
            + frame_layout.outgoing_args_size;

        if stack_size > 0 {
            insts.extend(Self::gen_sp_reg_adjust(-(stack_size as i32)));

            let mut cur_offset = 0;
            for reg in &frame_layout.clobbered_callee_saves {
                let r_reg_w = reg.to_reg();
                let ty = match r_reg_w.class() {
                    RegClass::Int => I64,
                    RegClass::Float => F64,
                    RegClass::Vector => I8X16,
                };
                cur_offset = align_to(cur_offset, ty.bytes());
                insts.push(Inst::gen_store(
                    AMode::SPOffset(i64::from(stack_size - cur_offset - ty.bytes())),
                    Reg::from(reg.to_reg()),
                    ty,
                    MemFlagsData::trusted(),
                ));

                if flags.unwind_info() {
                    insts.push(Inst::Unwind {
                        inst: UnwindInst::SaveReg {
                            clobber_offset: frame_layout.clobber_size - cur_offset - ty.bytes(),
                            reg: r_reg_w,
                        },
                    });
                }

                cur_offset += ty.bytes();
                assert!(cur_offset <= stack_size);
            }
        }
        insts
    }

    fn gen_clobber_restore(
        _call_conv: isa::CallConv,
        _flags: &settings::Flags,
        frame_layout: &FrameLayout,
    ) -> SmallVec<[Inst; 16]> {
        let mut insts = SmallVec::new();

        let stack_size = frame_layout.clobber_size
            + frame_layout.fixed_frame_storage_size
            + frame_layout.outgoing_args_size;
        let mut cur_offset = 0;

        for reg in &frame_layout.clobbered_callee_saves {
            let rreg = reg.to_reg();
            let ty = match rreg.class() {
                RegClass::Int => I64,
                RegClass::Float => F64,
                RegClass::Vector => I8X16,
            };
            cur_offset = align_to(cur_offset, ty.bytes());
            insts.push(Inst::gen_load(
                reg.map(Reg::from),
                AMode::SPOffset(i64::from(stack_size - cur_offset - ty.bytes())),
                ty,
                MemFlagsData::trusted(),
            ));
            cur_offset += ty.bytes();
        }

        if stack_size > 0 {
            insts.extend(Self::gen_sp_reg_adjust(stack_size as i32));
        }

        insts
    }

    fn gen_memcpy<F: FnMut(Type) -> Writable<Reg>>(
        call_conv: isa::CallConv,
        dst: Reg,
        src: Reg,
        size: usize,
        mut alloc_tmp: F,
    ) -> SmallVec<[Self::I; 8]> {
        let mut insts = SmallVec::new();
        let arg0 = Writable::from_reg(r_reg(3));
        let arg1 = Writable::from_reg(r_reg(4));
        let arg2 = Writable::from_reg(r_reg(5));
        let tmp = alloc_tmp(Self::word_type());
        insts.extend(Inst::load_constant_u64(tmp, size as u64));
        insts.push(Inst::Call {
            info: Box::new(CallInfo {
                dest: ExternalName::LibCall(LibCall::Memcpy),
                uses: smallvec![
                    CallArgPair {
                        vreg: dst,
                        preg: arg0.to_reg()
                    },
                    CallArgPair {
                        vreg: src,
                        preg: arg1.to_reg()
                    },
                    CallArgPair {
                        vreg: tmp.to_reg(),
                        preg: arg2.to_reg()
                    }
                ],
                defs: smallvec![],
                clobbers: Self::get_regs_clobbered_by_call(call_conv, false),
                caller_conv: call_conv,
                callee_conv: call_conv,
                callee_pop_size: 0,
                try_call_info: None,
                patchable: false,
            }),
        });
        insts
    }

    fn get_number_of_spillslots_for_value(
        rc: RegClass,
        _target_vector_bytes: u32,
        _isa_flags: &crate::isa::ppu64::settings::Flags,
    ) -> u32 {
        match rc {
            RegClass::Int => 1,
            RegClass::Float => 1,
            RegClass::Vector => 2,
        }
    }

    fn get_machine_env(_flags: &settings::Flags, _call_conv: isa::CallConv) -> &MachineEnv {
        static MACHINE_ENV: MachineEnv = create_reg_environment();
        &MACHINE_ENV
    }

    fn get_regs_clobbered_by_call(
        call_conv_of_callee: isa::CallConv,
        is_exception: bool,
    ) -> PRegSet {
        match call_conv_of_callee {
            isa::CallConv::Tail if is_exception => ALL_CLOBBERS,
            isa::CallConv::PreserveAll if is_exception => ALL_CLOBBERS,
            isa::CallConv::PreserveAll => NO_CLOBBERS,
            _ => DEFAULT_CLOBBERS,
        }
    }

    fn compute_frame_layout(
        call_conv: isa::CallConv,
        flags: &settings::Flags,
        _sig: &Signature,
        regs: &[Writable<RealReg>],
        function_calls: FunctionCalls,
        incoming_args_size: u32,
        tail_args_size: u32,
        stackslots_size: u32,
        fixed_frame_storage_size: u32,
        outgoing_args_size: u32,
    ) -> FrameLayout {
        let is_callee_saved = |reg: &Writable<RealReg>| match call_conv {
            isa::CallConv::PreserveAll => true,
            _ => DEFAULT_CALLEE_SAVES.contains(reg.to_reg().into()),
        };
        let mut regs: Vec<Writable<RealReg>> =
            regs.iter().cloned().filter(is_callee_saved).collect();

        regs.sort_unstable();

        let clobber_size = compute_clobber_size(&regs);

        let setup_area_size = if flags.preserve_frame_pointers()
            || function_calls != FunctionCalls::None
            || incoming_args_size > 0
            || clobber_size > 0
            || fixed_frame_storage_size > 0
        {
            16 // FP + LR
        } else {
            0
        };

        FrameLayout {
            word_bytes: 8,
            incoming_args_size,
            tail_args_size,
            setup_area_size,
            clobber_size,
            fixed_frame_storage_size,
            stackslots_size,
            outgoing_args_size,
            clobbered_callee_saves: regs,
            function_calls,
        }
    }

    fn retval_temp_reg(_call_conv_of_callee: isa::CallConv) -> Writable<Reg> {
        Writable::from_reg(regs::spilltmp_reg2())
    }

    fn exception_payload_regs(_call_conv: isa::CallConv) -> &'static [Reg] {
        const PAYLOAD_REGS: &'static [Reg] = &[regs::r_reg(3), regs::r_reg(4)];
        PAYLOAD_REGS
    }
}

// Caller-saved GPRs (r3-r12); r0/r1/r2/r13 are reserved. Floats f0-f13 are
// all caller-saved. LR and CR are implicitly clobbered but not modelled as
// allocatable registers.
const DEFAULT_CLOBBERS: PRegSet = PRegSet::empty()
    .with(pr_reg(3))
    .with(pr_reg(4))
    .with(pr_reg(5))
    .with(pr_reg(6))
    .with(pr_reg(7))
    .with(pr_reg(8))
    .with(pr_reg(9))
    .with(pr_reg(10))
    .with(pr_reg(11))
    .with(pr_reg(12));

const ALL_CLOBBERS: PRegSet = PRegSet::empty()
    .with(pr_reg(0))
    .with(pr_reg(1))
    .with(pr_reg(2))
    .with(pr_reg(3))
    .with(pr_reg(4))
    .with(pr_reg(5))
    .with(pr_reg(6))
    .with(pr_reg(7))
    .with(pr_reg(8))
    .with(pr_reg(9))
    .with(pr_reg(10))
    .with(pr_reg(11))
    .with(pr_reg(12))
    .with(pr_reg(13))
    .with(pr_reg(14))
    .with(pr_reg(15))
    .with(pr_reg(16))
    .with(pr_reg(17))
    .with(pr_reg(18))
    .with(pr_reg(19))
    .with(pr_reg(20))
    .with(pr_reg(21))
    .with(pr_reg(22))
    .with(pr_reg(23))
    .with(pr_reg(24))
    .with(pr_reg(25))
    .with(pr_reg(26))
    .with(pr_reg(27))
    .with(pr_reg(28))
    .with(pr_reg(29))
    .with(pr_reg(30))
    .with(pr_reg(31))
    .with(pf_reg(0))
    .with(pf_reg(1))
    .with(pf_reg(2))
    .with(pf_reg(3))
    .with(pf_reg(4))
    .with(pf_reg(5))
    .with(pf_reg(6))
    .with(pf_reg(7))
    .with(pf_reg(8))
    .with(pf_reg(9))
    .with(pf_reg(10))
    .with(pf_reg(11))
    .with(pf_reg(12))
    .with(pf_reg(13))
    .with(pf_reg(14))
    .with(pf_reg(15))
    .with(pf_reg(16))
    .with(pf_reg(17))
    .with(pf_reg(18))
    .with(pf_reg(19))
    .with(pf_reg(20))
    .with(pf_reg(21))
    .with(pf_reg(22))
    .with(pf_reg(23))
    .with(pf_reg(24))
    .with(pf_reg(25))
    .with(pf_reg(26))
    .with(pf_reg(27))
    .with(pf_reg(28))
    .with(pf_reg(29))
    .with(pf_reg(30))
    .with(pf_reg(31));

const NO_CLOBBERS: PRegSet = PRegSet::empty();

// r14-r31 and f14-f31 are callee-saved per the PPC64 SysV ABI.
const DEFAULT_CALLEE_SAVES: PRegSet = PRegSet::empty()
    .with(pr_reg(14))
    .with(pr_reg(15))
    .with(pr_reg(16))
    .with(pr_reg(17))
    .with(pr_reg(18))
    .with(pr_reg(19))
    .with(pr_reg(20))
    .with(pr_reg(21))
    .with(pr_reg(22))
    .with(pr_reg(23))
    .with(pr_reg(24))
    .with(pr_reg(25))
    .with(pr_reg(26))
    .with(pr_reg(27))
    .with(pr_reg(28))
    .with(pr_reg(29))
    .with(pr_reg(30))
    .with(pr_reg(31))
    .with(pf_reg(14))
    .with(pf_reg(15))
    .with(pf_reg(16))
    .with(pf_reg(17))
    .with(pf_reg(18))
    .with(pf_reg(19))
    .with(pf_reg(20))
    .with(pf_reg(21))
    .with(pf_reg(22))
    .with(pf_reg(23))
    .with(pf_reg(24))
    .with(pf_reg(25))
    .with(pf_reg(26))
    .with(pf_reg(27))
    .with(pf_reg(28))
    .with(pf_reg(29))
    .with(pf_reg(30))
    .with(pf_reg(31));

fn compute_clobber_size(clobbers: &[Writable<RealReg>]) -> u32 {
    let mut clobbered_size = 0;
    for reg in clobbers {
        match reg.to_reg().class() {
            RegClass::Int => {
                clobbered_size += 8;
            }
            RegClass::Float => {
                clobbered_size += 8;
            }
            RegClass::Vector => {
                clobbered_size = align_to(clobbered_size, 16);
                clobbered_size += 16;
            }
        }
    }
    // Always reserve 64 bytes at the bottom of the frame: 16 bytes for
    // GPR<->FPR bit moves plus three 16-byte vector scratch areas used
    // by lane insert/extract/splat/blend/reduction sequences.
    align_to(clobbered_size, 16) + 64
}

const fn create_reg_environment() -> MachineEnv {
    // Preferred: argument registers r3-r10 / f1-f13.
    // Non-preferred: remaining caller-saved temps plus callee-saved regs.
    // Reserved (not listed anywhere): r0, r1 (sp), r2 (toc), r13 (tls).

    let int_pref = PRegSet::empty()
        .with(pr_reg(3))
        .with(pr_reg(4))
        .with(pr_reg(5))
        .with(pr_reg(6))
        .with(pr_reg(7))
        .with(pr_reg(8))
        .with(pr_reg(9))
        .with(pr_reg(10));

    let int_nonpref = PRegSet::empty()
        .with(pr_reg(11))
        .with(pr_reg(12))
        .with(pr_reg(14))
        .with(pr_reg(15))
        .with(pr_reg(16))
        .with(pr_reg(17))
        .with(pr_reg(18))
        .with(pr_reg(19))
        .with(pr_reg(20))
        .with(pr_reg(21))
        .with(pr_reg(22))
        .with(pr_reg(23))
        .with(pr_reg(24))
        .with(pr_reg(25))
        .with(pr_reg(26))
        .with(pr_reg(27))
        .with(pr_reg(28))
        .with(pr_reg(29))
        .with(pr_reg(30))
        .with(pr_reg(31));

    let float_pref = PRegSet::empty()
        .with(pf_reg(0))
        .with(pf_reg(1))
        .with(pf_reg(2))
        .with(pf_reg(3))
        .with(pf_reg(4))
        .with(pf_reg(5))
        .with(pf_reg(6))
        .with(pf_reg(7))
        .with(pf_reg(8))
        .with(pf_reg(9))
        .with(pf_reg(10))
        .with(pf_reg(11))
        .with(pf_reg(12))
        .with(pf_reg(13));

    let float_nonpref = PRegSet::empty()
        .with(pf_reg(14))
        .with(pf_reg(15))
        .with(pf_reg(16))
        .with(pf_reg(17))
        .with(pf_reg(18))
        .with(pf_reg(19))
        .with(pf_reg(20))
        .with(pf_reg(21))
        .with(pf_reg(22))
        .with(pf_reg(23))
        .with(pf_reg(24))
        .with(pf_reg(25))
        .with(pf_reg(26))
        .with(pf_reg(27))
        .with(pf_reg(28))
        .with(pf_reg(29))
        .with(pf_reg(30))
        .with(pf_reg(31));

    let vector_pref = PRegSet::empty()
        .with(pv_reg(2))
        .with(pv_reg(3))
        .with(pv_reg(4))
        .with(pv_reg(5))
        .with(pv_reg(6))
        .with(pv_reg(7))
        .with(pv_reg(8))
        .with(pv_reg(9))
        .with(pv_reg(10))
        .with(pv_reg(11))
        .with(pv_reg(12))
        .with(pv_reg(13));

    let vector_nonpref = PRegSet::empty()
        .with(pv_reg(0))
        .with(pv_reg(1))
        .with(pv_reg(14))
        .with(pv_reg(15))
        .with(pv_reg(16))
        .with(pv_reg(17))
        .with(pv_reg(18))
        .with(pv_reg(19))
        .with(pv_reg(20))
        .with(pv_reg(21))
        .with(pv_reg(22))
        .with(pv_reg(23))
        .with(pv_reg(24))
        .with(pv_reg(25))
        .with(pv_reg(26))
        .with(pv_reg(27))
        .with(pv_reg(28))
        .with(pv_reg(29))
        .with(pv_reg(30))
        .with(pv_reg(31));

    MachineEnv {
        preferred_regs_by_class: [int_pref, float_pref, vector_pref],
        non_preferred_regs_by_class: [int_nonpref, float_nonpref, vector_nonpref],
        fixed_stack_slots: vec![],
        scratch_by_class: [None, None, None],
    }
}
