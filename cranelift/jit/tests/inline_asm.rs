use cranelift_codegen::ir::types::I64;
use cranelift_codegen::ir::{AbiParam, InstBuilder, Signature};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::Context;
use cranelift_frontend::*;
use cranelift_jit::*;
use cranelift_module::*;

fn module() -> JITModule {
    let mut flag_builder = settings::builder();
    flag_builder.set("is_pic", "true").unwrap();
    let isa = cranelift_native::builder()
        .unwrap()
        .finish(settings::Flags::new(flag_builder))
        .unwrap();
    JITModule::new(JITBuilder::with_isa(isa, default_libcall_names()))
}

fn finalize(builder: FunctionBuilder) {
    let mut builder = builder;
    builder.seal_all_blocks();
    builder.finalize(cranelift_codegen::isa::TargetFrontendConfig {
        default_call_conv: CallConv::SystemV,
        pointer_width: target_lexicon::PointerWidth::U64,
        page_size_align_log2: 12,
    });
}

/// `"=r"` output and `"r"` input: the asm increments its input.
#[test]
fn inline_asm_reg_operands() {
    if cfg!(not(target_arch = "x86_64")) {
        return;
    }
    let mut module = module();
    let sig = Signature {
        params: vec![AbiParam::new(I64)],
        returns: vec![AbiParam::new(I64)],
        call_conv: CallConv::SystemV,
    };
    let func_id = module
        .declare_function("inc", Linkage::Local, &sig)
        .unwrap();

    let mut ctx = Context::new();
    ctx.func.signature = sig;
    let mut builder_ctx = FunctionBuilderContext::new();
    {
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);
        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        builder.seal_block(entry);
        let x = builder.block_params(entry)[0];

        let contents = cranelift_codegen::ir::InlineAsmContents::new(
            "movq %1, %0\n\taddq $1, %0",
            /* att_syntax = */ true,
        )
        .operand("=r")
        .operand("r");
        let asm = builder.create_inline_asm(contents);
        let inst = builder.ins().inline_asm(asm, &[x]);
        let out = builder.inst_results(inst)[0];
        builder.ins().return_(&[out]);
        finalize(builder);
    }

    module.define_function(func_id, &mut ctx).unwrap();
    module.finalize_definitions().unwrap();
    module.clear_context(&mut ctx);

    let f: extern "C" fn(i64) -> i64 =
        unsafe { core::mem::transmute(module.get_finalized_function(func_id)) };
    assert_eq!(f(41), 42);
    assert_eq!(f(0), 1);

    unsafe {
        module.free_memory();
    }
}

/// `"+a"`: an inout operand pinned to %rax by the constraint letter. The
/// input is copied into %rax, the asm doubles it in place.
#[test]
fn inline_asm_fixed_inout() {
    if cfg!(not(target_arch = "x86_64")) {
        return;
    }
    let mut module = module();
    let sig = Signature {
        params: vec![AbiParam::new(I64)],
        returns: vec![AbiParam::new(I64)],
        call_conv: CallConv::SystemV,
    };
    let func_id = module
        .declare_function("double", Linkage::Local, &sig)
        .unwrap();

    let mut ctx = Context::new();
    ctx.func.signature = sig;
    let mut builder_ctx = FunctionBuilderContext::new();
    {
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);
        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        builder.seal_block(entry);
        let x = builder.block_params(entry)[0];

        let contents = cranelift_codegen::ir::InlineAsmContents::new("addq %0, %0", true)
            .operand("+a");
        let asm = builder.create_inline_asm(contents);
        let inst = builder.ins().inline_asm(asm, &[x]);
        let out = builder.inst_results(inst)[0];
        builder.ins().return_(&[out]);
        finalize(builder);
    }

    module.define_function(func_id, &mut ctx).unwrap();
    module.finalize_definitions().unwrap();
    module.clear_context(&mut ctx);

    let f: extern "C" fn(i64) -> i64 =
        unsafe { core::mem::transmute(module.get_finalized_function(func_id)) };
    assert_eq!(f(21), 42);

    unsafe {
        module.free_memory();
    }
}

/// An explicit clobber: a value live across the asm must survive even
/// though the asm destroys the registers the allocator would naturally
/// pick (%rax, %r10, %r11 are all fair game).
#[test]
fn inline_asm_clobber_preserves_live_values() {
    if cfg!(not(target_arch = "x86_64")) {
        return;
    }
    let mut module = module();
    let sig = Signature {
        params: vec![AbiParam::new(I64)],
        returns: vec![AbiParam::new(I64)],
        call_conv: CallConv::SystemV,
    };
    let func_id = module
        .declare_function("keep", Linkage::Local, &sig)
        .unwrap();

    let mut ctx = Context::new();
    ctx.func.signature = sig;
    let mut builder_ctx = FunctionBuilderContext::new();
    {
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);
        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        builder.seal_block(entry);
        let x = builder.block_params(entry)[0];

        let contents = cranelift_codegen::ir::InlineAsmContents::new(
            "movq $7, %rax\n\tmovq $7, %r10\n\tmovq $7, %r11",
            true,
        )
        .clobber("rax")
        .clobber("r10")
        .clobber("r11");
        let asm = builder.create_inline_asm(contents);
        builder.ins().inline_asm(asm, &[]);

        let sum = builder.ins().iadd(x, x);
        builder.ins().return_(&[sum]);
        finalize(builder);
    }

    module.define_function(func_id, &mut ctx).unwrap();
    module.finalize_definitions().unwrap();
    module.clear_context(&mut ctx);

    let f: extern "C" fn(i64) -> i64 =
        unsafe { core::mem::transmute(module.get_finalized_function(func_id)) };
    assert_eq!(f(21), 42);

    unsafe {
        module.free_memory();
    }
}

/// `"i"`: an immediate constant substituted into the template.
#[test]
fn inline_asm_immediate_operand() {
    if cfg!(not(target_arch = "x86_64")) {
        return;
    }
    let mut module = module();
    let sig = Signature {
        params: vec![AbiParam::new(I64)],
        returns: vec![AbiParam::new(I64)],
        call_conv: CallConv::SystemV,
    };
    let func_id = module
        .declare_function("add42", Linkage::Local, &sig)
        .unwrap();

    let mut ctx = Context::new();
    ctx.func.signature = sig;
    let mut builder_ctx = FunctionBuilderContext::new();
    {
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);
        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        builder.seal_block(entry);
        let x = builder.block_params(entry)[0];
        let imm = builder.ins().iconst(I64, 42);

        // %0 = output, %1 = x, %2 = the immediate.
        let contents =
            cranelift_codegen::ir::InlineAsmContents::new("movq %1, %0\n\taddq %2, %0", true)
            .operand("=r")
            .operand("r")
            .operand("i");
        let asm = builder.create_inline_asm(contents);
        let inst = builder.ins().inline_asm(asm, &[x, imm]);
        let out = builder.inst_results(inst)[0];
        builder.ins().return_(&[out]);
        finalize(builder);
    }

    module.define_function(func_id, &mut ctx).unwrap();
    module.finalize_definitions().unwrap();
    module.clear_context(&mut ctx);

    let f: extern "C" fn(i64) -> i64 =
        unsafe { core::mem::transmute(module.get_finalized_function(func_id)) };
    assert_eq!(f(0), 42);

    unsafe {
        module.free_memory();
    }
}

/// `"m"`: memory at the address held in the input value.
#[test]
fn inline_asm_memory_operand() {
    if cfg!(not(target_arch = "x86_64")) {
        return;
    }
    let mut module = module();
    let sig = Signature {
        params: vec![AbiParam::new(I64)],
        returns: vec![AbiParam::new(I64)],
        call_conv: CallConv::SystemV,
    };
    let func_id = module
        .declare_function("load", Linkage::Local, &sig)
        .unwrap();

    let mut ctx = Context::new();
    ctx.func.signature = sig;
    let mut builder_ctx = FunctionBuilderContext::new();
    {
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);
        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        builder.seal_block(entry);
        let ptr = builder.block_params(entry)[0];

        // Loads the i64 at the given address: `movq (%1), %0`.
        let contents = cranelift_codegen::ir::InlineAsmContents::new("movq %1, %0", true)
            .operand("=r")
            .operand("m");
        let asm = builder.create_inline_asm(contents);
        let inst = builder.ins().inline_asm(asm, &[ptr]);
        let out = builder.inst_results(inst)[0];
        builder.ins().return_(&[out]);
        finalize(builder);
    }

    module.define_function(func_id, &mut ctx).unwrap();
    module.finalize_definitions().unwrap();
    module.clear_context(&mut ctx);

    let f: extern "C" fn(i64) -> i64 =
        unsafe { core::mem::transmute(module.get_finalized_function(func_id)) };
    let value: i64 = 0x1234_5678;
    assert_eq!(
        f(core::ptr::addr_of!(value) as i64),
        0x1234_5678
    );

    unsafe {
        module.free_memory();
    }
}

/// Local labels inside the template: a counted loop with a clobbered
/// counter register.
#[test]
fn inline_asm_local_labels() {
    if cfg!(not(target_arch = "x86_64")) {
        return;
    }
    let mut module = module();
    let sig = Signature {
        params: vec![AbiParam::new(I64)],
        returns: vec![AbiParam::new(I64)],
        call_conv: CallConv::SystemV,
    };
    let func_id = module
        .declare_function("loop10", Linkage::Local, &sig)
        .unwrap();

    let mut ctx = Context::new();
    ctx.func.signature = sig;
    let mut builder_ctx = FunctionBuilderContext::new();
    {
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);
        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        builder.seal_block(entry);
        let x = builder.block_params(entry)[0];
        let count = builder.ins().iconst(I64, 10);

        let contents = cranelift_codegen::ir::InlineAsmContents::new(
            "movq %1, %0\n\tmovq %2, %rcx\n1: addq $1, %0\n decq %rcx\n jne 1b",
            true,
        )
        .operand("=r")
        .operand("r")
        .operand("r")
        .clobber("rcx");
        let asm = builder.create_inline_asm(contents);
        let inst = builder.ins().inline_asm(asm, &[x, count]);
        let out = builder.inst_results(inst)[0];
        builder.ins().return_(&[out]);
        finalize(builder);
    }

    module.define_function(func_id, &mut ctx).unwrap();
    module.finalize_definitions().unwrap();
    module.clear_context(&mut ctx);

    let f: extern "C" fn(i64) -> i64 =
        unsafe { core::mem::transmute(module.get_finalized_function(func_id)) };
    assert_eq!(f(32), 42);

    unsafe {
        module.free_memory();
    }
}

/// `"+r"`: a non-fixed inout -- the allocator picks one register for the
/// read and the write, and the input value survives for other uses.
#[test]
fn inline_asm_nonfixed_inout() {
    if cfg!(not(target_arch = "x86_64")) {
        return;
    }
    let mut module = module();
    let sig = Signature {
        params: vec![AbiParam::new(I64)],
        returns: vec![AbiParam::new(I64)],
        call_conv: CallConv::SystemV,
    };
    let func_id = module
        .declare_function("shl1", Linkage::Local, &sig)
        .unwrap();

    let mut ctx = Context::new();
    ctx.func.signature = sig;
    let mut builder_ctx = FunctionBuilderContext::new();
    {
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);
        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        builder.seal_block(entry);
        let x = builder.block_params(entry)[0];

        let contents =
            cranelift_codegen::ir::InlineAsmContents::new("addq %0, %0", true).operand("+r");
        let asm = builder.create_inline_asm(contents);
        let inst = builder.ins().inline_asm(asm, &[x]);
        let out = builder.inst_results(inst)[0];
        // x is still live after the asm: the result must be 3*x if the
        // allocator preserved the original value's register.
        let sum = builder.ins().iadd(out, x);
        builder.ins().return_(&[sum]);
        finalize(builder);
    }

    module.define_function(func_id, &mut ctx).unwrap();
    module.finalize_definitions().unwrap();
    module.clear_context(&mut ctx);

    let f: extern "C" fn(i64) -> i64 =
        unsafe { core::mem::transmute(module.get_finalized_function(func_id)) };
    assert_eq!(f(10), 30);

    unsafe {
        module.free_memory();
    }
}
