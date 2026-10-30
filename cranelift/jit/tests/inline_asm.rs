use cranelift_codegen::ir::immediates::Imm64;
use cranelift_codegen::ir::{AbiParam, InstBuilder, Signature};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::Context;
use cranelift_frontend::*;
use cranelift_jit::*;
use cranelift_module::*;

#[test]
fn inline_asm_executes() {
    if cfg!(not(target_arch = "x86_64")) {
        return;
    }

    let mut flag_builder = settings::builder();
    flag_builder.set("is_pic", "true").unwrap();
    let isa = cranelift_native::builder()
        .unwrap()
        .finish(settings::Flags::new(flag_builder))
        .unwrap();
    let mut module = JITModule::new(JITBuilder::with_isa(isa, default_libcall_names()));

    let sig = Signature {
        params: vec![AbiParam::new(cranelift_codegen::ir::types::I64)],
        returns: vec![],
        call_conv: CallConv::SystemV,
    };
    let func_id = module
        .declare_function("poke", Linkage::Local, &sig)
        .unwrap();

    let mut ctx = Context::new();
    ctx.func.signature = sig;
    let mut builder_ctx = FunctionBuilderContext::new();
    {
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);
        let entry = builder.create_block();
        builder.append_block_param(entry, cranelift_codegen::ir::types::I64);
        builder.switch_to_block(entry);
        builder.seal_block(entry);
        let ptr = builder.block_params(entry)[0];

        // mov dword [rdi], 42 (c7 47 00 2a 00 00 00), padded with one NOP;
        // RDI holds the pointer argument per the System V ABI. The store is
        // observable from the host after the call returns.
        builder
            .ins()
            .inline_asm(Imm64::new(0x9000_0000_2a00_47c7u64 as i64), Imm64::new(0));
        builder.ins().return_(&[]);
        builder.seal_all_blocks();
        builder.finalize(cranelift_codegen::isa::TargetFrontendConfig {
            default_call_conv: CallConv::SystemV,
            pointer_width: target_lexicon::PointerWidth::U64,
            page_size_align_log2: 12,
        });
    }

    module.define_function(func_id, &mut ctx).unwrap();
    module.finalize_definitions().unwrap();
    module.clear_context(&mut ctx);

    let code_ptr = module.get_finalized_function(func_id);
    let poke: extern "C" fn(i64) = unsafe { core::mem::transmute(code_ptr) };

    let mut slot: i64 = 0;
    poke(core::ptr::addr_of_mut!(slot) as i64);
    assert_eq!(slot, 42);

    unsafe {
        module.free_memory();
    }
}
