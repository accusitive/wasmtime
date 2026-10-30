use cranelift_assembler_glasm::encode_module;

#[test]
fn encodes_att_instructions() {
    let bytes = encode_module("movl %eax, %eax", true).unwrap();
    assert_eq!(bytes, vec![0x89, 0xc0]);

    let bytes = encode_module("movl $42, %eax", true).unwrap();
    assert_eq!(bytes, vec![0xb8, 42, 0, 0, 0]);

    let bytes = encode_module("movq %rdi, %rax\naddq $5, %rax", true).unwrap();
    assert_eq!(bytes, vec![0x48, 0x89, 0xf8, 0x48, 0x83, 0xc0, 0x05]);
}

#[test]
fn encodes_intel_instructions() {
    let bytes = encode_module("mov eax, eax", false).unwrap();
    assert_eq!(bytes, vec![0x89, 0xc0]);

    // nasm-style optimization: `mov r10d, 42` (zero-extension is identical).
    let bytes = encode_module("mov r10, 42", false).unwrap();
    assert_eq!(bytes, vec![0x41, 0xba, 42, 0, 0, 0]);
}

#[test]
fn rejects_symbols() {
    assert!(encode_module("movl foo, %eax", true).is_err());
}
