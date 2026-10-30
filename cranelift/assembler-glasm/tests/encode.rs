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

#[test]
fn resolves_local_labels_both_directions() {
    // Loop: %ecx counts down to zero.
    let bytes = encode_module(
        "xorl %ecx, %ecx\n1: incl %ecx\n cmpl $10, %ecx\n jne 1b",
        true,
    )
    .unwrap();
    // nasm picks the short `83 /7` form for the compare; the backward
    // `jne` displacement is -7 (label at 2, field at 8).
    assert_eq!(bytes, vec![0x31, 0xc9, 0xff, 0xc1, 0x83, 0xf9, 10, 0x75, 0xf9]);

    // Forward jump.
    let bytes = encode_module("jmp 1f\n nop\n1: nop", true).unwrap();
    assert_eq!(bytes, vec![0xeb, 0x01, 0x90, 0x90]);
}



