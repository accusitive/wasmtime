#!/usr/bin/env python3
"""Build a batched PPU test .self: CLIF cases -> cranelift bytes -> C harness."""
import subprocess, sys, os, re

CLIF_UTIL = "/home/gl/data/p/ps3game/wasmtime/target/debug/clif-util"

def compile_clif(path):
    out = subprocess.run([CLIF_UTIL, "compile", "-D", "--target",
                          "powerpc64-unknown-linux-gnu", path],
                         capture_output=True, text=True)
    m = re.search(r"^\.byte (.+)$", out.stdout, re.M)
    if not m:
        raise RuntimeError(f"compile failed for {path}:\n{out.stderr}")
    return m.group(1)

def main():
    manifest = sys.argv[1]   # lines: <clif-path> <expected-hex>
    outdir = sys.argv[2]
    os.makedirs(outdir, exist_ok=True)

    cases = []
    with open(manifest) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            path, expected = line.split()
            cases.append((path, int(expected, 16)))

    # Generate assembly with one exported function per case.
    # Each function gets an LV2 compact function descriptor so C code can
    # call it through the normal ELFv1 dot-symbol convention.
    parts = []
    for i, (path, _) in enumerate(cases):
        blob = compile_clif(path)
        d = {
            "i": i,
            "blob": blob,
            "opd": '.section ".opd","aw"',
            "toc": '.section ".toc","aw"',
        }
        parts.append(
            '\t.section ".opd","aw"\n'
            '\t.align 2\n'
            '\t.globl tf{i}\n'
            'tf{i}:\n'
            '\t.long .tf{i},.TOC.@tocbase32\n'
            '\t.section ".text"\n'
            '\t.align 2\n'
            '\t.type .tf{i},@function\n'
            '\t.globl .tf{i}\n'
            '.tf{i}:\n'
            '\t.byte {blob}\n'
            '\tblr'.format(**d)
        )
    open(os.path.join(outdir, "testfns.s"), "w").write("\n".join(parts) + "\n")

    # Generate the C driver.
    decls = "\n".join(f"extern unsigned long long tf{i}(void);" for i in range(len(cases)))
    checks = "\n".join(
        f'    check({i}, tf{i}(), 0x{expected & 0xFFFFFFFFFFFFFFFF:x}ull);'
        for i, (_, expected) in enumerate(cases))
    c = f"""#include <stdio.h>

{decls}

static int fails;

static void check(int id, unsigned long long got, unsigned long long want) {{
    if (got != want) {{
        fails++;
        printf("FAIL %d: got=%016llx want=%016llx\\n", id, got, want);
    }} else {{
        printf("PASS %d\\n", id);
    }}
    fflush(stdout);
}}

int main(void) {{
{checks}
    if (fails) printf("PPU_TESTS_FAILED %d\\n", fails);
    else printf("PPU_TESTS_PASSED\\n");
    fflush(stdout);
    return fails > 250 ? 250 : fails;
}}
"""
    open(os.path.join(outdir, "main.c"), "w").write(c)
    print(f"{len(cases)} cases staged in {outdir}")

if __name__ == "__main__":
    main()
