#!/usr/bin/env python3
"""Mach-O arm64 assembly (the runtime's gates, thunks and stack switches) -> ELF assembly for Linux.

- C symbols lose Darwin's leading underscore (_runtime_gate -> runtime_gate); text inside quotes is left alone.
- sym@PAGE / sym@PAGEOFF become sym / :lo12:sym.
- tpidrro_el0 becomes tpidr_el0: Darwin keeps each thread's TSD base in TPIDRRO_EL0, glibc's thread pointer is
  TPIDR_EL0, and the host-stack slot (runtime_hle_tsd_off) is an offset from that pointer on Linux (runtime/hle_stack.c).
- mach_absolute_time becomes mng_mach_absolute_time (a real function in linux/platform.c).
- A non-executable stack note is added.
usage: macho_asm_to_elf.py in.S out.S
"""
import re
import sys

SYMBOL = re.compile(r'(?<![\w.$])_([A-Za-z][A-Za-z0-9_]*)')
PAGEOFF = re.compile(r'_([A-Za-z][A-Za-z0-9_]*)@PAGEOFF')
PAGE = re.compile(r'_([A-Za-z][A-Za-z0-9_]*)@PAGE\b')


def convert_code(code):
    code = PAGEOFF.sub(r':lo12:\1', code)
    code = PAGE.sub(r'\1', code)
    code = SYMBOL.sub(r'\1', code)
    code = code.replace('tpidrro_el0', 'tpidr_el0')
    code = re.sub(r'\bmach_absolute_time\b', 'mng_mach_absolute_time', code)
    code = re.sub(r'^(\s*)\.private_extern\b', r'\1.hidden', code)
    code = re.sub(r'^(\s*)\.cstring\b', r'\1.section .rodata.str1.1,"aMS",@progbits,1', code)
    code = re.sub(r'^(\s*)\.const_data\b', r'\1.section .data.rel.ro,"aw"', code)
    code = re.sub(r'^(\s*)\.const\b', r'\1.section .rodata', code)
    return code


def convert_line(line):
    # keep string literals and comments as they are
    out, i, n = [], 0, len(line)
    while i < n:
        if line.startswith('//', i):
            out.append(line[i:])
            break
        c = line[i]
        if c == '"':
            j = i + 1
            while j < n and line[j] != '"':
                j += 2 if line[j] == '\\' else 1
            out.append(line[i:j + 1])
            i = j + 1
            continue
        j = i
        while j < n and line[j] != '"' and not line.startswith('//', j):
            j += 1
        out.append(convert_code(line[i:j]))
        i = j
    return ''.join(out)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    text = open(src, encoding='utf-8').read()
    lines = [convert_line(l) for l in text.split('\n')]
    body = '\n'.join(lines)
    if '.note.GNU-stack' not in body:
        body = body.rstrip('\n') + '\n    .section .note.GNU-stack,"",%progbits\n'
    open(dst, 'w', encoding='utf-8').write(body)


if __name__ == '__main__':
    main()
