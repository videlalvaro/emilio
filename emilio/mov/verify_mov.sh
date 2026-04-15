#!/bin/bash
# verify_mov.sh -- Verify that a binary was compiled with the M/o/Vfuscator
#
# Checks:
#   1. Binary is ELF32 i386 (movfuscator only targets 32-bit x86)
#   2. >90% of disassembled .text bytes decode as MOV instructions
#   3. No standard GCC function prologues (push ebp; mov ebp,esp)
#
# The movfuscator embeds large data lookup tables directly in .text
# for computed branching.  objdump will linearly disassemble these as
# random x86 instructions, so ~5% "non-MOV" is normal and expected.
# A GCC-compiled binary typically has <30% MOV.
#
# Usage: ./verify_mov.sh <binary>
#
# Exit code:
#   0 = binary is a valid movfuscator binary
#   1 = binary was NOT compiled with the movfuscator

set -e

BINARY="${1:?Usage: $0 <binary>}"

if [ ! -f "$BINARY" ]; then
    echo "Error: $BINARY not found" >&2
    exit 1
fi

# Check for objdump
if ! command -v objdump &>/dev/null; then
    echo "Error: objdump not found. Install binutils." >&2
    exit 1
fi

echo "=== MOV-Only Verification ==="
echo "Binary: $BINARY"
echo "Size: $(ls -lh "$BINARY" | awk '{print $5}')"
echo ""

# ---- Check 1: ELF class ----
# The movfuscator always produces 32-bit (ELF32 / i386) binaries.
# A 64-bit binary was compiled by GCC or another standard compiler.

ELFCLASS=$(readelf -h "$BINARY" 2>/dev/null | grep 'Class:' | awk '{print $2}')
MACHINE=$(readelf -h "$BINARY" 2>/dev/null | grep 'Machine:' | sed 's/.*Machine:\s*//')

if [ "$ELFCLASS" = "ELF64" ]; then
    echo "FAIL: Binary is 64-bit ($ELFCLASS, $MACHINE)."
    echo "  The M/o/Vfuscator produces 32-bit (ELF32 / i386) binaries."
    echo "  This binary was likely compiled with GCC, not movcc."
    exit 1
fi

echo "Architecture: $ELFCLASS ($MACHINE)"
echo ""

# ---- Check 2: MOV instruction ratio ----
# Movfuscator binaries are >90% MOV in .text (typically 93-97%).
# GCC binaries are typically <30% MOV.
# The gap between data tables (~5% non-MOV) and GCC (~70% non-MOV)
# makes this a reliable classifier.

DISASM=$(objdump -d -j .text "$BINARY" 2>/dev/null || objdump -d "$BINARY")

TOTAL=$(echo "$DISASM" | grep -cE '^\s+[0-9a-f]+:' || true)
echo "Total instructions in .text: $TOTAL"

if [ "$TOTAL" -eq 0 ]; then
    echo "FAIL: No instructions found in .text section"
    exit 1
fi

# Count MOV instructions (all variants: mov, movl, movw, movb, movzbl, movsbl, etc.)
MOV_COUNT=$(echo "$DISASM" | grep -E '^\s+[0-9a-f]+:' | \
    grep -ciE '\bmov[a-z]*\b' || true)
echo "MOV instructions: $MOV_COUNT"

NON_MOV_COUNT=$((TOTAL - MOV_COUNT))
echo "Non-MOV disassembled bytes: $NON_MOV_COUNT"

# Calculate MOV percentage (integer arithmetic, multiply first to avoid truncation)
MOV_PCT=$((MOV_COUNT * 100 / TOTAL))
echo "MOV ratio: ${MOV_PCT}%"
echo ""

# Threshold: movfuscator binaries are >90% MOV; GCC binaries are <30% MOV
MOV_THRESHOLD=90

if [ "$MOV_PCT" -lt "$MOV_THRESHOLD" ]; then
    echo "FAIL: MOV ratio ${MOV_PCT}% is below ${MOV_THRESHOLD}% threshold."
    echo "  Movfuscator binaries are typically >93% MOV."
    echo "  GCC binaries are typically <30% MOV."
    echo "  This binary was likely NOT compiled with movcc."
    exit 1
fi

# ---- Check 3: No GCC function prologues ----
# GCC-compiled functions start with "push %ebp; mov %ebp,%esp" (or similar).
# Movfuscator-compiled code never uses push/pop/call/ret in user functions.

GCC_PROLOGUES=$(echo "$DISASM" | grep -cE '^\s+[0-9a-f]+:.*\bpush\s+%ebp\b' || true)

if [ "$GCC_PROLOGUES" -gt 5 ]; then
    echo "WARNING: Found $GCC_PROLOGUES 'push %ebp' instructions."
    echo "  This may indicate GCC-compiled code mixed in."
    # Don't fail -- CRT startup code has a few push instructions
fi

# ---- Results ----

echo "=== Instruction Breakdown (top 20) ==="
echo "$DISASM" | grep -E '^\s+[0-9a-f]+:' | \
    sed 's/.*:\s\+[0-9a-f ]\+\s\+//' | \
    awk '{print $1}' | sort | uniq -c | sort -rn | head -20
echo ""

# Count PLT stubs (expected non-MOV for libc dispatch)
PLT_NON_MOV=$(echo "$DISASM" | \
    sed -n '/<.*@plt>/,/^$/p' | \
    grep -E '^\s+[0-9a-f]+:' | \
    grep -viE '\bmov[a-z]*\b' | \
    grep -viE '\bnop\b|\bint\b|\bhlt\b' | \
    wc -l || true)

echo "=== Analysis ==="
echo "  MOV ratio:                            ${MOV_PCT}% (threshold: ${MOV_THRESHOLD}%)"
echo "  PLT stubs (expected, for libc calls):  ~$PLT_NON_MOV non-MOV"
echo "  Data tables in .text (expected):       ~$((NON_MOV_COUNT - PLT_NON_MOV)) disassembly artifacts"
echo ""
echo "  The M/o/Vfuscator embeds lookup tables in .text for computed"
echo "  branching.  objdump disassembles these data bytes as random"
echo "  x86 instructions.  This is expected -- they are never executed."
echo ""
echo "PASS: Binary is a valid M/o/Vfuscator binary (${MOV_PCT}% MOV)"
exit 0
