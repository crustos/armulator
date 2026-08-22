"""
Advanced SIMD.

Reached from the SIMD/FP group when instr[28] is clear. The classes modelled here are
told apart by instr[24:21] together with the low bits of the opcode field:

    instr[24] = 1, instr[21] = 1, instr[10] = 1     Three same
    instr[24] = 1, instr[21] = 0, instr[10] = 1     Modified immediate
    instr[24] = 0, instr[21] = 0, instr[10] = 1     Copy: DUP, INS, UMOV, SMOV

Everything else in the Advanced SIMD space - the structure loads and stores, pairwise
and across-lane reductions, saturating and widening arithmetic, table lookups and the
by-element forms - is not modelled and decodes as undefined.
"""

from armulator.armv8.bits_ops import bit_at, substring
from armulator.armv8.opcodes.concrete.advanced_simd import (
    SimdBitwiseA64,
    SimdDuplicateA64,
    SimdExtractA64,
    SimdImmediateA64,
    SimdAcrossLanesA64,
    SimdByElementA64,
    SimdInsertA64,
    SimdPairwiseA64,
    SimdShiftImmediateA64,
    SimdThreeSameA64,
)


def decode_instruction(instr):
    # The modelled classes all live in the 0x0E/0x2E/0x4E/0x6E encoding space, which
    # requires instr[27:25] = 111 and instr[31] clear.
    if bit_at(instr, 31) or substring(instr, 27, 25) != 0b111:
        return None

    if bit_at(instr, 24):
        # The 1111 space: modified immediate, shift by immediate, by element.
        if not bit_at(instr, 10):
            # The by-element forms are the only ones here with instr[10] clear.
            return SimdByElementA64
        if substring(instr, 22, 19) == 0b0000:
            # immh of zero distinguishes the modified immediates from the shifts.
            if substring(instr, 20, 19) == 0b00 and not bit_at(instr, 28):
                return SimdImmediateA64
            return None
        return SimdShiftImmediateA64

    # The 1110 space.
    #
    # Across-lane reductions are tested first because they sit at instr[11:10] = 10 -
    # bit 10 is *clear* - while the three-same and copy forms below both require it set.
    # Checking bit 10 first would reject every reduction before it was ever considered.
    if substring(instr, 21, 17) == 0b11000 and substring(instr, 11, 10) == 0b10:
        return SimdAcrossLanesA64

    # instr[10] set is what separates the remaining modelled classes from the
    # two-register miscellaneous group.
    if not bit_at(instr, 10):
        return None

    if bit_at(instr, 21):
        return _decode_three_same(instr)

    if substring(instr, 20, 16) != 0 and not bit_at(instr, 15):
        return _decode_copy(instr)

    return None


def _decode_three_same(instr):
    """
    The bitwise operations share the opcode 00011 with nothing else, so they are split
    out before the arithmetic; they alone ignore the lane width.
    """
    if substring(instr, 15, 11) == 0b00011:
        return SimdBitwiseA64
    if substring(instr, 15, 11) == 0b10111:
        # ADDP folds adjacent lanes rather than working lane against lane, so it needs
        # its own implementation despite living in the three-same encoding space.
        return SimdPairwiseA64
    return SimdThreeSameA64


def _decode_copy(instr):
    """
    DUP, INS, UMOV and SMOV are separated by imm4 at instr[14:11].
    """
    imm4 = substring(instr, 14, 11)
    op = bit_at(instr, 29)

    if op:
        # INS (element) - only the 128-bit form exists.
        return SimdInsertA64
    if imm4 == 0b0000:
        return SimdDuplicateA64          # DUP (element)
    if imm4 == 0b0001:
        return SimdDuplicateA64          # DUP (general)
    if imm4 == 0b0011:
        return SimdInsertA64             # INS (general)
    if imm4 in (0b0101, 0b0111):
        return SimdExtractA64            # SMOV / UMOV
    return None
