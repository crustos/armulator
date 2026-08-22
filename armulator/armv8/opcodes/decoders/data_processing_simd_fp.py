"""
Data processing - scalar floating point and Advanced SIMD.

Selected by op0 = x111 at instr[28:25]. instr[28] then separates the two halves: set for
the scalar floating point instructions, clear for Advanced SIMD.

Within the scalar floating point half the classes are told apart by how much of
instr[15:10] is fixed, so the tests must run from most specific to least:

    instr[24] = 1                       Data processing (3 source): FMADD and friends
    instr[21] = 0                       Conversion between floating point and fixed point
    instr[15:10] = 000000               Conversion between floating point and integer
    instr[14:10] = 10000                Data processing (1 source)
    instr[13:10] = 1000                 Compare
    instr[12:10] = 100                  Immediate
    instr[11:10] = 01                   Conditional compare
    instr[11:10] = 10                   Data processing (2 source)
    instr[11:10] = 11                   Conditional select
"""

from armulator.armv8.bits_ops import bit_at, substring
from armulator.armv8.opcodes.concrete.floating_point import (
    FpCompareA64,
    FpConditionalCompareA64,
    FpConditionalSelectA64,
    FpDataProcessing1SourceA64,
    FpDataProcessing2SourceA64,
    FpDataProcessing3SourceA64,
    FpImmediateA64,
    FpIntegerConvertA64,
)


def decode_instruction(instr):
    if bit_at(instr, 28) and not bit_at(instr, 30):
        return _decode_scalar_fp(instr)
    # instr[30] set alongside instr[28] is the Advanced SIMD scalar space, which shares
    # its shift-by-immediate encodings with the vector forms.
    return _decode_advanced_simd(instr)


def _decode_scalar_fp(instr):
    # The scalar floating point encodings all have instr[30] clear and S (instr[29])
    # clear. When instr[30] is set this is an Advanced SIMD scalar instruction - USHR,
    # scalar ADD and so on - which is a different space entirely. Without this guard
    # those encodings fall through into the floating point classes below and execute as
    # something else, which is far worse than not decoding them at all.
    if bit_at(instr, 30) or bit_at(instr, 29):
        return None

    if bit_at(instr, 24):
        return FpDataProcessing3SourceA64

    if not bit_at(instr, 21):
        # Conversion between floating point and fixed point is not modelled; the
        # integer conversions below cover what compilers actually emit.
        return None

    low = substring(instr, 15, 10)
    if low == 0b000000:
        return FpIntegerConvertA64
    if substring(instr, 14, 10) == 0b10000:
        return FpDataProcessing1SourceA64
    if substring(instr, 13, 10) == 0b1000:
        return FpCompareA64
    if substring(instr, 12, 10) == 0b100:
        return FpImmediateA64

    tail = substring(instr, 11, 10)
    if tail == 0b01:
        return FpConditionalCompareA64
    if tail == 0b10:
        return FpDataProcessing2SourceA64
    if tail == 0b11:
        return FpConditionalSelectA64
    return None


def _decode_advanced_simd(instr):
    # Filled in by the Advanced SIMD decoder.
    from armulator.armv8.opcodes.decoders import advanced_simd
    return advanced_simd.decode_instruction(instr)
