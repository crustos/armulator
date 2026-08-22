"""
Loads and stores.

Selected by op0 = x1x0 at instr[28:25]. Within the group the useful discriminator is
instr[29:27], which separates the major addressing families:

    001 (with instr[29:24] = 001000)    Load/store exclusive
    011 (with instr[24] = 0)            Load register (literal)
    101                                 Load/store pair
    111                                 Load/store register

The register family then splits on instr[24] and instr[21]: a set instr[24] is the scaled
unsigned offset form, otherwise instr[21] chooses between a register offset and the
immediate pre/post/unscaled forms.

instr[26] is V, selecting the SIMD and floating point variants. Those are not modelled,
so they decode as undefined rather than being silently treated as integer accesses.
"""

from armulator.armv8.bits_ops import bit_at, substring
from armulator.armv8.opcodes.concrete.loads_and_stores import (
    LoadFpLiteralA64,
    LoadLiteralA64,
    LoadStoreFpIndexedA64,
    LoadStoreFpPairA64,
    LoadStoreFpRegisterOffsetA64,
    LoadStoreFpUnsignedOffsetA64,
    SimdLoadStoreMultipleA64,
    LoadStoreExclusiveA64,
    LoadStoreIndexedA64,
    LoadStorePairA64,
    LoadStoreRegisterOffsetA64,
    LoadStoreUnsignedOffsetA64,
)


def decode_instruction(instr):
    simd = bit_at(instr, 26)
    family = substring(instr, 29, 27)

    if simd:
        return _decode_simd(instr, family)

    if family == 0b111:
        if bit_at(instr, 24):
            return LoadStoreUnsignedOffsetA64
        if bit_at(instr, 21):
            # Only the 10 form in instr[11:10] is a register offset; the rest are
            # atomic and unscaled-immediate encodings added after ARMv8.0.
            if substring(instr, 11, 10) == 0b10:
                return LoadStoreRegisterOffsetA64
            return None
        return LoadStoreIndexedA64

    if family == 0b101:
        return LoadStorePairA64

    if family == 0b011 and not bit_at(instr, 24):
        return LoadLiteralA64

    if substring(instr, 29, 24) == 0b001000:
        return LoadStoreExclusiveA64

    return None


def _decode_simd(instr, family):
    """
    The SIMD and floating point variants mirror the integer families, minus the
    exclusives (there are no vector exclusives) and the unprivileged forms.
    """
    if family == 0b111:
        if bit_at(instr, 24):
            return LoadStoreFpUnsignedOffsetA64
        if bit_at(instr, 21):
            if substring(instr, 11, 10) == 0b10:
                return LoadStoreFpRegisterOffsetA64
            return None
        return LoadStoreFpIndexedA64

    if family == 0b101:
        return LoadStoreFpPairA64

    if family == 0b011 and not bit_at(instr, 24):
        return LoadFpLiteralA64

    if family == 0b001:
        # Advanced SIMD load/store multiple structures: LD1/ST1 and the de-interleaving
        # LD2/LD3/LD4. Only the contiguous forms are modelled.
        return SimdLoadStoreMultipleA64

    return None
