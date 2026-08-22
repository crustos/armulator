"""
Advanced SIMD - the subset a compiler reaches for on integer and floating point vectors.

Modelled here: the three-same arithmetic, comparison and bitwise operations, the vector
immediates (MOVI/MVNI), DUP, INS, UMOV and SMOV. Not modelled: the load/store structure
instructions (LD1-LD4), pairwise and across-lane reductions, saturating arithmetic,
widening and narrowing forms, table lookups and the by-element multiplies. Those decode
as undefined rather than executing incorrectly.

Throughout, Q at instr[30] selects between a 64-bit and a 128-bit operation, and ``size``
at instr[23:22] gives the lane width as 8 << size bits.
"""

from armulator.armv8.bits_ops import bit_at, replicate, substring
from armulator.armv8.opcodes.abstract_opcodes.simd_move import (
    SimdDuplicate,
    SimdExtract,
    SimdImmediate,
    SimdInsert,
)
from armulator.armv8.opcodes.abstract_opcodes.simd_reduce import (
    SimdAcrossLanes,
    SimdPairwise,
)
from armulator.armv8.opcodes.abstract_opcodes.simd_three_same import (
    SimdBitwise,
    SimdThreeSame,
)

#: opcode at instr[16:12] -> across-lane reduction, keyed with U.
_ACROSS_LANES = {
    (0, 0b11011): 'addv',
    (0, 0b01010): 'smaxv', (1, 0b01010): 'umaxv',
    (0, 0b11010): 'sminv', (1, 0b11010): 'uminv',
}

#: (U, opcode) -> integer operation for the three-same group.
_THREE_SAME_INTEGER = {
    (0, 0b10000): 'add', (1, 0b10000): 'sub',
    (0, 0b10011): 'mul',
    (0, 0b00001): 'sqadd', (1, 0b00001): 'uqadd',
    (0, 0b00101): 'sqsub', (1, 0b00101): 'uqsub',
    (0, 0b10111): 'addp',
    (0, 0b00110): 'cmgt', (0, 0b00111): 'cmge',
    (1, 0b00110): 'cmhi', (1, 0b00111): 'cmhs',
    (1, 0b10001): 'cmeq',
    (0, 0b01100): 'smax', (0, 0b01101): 'smin',
    (1, 0b01100): 'umax', (1, 0b01101): 'umin',
}

#: (U, op, opcode) -> floating point operation for the three-same group.
#:
#: In the floating point forms the size field splits: instr[23] is part of the operation
#: selector while instr[22] alone gives the precision. Keying on the whole size field
#: would need a duplicate entry per precision and silently miss the double forms.
_THREE_SAME_FLOAT = {
    (0, 0, 0b11010): 'fadd', (0, 1, 0b11010): 'fsub',
    (1, 0, 0b11011): 'fmul', (1, 0, 0b11111): 'fdiv',
    (0, 0, 0b11110): 'fmax', (0, 1, 0b11110): 'fmin',
}

#: (U, size) -> bitwise operation. These ignore the lane width entirely.
_BITWISE = {
    (0, 0b00): 'and', (0, 0b01): 'bic', (0, 0b10): 'orr', (0, 0b11): 'orn',
    (1, 0b00): 'eor',
}


def expand_simd_immediate(op, cmode, imm8):
    """
    Expand the MOVI/MVNI immediate into the 64-bit pattern that fills the register.

    ``cmode`` selects how the eight bits are spread: replicated into each byte, placed
    at a shifted position within each halfword or word, or - for the ``op``/``cmode``
    combination used by MOVI Dn - expanded one bit per byte into a 64-bit mask.
    """
    cmode_high = substring(cmode, 3, 1)

    if not bit_at(cmode, 3):
        # 0xxx: an 8-bit value shifted within each 32-bit lane.
        shift = 8 * substring(cmode, 2, 1)
        return replicate(imm8 << shift, 32, 64)
    if substring(cmode, 3, 2) == 0b10:
        # 10x0 / 10x1: shifted within each 16-bit lane.
        shift = 8 * bit_at(cmode, 1)
        return replicate(imm8 << shift, 16, 64)
    if cmode_high == 0b110:
        # 110x: shifted into a 32-bit lane with the low bits set to one.
        shift = 8 if bit_at(cmode, 0) else 0
        value = (imm8 << (8 + shift)) | ((1 << (8 + shift)) - 1)
        return replicate(value & 0xFFFFFFFF, 32, 64)
    if cmode == 0b1110 and op == 0:
        # Replicate the byte into all eight lanes.
        return replicate(imm8, 8, 64)
    if cmode == 0b1110 and op == 1:
        # MOVI Dn: each bit of imm8 becomes a whole byte of zeros or ones.
        result = 0
        for index in range(8):
            if bit_at(imm8, index):
                result |= 0xFF << (8 * index)
        return result
    if cmode == 0b1111:
        from armulator.armv8.opcodes.concrete.floating_point import expand_fp_immediate
        if op == 0:
            # FMOV (vector) - a single precision constant in every 32-bit lane.
            return replicate(expand_fp_immediate(imm8, 32), 32, 64)
        # FMOV (vector, double precision) - one constant per 64-bit lane.
        return expand_fp_immediate(imm8, 64)
    return None


class SimdThreeSameA64(SimdThreeSame):
    """
    0 Q U 0 1 1 1 0 size 1 Rm opcode 1 Rn Rd
    """

    @staticmethod
    def from_bitarray(instr, processor):
        q = bit_at(instr, 30)
        u = bit_at(instr, 29)
        size = substring(instr, 23, 22)
        opcode = substring(instr, 15, 11)
        datasize = 128 if q else 64

        float_op = _THREE_SAME_FLOAT.get((u, bit_at(size, 1), opcode))
        if float_op is not None:
            # Only the low bit of the size field is the precision selector.
            element_size = 64 if bit_at(size, 0) else 32
            if element_size == 64 and not q:
                return None
            return SimdThreeSameA64(
                instr, d=substring(instr, 4, 0), n=substring(instr, 9, 5),
                m=substring(instr, 20, 16), operation=float_op,
                element_size=element_size, elements=datasize // element_size,
                datasize=datasize, is_float=True,
            )

        operation = _THREE_SAME_INTEGER.get((u, opcode))
        if operation is None:
            return None
        element_size = 8 << size
        # A 64-bit operation on 64-bit lanes would be a single element; only ADD, SUB
        # and the comparisons allow it, and never in the 64-bit vector form.
        if element_size == 64 and not q:
            return None
        if operation == 'mul' and element_size == 64:
            return None
        return SimdThreeSameA64(
            instr, d=substring(instr, 4, 0), n=substring(instr, 9, 5),
            m=substring(instr, 20, 16), operation=operation,
            element_size=element_size, elements=datasize // element_size,
            datasize=datasize,
        )


class SimdPairwiseA64(SimdPairwise):
    """
    0 Q 0 0 1 1 1 0 size 1 Rm 1 0 1 1 1 1 Rn Rd - ADDP.
    """

    @staticmethod
    def from_bitarray(instr, processor):
        size = substring(instr, 23, 22)
        q = bit_at(instr, 30)
        element_size = 8 << size
        datasize = 128 if q else 64
        if element_size == 64 and not q:
            return None
        return SimdPairwiseA64(
            instr, d=substring(instr, 4, 0), n=substring(instr, 9, 5),
            m=substring(instr, 20, 16), element_size=element_size,
            elements=datasize // element_size, datasize=datasize,
        )


class SimdAcrossLanesA64(SimdAcrossLanes):
    """
    0 Q U 0 1 1 1 0 size 1 1 0 0 0 opcode 1 0 Rn Rd
    """

    @staticmethod
    def from_bitarray(instr, processor):
        operation = _ACROSS_LANES.get((bit_at(instr, 29), substring(instr, 16, 12)))
        if operation is None:
            return None
        size = substring(instr, 23, 22)
        q = bit_at(instr, 30)
        element_size = 8 << size
        datasize = 128 if q else 64
        # There is no reduction over 64-bit lanes: a 2D vector would reduce two
        # elements, which the architecture leaves to ADDP instead.
        if element_size == 64:
            return None
        if size == 0b10 and not q:
            return None
        return SimdAcrossLanesA64(
            instr, d=substring(instr, 4, 0), n=substring(instr, 9, 5),
            operation=operation, element_size=element_size,
            elements=datasize // element_size,
        )


class SimdBitwiseA64(SimdBitwise):
    """
    0 Q U 0 1 1 1 0 size 1 Rm 0 0 0 1 1 1 Rn Rd
    """

    @staticmethod
    def from_bitarray(instr, processor):
        operation = _BITWISE.get((bit_at(instr, 29), substring(instr, 23, 22)))
        if operation is None:
            return None
        return SimdBitwiseA64(
            instr, d=substring(instr, 4, 0), n=substring(instr, 9, 5),
            m=substring(instr, 20, 16), operation=operation,
            datasize=128 if bit_at(instr, 30) else 64,
        )


class SimdImmediateA64(SimdImmediate):
    """
    0 Q op 0 1 1 1 1 0 0 0 0 0 abc cmode o2 1 defgh Rd
    """

    @staticmethod
    def from_bitarray(instr, processor):
        op = bit_at(instr, 29)
        cmode = substring(instr, 15, 12)
        imm8 = (substring(instr, 18, 16) << 5) | substring(instr, 9, 5)
        imm64 = expand_simd_immediate(op, cmode, imm8)
        if imm64 is None:
            return None
        # With op set the shifted forms are MVNI rather than MOVI, but cmode 1110 and
        # 1111 are the MOVI Dn and double precision FMOV encodings, which do not invert.
        invert = bool(op) and cmode not in (0b1110, 0b1111)
        return SimdImmediateA64(
            instr, d=substring(instr, 4, 0), imm64=imm64,
            datasize=128 if bit_at(instr, 30) else 64, invert=invert,
        )


def _index_and_size(imm5):
    """
    Decode the imm5 field shared by DUP, INS, UMOV and SMOV: the position of its lowest
    set bit gives the lane width, and the bits above it give the lane index.
    """
    for size_log in range(4):
        if bit_at(imm5, size_log):
            element_size = 8 << size_log
            index = imm5 >> (size_log + 1)
            return index, element_size
    return None, None


class SimdDuplicateA64(SimdDuplicate):
    """
    0 Q op 0 1 1 1 0 0 0 0 imm5 0 0 0 0 op2 1 Rn Rd - DUP from a vector lane or a
    general purpose register.
    """

    @staticmethod
    def from_bitarray(instr, processor):
        index, element_size = _index_and_size(substring(instr, 20, 16))
        if element_size is None:
            return None
        q = bit_at(instr, 30)
        datasize = 128 if q else 64
        if element_size == 64 and not q:
            return None
        from_general = bit_at(instr, 11)
        return SimdDuplicateA64(
            instr, d=substring(instr, 4, 0), n=substring(instr, 9, 5), index=index,
            element_size=element_size, elements=datasize // element_size,
            datasize=datasize, from_general=bool(from_general),
        )


class SimdInsertA64(SimdInsert):
    """
    0 1 op 0 1 1 1 0 0 0 0 imm5 0 imm4 1 Rn Rd
    """

    @staticmethod
    def from_bitarray(instr, processor):
        if not bit_at(instr, 30):
            return None
        dst_index, element_size = _index_and_size(substring(instr, 20, 16))
        if element_size is None:
            return None
        from_general = not bit_at(instr, 29)
        src_index = 0
        if not from_general:
            # INS (element) takes its source lane from imm4, scaled by the lane width.
            size_log = (element_size.bit_length() - 4)
            src_index = substring(instr, 14, 11) >> size_log
        return SimdInsertA64(
            instr, d=substring(instr, 4, 0), n=substring(instr, 9, 5),
            dst_index=dst_index, src_index=src_index, element_size=element_size,
            from_general=from_general,
        )


class SimdExtractA64(SimdExtract):
    """
    0 Q 0 0 1 1 1 0 0 0 0 imm5 0 0 op 1 1 1 Rn Rd - UMOV and SMOV.
    """

    @staticmethod
    def from_bitarray(instr, processor):
        index, element_size = _index_and_size(substring(instr, 20, 16))
        if element_size is None:
            return None
        signed = substring(instr, 14, 11) == 0b0101
        q = bit_at(instr, 30)
        regsize = 64 if q else 32
        if not signed:
            # UMOV of a doubleword only exists in the 64-bit form.
            if element_size == 64 and not q:
                return None
            if element_size != 64 and q:
                return None
        elif element_size >= regsize:
            return None
        return SimdExtractA64(
            instr, d=substring(instr, 4, 0), n=substring(instr, 9, 5), index=index,
            element_size=element_size, regsize=regsize, signed=signed,
        )


from armulator.armv8.opcodes.abstract_opcodes.simd_shift_immediate import (  # noqa: E402
    SimdShiftImmediate,
)

#: (U, opcode) -> shift operation. Only the plain shifts are modelled; the rounding,
#: accumulating, saturating and narrowing variants are not.
_SHIFT_OPS = {(0, 0b00000): 'sshr', (1, 0b00000): 'ushr', (0, 0b01010): 'shl'}


class SimdShiftImmediateA64(SimdShiftImmediate):
    """
    0 Q U 0 1 1 1 1 0 immh immb opcode 1 Rn Rd   (vector, instr[28] clear)
    0 1 U 1 1 1 1 1 0 immh immb opcode 1 Rn Rd   (scalar, instr[28] set)

    immh does double duty: the position of its highest set bit gives the lane width, and
    immh:immb together give the shift amount relative to that width.
    """

    @staticmethod
    def from_bitarray(instr, processor):
        immh = substring(instr, 22, 19)
        if immh == 0:
            # immh of zero is the modified-immediate space, not a shift.
            return None
        operation = _SHIFT_OPS.get((bit_at(instr, 29), substring(instr, 15, 11)))
        if operation is None:
            return None

        # The highest set bit of immh selects the lane width.
        element_size = 8 << (immh.bit_length() - 1)
        immhb = (immh << 3) | substring(instr, 18, 16)

        scalar = bit_at(instr, 28)
        if scalar:
            # Scalar shifts only exist for doubleword lanes.
            if element_size != 64:
                return None
            elements, datasize = 1, 64
        else:
            q = bit_at(instr, 30)
            datasize = 128 if q else 64
            if element_size == 64 and not q:
                return None
            elements = datasize // element_size

        if operation == 'shl':
            shift = immhb - element_size
        else:
            shift = 2 * element_size - immhb
        if not 0 <= shift < element_size:
            return None

        return SimdShiftImmediateA64(
            instr, d=substring(instr, 4, 0), n=substring(instr, 9, 5), shift=shift,
            operation=operation, element_size=element_size, elements=elements,
            datasize=datasize,
        )


from armulator.armv8.opcodes.abstract_opcodes.simd_by_element import (  # noqa: E402
    SimdByElement,
)

#: (U, opcode) -> floating point by-element operation.
_BY_ELEMENT_FP = {(0, 0b0001): 'fmla', (0, 0b0101): 'fmls', (0, 0b1001): 'fmul'}


class SimdByElementA64(SimdByElement):
    """
    0 Q U 0 1 1 1 1 1 sz L M Rm opcode H 0 Rn Rd

    The lane index is assembled from H, L and (for narrow elements) M, which is why it
    cannot simply be read out of one field.
    """

    @staticmethod
    def from_bitarray(instr, processor):
        operation = _BY_ELEMENT_FP.get((bit_at(instr, 29), substring(instr, 15, 12)))
        if operation is None:
            return None
        # Bit 23 set marks the floating point by-element forms.
        if not bit_at(instr, 23):
            return None

        double = bit_at(instr, 22)
        q = bit_at(instr, 30)
        datasize = 128 if q else 64
        h = bit_at(instr, 11)
        low = bit_at(instr, 21)
        m_bit = bit_at(instr, 20)

        if double:
            if not q or low:
                return None
            element_size = 64
            index = h
            m = (m_bit << 4) | substring(instr, 19, 16)
        else:
            element_size = 32
            index = (h << 1) | low
            m = (m_bit << 4) | substring(instr, 19, 16)

        return SimdByElementA64(
            instr, d=substring(instr, 4, 0), n=substring(instr, 9, 5), m=m, index=index,
            operation=operation, element_size=element_size,
            elements=datasize // element_size, datasize=datasize,
        )
