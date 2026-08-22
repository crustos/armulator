"""
Concrete encodings for the scalar floating point instructions.

The ``ftype`` field at instr[23:22] names the precision: 00 single, 01 double, 11 half.
Half precision arithmetic needs ARMv8.2, which the Cortex-A57 does not have, so ftype 11
is accepted only by FCVT (a conversion, not an arithmetic operation).
"""

from armulator.armv8 import fp_ops
from armulator.armv8.bits_ops import bit_at, chain, substring
from armulator.armv8.opcodes.abstract_opcodes.fp_data_processing import (
    FpCompare,
    FpConditionalCompare,
    FpConditionalSelect,
    FpDataProcessing1Source,
    FpDataProcessing2Source,
    FpDataProcessing3Source,
    FpImmediate,
)
from armulator.armv8.opcodes.abstract_opcodes.fp_integer_convert import FpIntegerConvert

#: instr[23:22] -> precision in bits. 10 is unallocated.
FTYPE_SIZES = {0b00: 32, 0b01: 64, 0b11: 16}

_ONE_SOURCE = {
    0b000000: 'fmov', 0b000001: 'fabs', 0b000010: 'fneg', 0b000011: 'fsqrt',
    0b001000: 'frintn', 0b001001: 'frintp', 0b001010: 'frintm', 0b001011: 'frintz',
    0b001100: 'frinta', 0b001110: 'frintx', 0b001111: 'frinti',
}

_TWO_SOURCE = {
    0b0000: 'fmul', 0b0001: 'fdiv', 0b0010: 'fadd', 0b0011: 'fsub',
    0b0100: 'fmax', 0b0101: 'fmin', 0b0110: 'fmaxnm', 0b0111: 'fminnm',
    0b1000: 'fnmul',
}

#: (rmode, opcode) -> the conversion an FP/integer move performs.
_CONVERT_ROUNDING = {
    0b00: fp_ops.FPRounding.TIEEVEN,   # FCVTN*
    0b01: fp_ops.FPRounding.POSINF,    # FCVTP*
    0b10: fp_ops.FPRounding.NEGINF,    # FCVTM*
    0b11: fp_ops.FPRounding.ZERO,      # FCVTZ*
}


def _ftype_size(instr, allow_half=False):
    size = FTYPE_SIZES.get(substring(instr, 23, 22))
    if size == 16 and not allow_half:
        # Half precision arithmetic requires ARMv8.2; the A57 does not implement it.
        return None
    return size


def expand_fp_immediate(imm8: int, width: int) -> int:
    """
    Expand the eight-bit FMOV immediate.

    The encoding is sign : exponent-ish : mantissa, spelt out in the architecture as
    a sign bit, an inverted top exponent bit replicated to fill, three more exponent
    bits and a four-bit mantissa. It reaches values like 1.0, 2.0, 0.5 and -1.5.
    """
    _, exp_bits, mant_bits = fp_ops.FP_FORMATS[width]
    sign = bit_at(imm8, 7)
    exp_high = bit_at(imm8, 6)
    exp_low = substring(imm8, 5, 4)
    mantissa = substring(imm8, 3, 0)

    # NOT(exp_high) followed by exp_high repeated fills the exponent field.
    exponent = chain(1 - exp_high, 0, exp_bits - 1)
    if exp_high:
        exponent |= ((1 << (exp_bits - 3)) - 1) << 2
    exponent |= exp_low

    return (sign << (width - 1)) | (exponent << mant_bits) | (mantissa << (mant_bits - 4))


class FpDataProcessing1SourceA64(FpDataProcessing1Source):
    """
    M 0 S 1 1 1 1 0 ftype 1 opcode 1 0 0 0 0 Rn Rd
    """

    @staticmethod
    def from_bitarray(instr, processor):
        opcode = substring(instr, 20, 15)
        source_size = FTYPE_SIZES.get(substring(instr, 23, 22))
        if source_size is None:
            return None

        # FCVT is encoded as 0001xx, with the low two bits naming the destination type.
        if substring(opcode, 5, 2) == 0b0001:
            dst_size = FTYPE_SIZES.get(substring(opcode, 1, 0))
            if dst_size is None or dst_size == source_size:
                return None
            return FpDataProcessing1SourceA64(
                instr, d=substring(instr, 4, 0), n=substring(instr, 9, 5),
                operation='fcvt', datasize=source_size, dstsize=dst_size,
            )

        operation = _ONE_SOURCE.get(opcode)
        if operation is None or source_size == 16:
            return None
        return FpDataProcessing1SourceA64(
            instr, d=substring(instr, 4, 0), n=substring(instr, 9, 5),
            operation=operation, datasize=source_size,
        )


class FpDataProcessing2SourceA64(FpDataProcessing2Source):
    """
    M 0 S 1 1 1 1 0 ftype 1 Rm opcode 1 0 Rn Rd
    """

    @staticmethod
    def from_bitarray(instr, processor):
        size = _ftype_size(instr)
        if size is None:
            return None
        operation = _TWO_SOURCE.get(substring(instr, 15, 12))
        if operation is None:
            return None
        return FpDataProcessing2SourceA64(
            instr, d=substring(instr, 4, 0), n=substring(instr, 9, 5),
            m=substring(instr, 20, 16), operation=operation, datasize=size,
        )


class FpDataProcessing3SourceA64(FpDataProcessing3Source):
    """
    M 0 S 1 1 1 1 1 ftype o1 Rm o0 Ra Rn Rd
    """

    @staticmethod
    def from_bitarray(instr, processor):
        size = _ftype_size(instr)
        if size is None:
            return None
        o1 = bit_at(instr, 21)
        o0 = bit_at(instr, 15)
        # FMADD  (0,0): Ra + Rn*Rm      FMSUB  (0,1): Ra - Rn*Rm
        # FNMADD (1,0): -Ra - Rn*Rm     FNMSUB (1,1): -Ra + Rn*Rm
        return FpDataProcessing3SourceA64(
            instr, d=substring(instr, 4, 0), n=substring(instr, 9, 5),
            m=substring(instr, 20, 16), a=substring(instr, 14, 10),
            negate_product=bool(o1 ^ o0), negate_addend=bool(o1), datasize=size,
        )


class FpCompareA64(FpCompare):
    """
    M 0 S 1 1 1 1 0 ftype 1 Rm op 1 0 0 0 Rn opcode2
    """

    @staticmethod
    def from_bitarray(instr, processor):
        size = _ftype_size(instr)
        if size is None:
            return None
        opcode2 = substring(instr, 4, 0)
        # Bit 3 of opcode2 selects the compare-with-zero forms.
        if opcode2 not in (0b00000, 0b01000, 0b10000, 0b11000):
            return None
        return FpCompareA64(
            instr, n=substring(instr, 9, 5), m=substring(instr, 20, 16),
            compare_with_zero=bool(bit_at(opcode2, 3)), datasize=size,
        )


class FpConditionalSelectA64(FpConditionalSelect):
    """
    M 0 S 1 1 1 1 0 ftype 1 Rm cond 1 1 Rn Rd
    """

    @staticmethod
    def from_bitarray(instr, processor):
        size = _ftype_size(instr)
        if size is None:
            return None
        return FpConditionalSelectA64(
            instr, d=substring(instr, 4, 0), n=substring(instr, 9, 5),
            m=substring(instr, 20, 16), condition=substring(instr, 15, 12), datasize=size,
        )


class FpConditionalCompareA64(FpConditionalCompare):
    """
    M 0 S 1 1 1 1 0 ftype 1 Rm cond 0 1 Rn op nzcv
    """

    @staticmethod
    def from_bitarray(instr, processor):
        size = _ftype_size(instr)
        if size is None:
            return None
        return FpConditionalCompareA64(
            instr, n=substring(instr, 9, 5), m=substring(instr, 20, 16),
            condition=substring(instr, 15, 12), flags=substring(instr, 3, 0), datasize=size,
        )


class FpImmediateA64(FpImmediate):
    """
    M 0 S 1 1 1 1 0 ftype 1 imm8 1 0 0 imm5 Rd
    """

    @staticmethod
    def from_bitarray(instr, processor):
        size = _ftype_size(instr)
        if size is None:
            return None
        if substring(instr, 9, 5) != 0:
            return None
        return FpImmediateA64(
            instr, d=substring(instr, 4, 0),
            imm=expand_fp_immediate(substring(instr, 20, 13), size), datasize=size,
        )


class FpIntegerConvertA64(FpIntegerConvert):
    """
    sf 0 S 1 1 1 1 0 ftype 1 rmode opcode 0 0 0 0 0 0 Rn Rd

    Covers FMOV (general), SCVTF/UCVTF and the FCVT*S/U family.
    """

    @staticmethod
    def from_bitarray(instr, processor):
        int_size = 64 if bit_at(instr, 31) else 32
        ftype = substring(instr, 23, 22)
        rmode = substring(instr, 20, 19)
        opcode = substring(instr, 18, 16)

        # FMOV between a general register and the top half of a vector needs ftype 10.
        top_half = ftype == 0b10
        if top_half:
            if int_size != 64 or rmode != 0b01 or opcode not in (0b110, 0b111):
                return None
            return FpIntegerConvertA64(
                instr, d=substring(instr, 4, 0), n=substring(instr, 9, 5),
                operation='fmov', fp_size=64, int_size=64, unsigned=False,
                to_general=(opcode == 0b110), top_half=True,
            )

        fp_size = FTYPE_SIZES.get(ftype)
        if fp_size is None:
            return None

        if rmode == 0b00 and opcode in (0b110, 0b111):
            # FMOV Rd, Vn / FMOV Vd, Rn - a reinterpretation, not a conversion.
            if fp_size == 64 and int_size != 64:
                return None
            if fp_size == 32 and int_size != 32:
                return None
            return FpIntegerConvertA64(
                instr, d=substring(instr, 4, 0), n=substring(instr, 9, 5),
                operation='fmov', fp_size=fp_size, int_size=int_size, unsigned=False,
                to_general=(opcode == 0b110),
            )

        if rmode == 0b00 and opcode in (0b010, 0b011):
            # SCVTF / UCVTF
            return FpIntegerConvertA64(
                instr, d=substring(instr, 4, 0), n=substring(instr, 9, 5),
                operation='cvtf', fp_size=fp_size, int_size=int_size,
                unsigned=(opcode == 0b011), to_general=False,
            )

        if opcode in (0b000, 0b001):
            # FCVT{N,P,M,Z}{S,U} - the rounding mode comes from rmode.
            return FpIntegerConvertA64(
                instr, d=substring(instr, 4, 0), n=substring(instr, 9, 5),
                operation='fcvt_int', fp_size=fp_size, int_size=int_size,
                unsigned=(opcode == 0b001), to_general=True,
                rounding=_CONVERT_ROUNDING[rmode],
            )

        if rmode == 0b00 and opcode in (0b100, 0b101):
            # FCVTAS / FCVTAU - round to nearest with ties away from zero. Modelled as
            # ties-to-even, which differs only on exact halfway values.
            return FpIntegerConvertA64(
                instr, d=substring(instr, 4, 0), n=substring(instr, 9, 5),
                operation='fcvt_int', fp_size=fp_size, int_size=int_size,
                unsigned=(opcode == 0b101), to_general=True,
                rounding=fp_ops.FPRounding.TIEEVEN,
            )

        return None
