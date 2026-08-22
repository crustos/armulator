"""
Bit manipulation primitives for AArch64.

The ARMv6 implementation is already width-parameterized, so the shared primitives are
re-exported here rather than duplicated. Only the operations that AArch64 adds (or that
need 64-bit defaults) are defined in this module.
"""

from armulator.armv6.bits_ops import (  # noqa: F401
    add,
    sub,
    align,
    bit_at,
    bit_count,
    bit_not,
    big_endian_reverse,
    chain,
    is_ones,
    lower_chunk,
    set_bit_at,
    set_substring,
    sign_extend,
    substring,
    to_signed,
    to_unsigned,
)
from armulator.armv6.bits_ops import add_with_carry as _add_with_carry

__all__ = [
    'add', 'sub', 'align', 'bit_at', 'bit_count', 'bit_not', 'big_endian_reverse', 'chain',
    'is_ones', 'lower_chunk', 'set_bit_at', 'set_substring', 'sign_extend', 'substring',
    'to_signed', 'to_unsigned', 'add_with_carry', 'ones', 'zeros', 'zero_extend', 'replicate',
    'ror', 'lsl', 'lsr', 'asr', 'ror_c', 'lsl_c', 'lsr_c', 'asr_c', 'shift_reg_value',
    'highest_set_bit', 'lowest_set_bit', 'count_leading_zeros', 'count_leading_sign_bits',
    'reverse_bits', 'decode_bit_masks', 'extend', 'decode_reg_extend',
]


def add_with_carry(x: int, y: int, carry_in: int, size: int = 64):
    """
    AArch64 defaults to 64-bit datapath, unlike the ARMv6 helper which defaults to 32.
    """
    return _add_with_carry(x, y, carry_in, size)


def ones(length: int) -> int:
    """
    Ones(N) - an N-bit value with every bit set.
    """
    return (1 << length) - 1 if length > 0 else 0


def zeros(length: int) -> int:
    """
    Zeros(N) - present for symmetry with the pseudocode.
    """
    return 0


def zero_extend(x: int, dst_length: int) -> int:
    """
    ZeroExtend(x, N). Values are held as non-negative python ints, so this only needs to mask.
    """
    return lower_chunk(x, dst_length)


def replicate(x: int, src_length: int, dst_length: int) -> int:
    """
    Replicate(x, N) - repeat the src_length-bit pattern until dst_length bits are filled.
    """
    assert dst_length % src_length == 0, 'Replicate requires an integral number of copies'
    result = 0
    for _ in range(dst_length // src_length):
        result = (result << src_length) | lower_chunk(x, src_length)
    return result


def lsl_c(x: int, length: int, shift: int):
    """
    LSL_C(x, shift) -> (result, carry_out). shift must be > 0.
    """
    assert shift > 0, 'LSL_C requires a shift greater than zero'
    extended = x << shift
    carry_out = bit_at(extended, length)
    return lower_chunk(extended, length), carry_out


def lsl(x: int, length: int, shift: int) -> int:
    assert shift >= 0
    if shift == 0:
        return lower_chunk(x, length)
    return lsl_c(x, length, shift)[0]


def lsr_c(x: int, length: int, shift: int):
    """
    LSR_C(x, shift) -> (result, carry_out). shift must be > 0.
    """
    assert shift > 0, 'LSR_C requires a shift greater than zero'
    x = lower_chunk(x, length)
    carry_out = bit_at(x, shift - 1) if shift <= length else 0
    return x >> shift, carry_out


def lsr(x: int, length: int, shift: int) -> int:
    assert shift >= 0
    if shift == 0:
        return lower_chunk(x, length)
    return lsr_c(x, length, shift)[0]


def asr_c(x: int, length: int, shift: int):
    """
    ASR_C(x, shift) -> (result, carry_out). shift must be > 0.
    """
    assert shift > 0, 'ASR_C requires a shift greater than zero'
    signed = to_signed(x, length)
    result = to_unsigned(signed >> shift, length)
    carry_out = bit_at(x, shift - 1) if shift <= length else bit_at(x, length - 1)
    return result, carry_out


def asr(x: int, length: int, shift: int) -> int:
    assert shift >= 0
    if shift == 0:
        return lower_chunk(x, length)
    return asr_c(x, length, shift)[0]


def ror_c(x: int, length: int, shift: int):
    """
    ROR_C(x, shift) -> (result, carry_out). shift must be > 0.
    """
    assert shift > 0, 'ROR_C requires a shift greater than zero'
    m = shift % length
    result = lower_chunk(lsr(x, length, m) | lsl(x, length, length - m), length)
    carry_out = bit_at(result, length - 1)
    return result, carry_out


def ror(x: int, length: int, shift: int) -> int:
    """
    ROR(x, shift) - rotate right, shift may be zero.
    """
    if shift % length == 0:
        return lower_chunk(x, length)
    return ror_c(x, length, shift)[0]


def shift_reg_value(value: int, shift_type, amount: int, length: int) -> int:
    """
    ShiftReg() body - apply a decoded shift type to an already-read register value.
    Takes the ShiftType enum from armulator.armv8.enums.
    """
    from armulator.armv8.enums import ShiftType
    if amount == 0:
        return lower_chunk(value, length)
    if shift_type == ShiftType.LSL:
        return lsl(value, length, amount)
    if shift_type == ShiftType.LSR:
        return lsr(value, length, amount)
    if shift_type == ShiftType.ASR:
        return asr(value, length, amount)
    return ror(value, length, amount)


def highest_set_bit(x: int, length: int = 64) -> int:
    """
    HighestSetBit(x) - index of the most significant set bit, or -1 when x is zero.
    """
    x = lower_chunk(x, length)
    return x.bit_length() - 1


def lowest_set_bit(x: int, length: int = 64) -> int:
    """
    LowestSetBit(x) - index of the least significant set bit, or `length` when x is zero.
    """
    x = lower_chunk(x, length)
    if x == 0:
        return length
    return (x & -x).bit_length() - 1


def count_leading_zeros(x: int, length: int = 64) -> int:
    """
    CountLeadingZeroBits(x).
    """
    return length - 1 - highest_set_bit(x, length)


def count_leading_sign_bits(x: int, length: int = 64) -> int:
    """
    CountLeadingSignBits(x) - leading bits equal to the sign bit, excluding the sign bit itself.
    """
    upper = substring(x, length - 1, 1)
    lower = lower_chunk(x, length - 1)
    return count_leading_zeros(upper ^ lower, length - 1)


def reverse_bits(x: int, length: int = 64) -> int:
    """
    RBIT - reverse the bit order of a length-bit value.
    """
    result = 0
    for i in range(length):
        result = (result << 1) | bit_at(x, i)
    return result


def decode_bit_masks(m: int, imm_n: int, imms: int, immr: int, immediate: bool):
    """
    DecodeBitMasks(M, immN, imms, immr, immediate) -> (wmask, tmask).

    Decodes the compact logical-immediate encoding shared by AND/ORR/EOR/ANDS (immediate)
    and the bitfield instructions. Returns None when the encoding is reserved, so callers
    can raise UNDEFINED.
    """
    # len = HighestSetBit(immN:NOT(imms))
    concat = chain(imm_n, bit_not(imms, 6), 6)
    length = highest_set_bit(concat, 7)
    if length < 1:
        return None
    if m < (1 << length):
        return None
    levels = zero_extend(ones(length), 6)
    # For logical immediates an all-ones imms field is reserved.
    if immediate and (imms & levels) == levels:
        return None
    s = imms & levels
    r = immr & levels
    diff = sub(s, r, 6)
    esize = 1 << length
    d = substring(diff, length - 1, 0)
    welem = zero_extend(ones(s + 1), esize)
    telem = zero_extend(ones(d + 1), esize)
    wmask = replicate(ror(welem, esize, r), esize, m)
    tmask = replicate(telem, esize, m)
    return wmask, tmask


def extend(x: int, src_length: int, dst_length: int, unsigned: bool) -> int:
    """
    Extend(x, N, unsigned).
    """
    if unsigned:
        return zero_extend(x, dst_length)
    return sign_extend(x, src_length, dst_length)


def decode_reg_extend(value: int, extend_type, shift: int, length: int) -> int:
    """
    ExtendReg() body - extend a register value according to the extend type, then shift left.
    Takes the ExtendType enum from armulator.armv8.enums.
    """
    from armulator.armv8.enums import ExtendType
    assert 0 <= shift <= 4, 'ExtendReg shift amount must be 0..4'
    src_lengths = {
        ExtendType.SXTB: 8, ExtendType.SXTH: 16, ExtendType.SXTW: 32, ExtendType.SXTX: 64,
        ExtendType.UXTB: 8, ExtendType.UXTH: 16, ExtendType.UXTW: 32, ExtendType.UXTX: 64,
    }
    unsigned = extend_type in (ExtendType.UXTB, ExtendType.UXTH, ExtendType.UXTW, ExtendType.UXTX)
    src_length = min(src_lengths[extend_type], length - shift)
    return lower_chunk(extend(lower_chunk(value, src_length), src_length, length, unsigned) << shift, length)
