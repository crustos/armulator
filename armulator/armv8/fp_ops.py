"""
IEEE-754 helpers for AArch64 floating point.

Values are held as integer bit patterns everywhere else in the model, so this module is
the boundary where they become numbers and back again. Python's own floats are binary64,
which matches the double-precision format exactly; single and half precision are obtained
by round-tripping through ``struct``, which applies round-to-nearest-even for us.

The architecture distinguishes signalling from quiet NaNs and specifies exactly which
input NaN propagates through an operation. That matters because firmware sometimes uses
NaN payloads to carry information, so operations here propagate an incoming NaN rather
than always returning a fresh default one.
"""

import math
import struct

__all__ = [
    'FP_FORMATS', 'default_nan', 'is_nan', 'is_signalling_nan', 'is_infinity', 'is_zero',
    'bits_to_float', 'float_to_bits', 'fp_neg', 'fp_abs', 'fp_add', 'fp_sub', 'fp_mul',
    'fp_div', 'fp_sqrt', 'fp_max', 'fp_min', 'fp_max_num', 'fp_min_num', 'fp_compare',
    'fp_convert', 'fp_to_fixed', 'fixed_to_fp', 'fp_round_to_int', 'FPRounding',
]


class FPRounding:
    """FPCR.RMode encodings."""
    TIEEVEN = 0b00      # to nearest, ties to even
    POSINF = 0b01       # toward +infinity
    NEGINF = 0b10       # toward -infinity
    ZERO = 0b11         # toward zero


#: width in bits -> (struct format, exponent bits, mantissa bits)
FP_FORMATS = {
    16: ('e', 5, 10),
    32: ('f', 8, 23),
    64: ('d', 11, 52),
}

#: width in bits -> the architectural default (quiet) NaN
_DEFAULT_NAN = {
    16: 0x7E00,
    32: 0x7FC00000,
    64: 0x7FF8000000000000,
}

_PACK = {16: '<e', 32: '<f', 64: '<d'}
_UNPACK_INT = {16: '<H', 32: '<I', 64: '<Q'}


def default_nan(width: int) -> int:
    return _DEFAULT_NAN[width]


def _exponent_and_mantissa(bits: int, width: int):
    _, exp_bits, mant_bits = FP_FORMATS[width]
    exponent = (bits >> mant_bits) & ((1 << exp_bits) - 1)
    mantissa = bits & ((1 << mant_bits) - 1)
    return exponent, mantissa, exp_bits, mant_bits


def is_nan(bits: int, width: int) -> bool:
    exponent, mantissa, exp_bits, _ = _exponent_and_mantissa(bits, width)
    return exponent == (1 << exp_bits) - 1 and mantissa != 0


def is_signalling_nan(bits: int, width: int) -> bool:
    """
    A NaN whose most significant mantissa bit is clear signals rather than propagating
    quietly.
    """
    if not is_nan(bits, width):
        return False
    _, _, mant_bits = FP_FORMATS[width]
    return not (bits >> (mant_bits - 1)) & 1


def quiet_nan(bits: int, width: int) -> int:
    """Turn a signalling NaN into its quiet counterpart, preserving the payload."""
    _, _, mant_bits = FP_FORMATS[width]
    return bits | (1 << (mant_bits - 1))


def is_infinity(bits: int, width: int) -> bool:
    exponent, mantissa, exp_bits, _ = _exponent_and_mantissa(bits, width)
    return exponent == (1 << exp_bits) - 1 and mantissa == 0


def is_zero(bits: int, width: int) -> bool:
    return (bits & ((1 << (width - 1)) - 1)) == 0


def sign_of(bits: int, width: int) -> int:
    return (bits >> (width - 1)) & 1


def bits_to_float(bits: int, width: int) -> float:
    """
    Reinterpret a bit pattern as a Python float. NaNs survive the trip but lose their
    payload, so callers that care about payloads must check for NaN first.
    """
    packed = struct.pack(_UNPACK_INT[width], bits & ((1 << width) - 1))
    return struct.unpack(_PACK[width], packed)[0]


def float_to_bits(value: float, width: int) -> int:
    """
    Round a Python float into the given format and return its bit pattern. Overflow
    becomes an infinity of the right sign, as IEEE requires under round-to-nearest.
    """
    try:
        packed = struct.pack(_PACK[width], value)
    except OverflowError:
        return _infinity(1 if value < 0 else 0, width)
    return struct.unpack(_UNPACK_INT[width], packed)[0]


def _infinity(sign: int, width: int) -> int:
    _, exp_bits, mant_bits = FP_FORMATS[width]
    return (sign << (width - 1)) | (((1 << exp_bits) - 1) << mant_bits)


def _propagate_nan(width, *operands):
    """
    Pick which NaN an operation returns. A signalling NaN takes priority over a quiet
    one, matching the architecture's NaN propagation rules.
    """
    for bits in operands:
        if is_signalling_nan(bits, width):
            return quiet_nan(bits, width)
    for bits in operands:
        if is_nan(bits, width):
            return bits
    return None


def fp_neg(bits: int, width: int) -> int:
    """Flip the sign bit. Defined even for NaNs, which is why it is not 0 - x."""
    return bits ^ (1 << (width - 1))


def fp_abs(bits: int, width: int) -> int:
    return bits & ((1 << (width - 1)) - 1)


def _binary_op(op, a, b, width):
    nan = _propagate_nan(width, a, b)
    if nan is not None:
        return nan
    x, y = bits_to_float(a, width), bits_to_float(b, width)
    try:
        result = op(x, y)
    except (ValueError, ZeroDivisionError):
        # Invalid operations (inf - inf, 0 * inf, 0 / 0) produce the default NaN.
        return default_nan(width)
    if isinstance(result, float) and math.isnan(result):
        return default_nan(width)
    return float_to_bits(result, width)


def fp_add(a: int, b: int, width: int) -> int:
    return _binary_op(lambda x, y: x + y, a, b, width)


def fp_sub(a: int, b: int, width: int) -> int:
    return _binary_op(lambda x, y: x - y, a, b, width)


def fp_mul(a: int, b: int, width: int) -> int:
    return _binary_op(lambda x, y: x * y, a, b, width)


def fp_div(a: int, b: int, width: int) -> int:
    nan = _propagate_nan(width, a, b)
    if nan is not None:
        return nan
    x, y = bits_to_float(a, width), bits_to_float(b, width)
    if y == 0.0:
        if x == 0.0:
            return default_nan(width)
        # Division by zero gives a correctly signed infinity, not a NaN.
        sign = sign_of(a, width) ^ sign_of(b, width)
        return _infinity(sign, width)
    return _binary_op(lambda p, q: p / q, a, b, width)


def fp_sqrt(bits: int, width: int) -> int:
    nan = _propagate_nan(width, bits)
    if nan is not None:
        return nan
    value = bits_to_float(bits, width)
    if value < 0.0:
        return default_nan(width)
    if is_zero(bits, width):
        # Preserves the sign, so sqrt(-0.0) is -0.0.
        return bits
    return float_to_bits(math.sqrt(value), width)


def fp_max(a: int, b: int, width: int) -> int:
    nan = _propagate_nan(width, a, b)
    if nan is not None:
        return nan
    x, y = bits_to_float(a, width), bits_to_float(b, width)
    if x == y == 0.0:
        # +0 and -0 compare equal, so the sign decides.
        return a if sign_of(a, width) == 0 else b
    return a if x > y else b


def fp_min(a: int, b: int, width: int) -> int:
    nan = _propagate_nan(width, a, b)
    if nan is not None:
        return nan
    x, y = bits_to_float(a, width), bits_to_float(b, width)
    if x == y == 0.0:
        return a if sign_of(a, width) == 1 else b
    return a if x < y else b


def fp_max_num(a: int, b: int, width: int) -> int:
    """FMAXNM: a NaN operand is ignored rather than propagated."""
    if is_nan(a, width) and not is_nan(b, width):
        return b
    if is_nan(b, width) and not is_nan(a, width):
        return a
    return fp_max(a, b, width)


def fp_min_num(a: int, b: int, width: int) -> int:
    if is_nan(a, width) and not is_nan(b, width):
        return b
    if is_nan(b, width) and not is_nan(a, width):
        return a
    return fp_min(a, b, width)


def fp_compare(a: int, b: int, width: int):
    """
    Compare two values and return NZCV. Unordered (either operand NaN) gives 0b0011,
    which is what makes an unordered comparison fail both the LT and GT conditions.
    """
    if is_nan(a, width) or is_nan(b, width):
        return 0b0011
    x, y = bits_to_float(a, width), bits_to_float(b, width)
    if x == y:
        return 0b0110          # Z and C
    if x < y:
        return 0b1000          # N
    return 0b0010              # C


def fp_convert(bits: int, from_width: int, to_width: int) -> int:
    """
    FCVT between precisions. A NaN keeps its quietness but is re-encoded in the target
    format, since the payload width differs.
    """
    if is_nan(bits, from_width):
        return default_nan(to_width)
    if is_infinity(bits, from_width):
        return _infinity(sign_of(bits, from_width), to_width)
    return float_to_bits(bits_to_float(bits, from_width), to_width)


def _round_with_mode(value: float, rounding: int) -> float:
    if rounding == FPRounding.ZERO:
        return math.trunc(value)
    if rounding == FPRounding.POSINF:
        return math.ceil(value)
    if rounding == FPRounding.NEGINF:
        return math.floor(value)
    # Round to nearest, ties to even - which is what Python's round() does for floats.
    return float(round(value))


def fp_round_to_int(bits: int, width: int, rounding: int) -> int:
    """FRINT*: round to an integral value but stay in floating point."""
    nan = _propagate_nan(width, bits)
    if nan is not None:
        return nan
    if is_infinity(bits, width) or is_zero(bits, width):
        return bits
    return float_to_bits(_round_with_mode(bits_to_float(bits, width), rounding), width)


def fp_to_fixed(bits: int, fp_width: int, int_width: int, unsigned: bool,
                fbits: int = 0, rounding: int = FPRounding.ZERO) -> int:
    """
    FCVTZS/FCVTZU and the fixed-point forms.

    Out-of-range values saturate rather than wrapping, and a NaN converts to zero - both
    are architectural requirements that differ from what a C cast would do.
    """
    if is_nan(bits, fp_width):
        return 0
    value = bits_to_float(bits, fp_width) * (2 ** fbits)

    if math.isinf(value):
        rounded = value
    else:
        rounded = _round_with_mode(value, rounding)

    if unsigned:
        low, high = 0, (1 << int_width) - 1
    else:
        low, high = -(1 << (int_width - 1)), (1 << (int_width - 1)) - 1

    if rounded != rounded or rounded < low:      # NaN already handled; this is saturation
        rounded = low
    elif rounded > high:
        rounded = high
    return int(rounded) & ((1 << int_width) - 1)


def fixed_to_fp(value: int, int_width: int, fp_width: int, unsigned: bool,
                fbits: int = 0) -> int:
    """SCVTF/UCVTF, including the fixed-point forms."""
    if not unsigned and value >> (int_width - 1):
        value -= 1 << int_width
    return float_to_bits(value / (2 ** fbits), fp_width)
