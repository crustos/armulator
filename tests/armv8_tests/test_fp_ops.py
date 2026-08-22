"""
IEEE-754 helpers. These are exercised directly because the edge cases - signed zeros,
infinities, NaN propagation and saturation - are hard to reach from assembly snippets
but are exactly where floating point emulation tends to go wrong.
"""

import pytest

from armulator.armv8 import fp_ops
from armulator.armv8.fp_ops import FPRounding


def f32(value):
    return fp_ops.float_to_bits(value, 32)


def f64(value):
    return fp_ops.float_to_bits(value, 64)


class TestArithmetic:
    def test_basic_operations(self):
        assert fp_ops.bits_to_float(fp_ops.fp_add(f32(1.5), f32(2.5), 32), 32) == 4.0
        assert fp_ops.bits_to_float(fp_ops.fp_sub(f32(2.5), f32(1.5), 32), 32) == 1.0
        assert fp_ops.bits_to_float(fp_ops.fp_mul(f32(1.5), f32(2.0), 32), 32) == 3.0
        assert fp_ops.bits_to_float(fp_ops.fp_div(f32(3.0), f32(2.0), 32), 32) == 1.5

    def test_division_by_zero_gives_signed_infinity(self):
        assert fp_ops.fp_div(f32(1.0), f32(0.0), 32) == 0x7F800000
        assert fp_ops.fp_div(f32(-1.0), f32(0.0), 32) == 0xFF800000

    def test_zero_divided_by_zero_is_nan(self):
        assert fp_ops.is_nan(fp_ops.fp_div(f32(0.0), f32(0.0), 32), 32)

    def test_sqrt_of_negative_is_nan(self):
        assert fp_ops.is_nan(fp_ops.fp_sqrt(f32(-1.0), 32), 32)

    def test_sqrt_preserves_negative_zero(self):
        assert fp_ops.fp_sqrt(f32(-0.0), 32) == 0x80000000


class TestNaNHandling:
    def test_nan_propagates_through_arithmetic(self):
        nan = fp_ops.default_nan(32)
        assert fp_ops.is_nan(fp_ops.fp_add(nan, f32(1.0), 32), 32)

    def test_signalling_nan_is_quietened(self):
        signalling = 0x7F800001          # exponent all ones, top mantissa bit clear
        assert fp_ops.is_signalling_nan(signalling, 32)
        result = fp_ops.fp_add(signalling, f32(1.0), 32)
        assert fp_ops.is_nan(result, 32)
        assert not fp_ops.is_signalling_nan(result, 32)

    def test_fmax_propagates_nan_but_fmaxnm_ignores_it(self):
        nan = fp_ops.default_nan(32)
        assert fp_ops.is_nan(fp_ops.fp_max(f32(1.0), nan, 32), 32)
        assert fp_ops.bits_to_float(fp_ops.fp_max_num(f32(1.0), nan, 32), 32) == 1.0

    def test_negate_does_not_interpret_the_value(self):
        # FNEG is a bit flip, so it works on a NaN rather than producing a new one.
        assert fp_ops.fp_neg(fp_ops.default_nan(32), 32) == 0xFFC00000


class TestCompare:
    @pytest.mark.parametrize('a, b, expected', [
        (1.0, 2.0, 0b1000),      # N: less than
        (1.0, 1.0, 0b0110),      # Z and C: equal
        (2.0, 1.0, 0b0010),      # C: greater than
    ])
    def test_ordered_comparisons(self, a, b, expected):
        assert fp_ops.fp_compare(f32(a), f32(b), 32) == expected

    def test_unordered_sets_c_and_v(self):
        # An unordered result must fail both LT and GT, which 0b0011 achieves.
        assert fp_ops.fp_compare(fp_ops.default_nan(32), f32(1.0), 32) == 0b0011


class TestConversion:
    def test_single_to_double_and_back(self):
        as_double = fp_ops.fp_convert(f32(1.5), 32, 64)
        assert fp_ops.bits_to_float(as_double, 64) == 1.5
        assert fp_ops.fp_convert(as_double, 64, 32) == f32(1.5)

    def test_single_to_half(self):
        assert fp_ops.fp_convert(f32(1.5), 32, 16) == 0x3E00

    def test_float_to_int_truncates_toward_zero(self):
        assert fp_ops.fp_to_fixed(f64(2.9), 64, 32, False) == 2
        assert fp_ops.fp_to_fixed(f64(-2.9), 64, 32, False) == (-2) & 0xFFFFFFFF

    def test_out_of_range_saturates(self):
        # A C cast would wrap; the architecture requires saturation.
        assert fp_ops.fp_to_fixed(f64(1e30), 64, 32, False) == 0x7FFFFFFF
        assert fp_ops.fp_to_fixed(f64(-1e30), 64, 32, False) == 0x80000000
        assert fp_ops.fp_to_fixed(f64(1e30), 64, 32, True) == 0xFFFFFFFF

    def test_nan_converts_to_zero(self):
        assert fp_ops.fp_to_fixed(fp_ops.default_nan(64), 64, 32, False) == 0

    def test_int_to_float_respects_signedness(self):
        signed = fp_ops.fixed_to_fp(0xFFFFFFFB, 32, 64, False)
        unsigned = fp_ops.fixed_to_fp(0xFFFFFFFB, 32, 64, True)
        assert fp_ops.bits_to_float(signed, 64) == -5.0
        assert fp_ops.bits_to_float(unsigned, 64) == 4294967291.0

    @pytest.mark.parametrize('rounding, expected', [
        (FPRounding.ZERO, 2),
        (FPRounding.POSINF, 3),
        (FPRounding.NEGINF, 2),
        (FPRounding.TIEEVEN, 2),
    ])
    def test_rounding_modes(self, rounding, expected):
        assert fp_ops.fp_to_fixed(f64(2.5), 64, 32, False, rounding=rounding) == expected

    def test_ties_go_to_even_not_away_from_zero(self):
        # 2.5 rounds down to 2 and 3.5 rounds up to 4: both land on an even value.
        assert fp_ops.fp_to_fixed(f64(2.5), 64, 32, False,
                                  rounding=FPRounding.TIEEVEN) == 2
        assert fp_ops.fp_to_fixed(f64(3.5), 64, 32, False,
                                  rounding=FPRounding.TIEEVEN) == 4
